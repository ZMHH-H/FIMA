import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from moco.loader import Augment_GPU_pre
from moco.loss import Dense_Contrastive_Loss
from moco.loader_utils import Gen_CAAM_MSAK,Gen_Static_Diff
from utils.analysis_utils import show_video_frame,show_caam_mask,show_caam
from backbone import transformer
# import os
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, Dense_T=0.3, pos_ratio=0.7, crop_size=224, motion_patch_ratio=0.5):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        # self.dense_K = 1568 # 49*32

        print('pos_ratio',pos_ratio)
        print('Dense_T',Dense_T)
        self.aug_gpu = Augment_GPU_pre(crop_size)
        self.dense_loss = Dense_Contrastive_Loss(pos_ratio = pos_ratio, temperature = Dense_T)
        self.caam_mask = Gen_CAAM_MSAK(crop_size=crop_size)
        self.gen_static_diff = Gen_Static_Diff(motion_patch_ratio=motion_patch_ratio)
        # create the encoders
        # num_classes is the output fc dimension
        # 初始化query encoder(encoder)和key encoder(momentum encoder)
        # dense and global loss share same projector
        self.encoder_q = base_encoder(with_classifier=False, projection=False)
        self.encoder_k = base_encoder(with_classifier=False, projection=False)

        

        if base_encoder.__name__ == "I3D":
            self.projector_dense = Projection_Head_Conv(input_dim=1024, projection_hidden_size=1024, output_dim=128)
            self.projector_dense_k = Projection_Head_Conv(input_dim=1024, projection_hidden_size=1024, output_dim=128)
                        
            self.embedding1 = nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0, bias=True)
            self.embedding2 = nn.Conv2d(512, 1024, kernel_size=1, stride=1, padding=0, bias=True)
            self.avg_pool = torch.nn.AdaptiveAvgPool3d((1,1,1))

            transformer_layer = transformer.TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=512,dropout=0.1,batch_first=False,norm_first=True)
            self.transformer_decoder = transformer.TransformerDecoder(transformer_layer, num_layers=2)
            self.pos_embedding = transformer.PositionalEncoding(512, max_len=100)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))   # S,N,E (patchs,batch,embedding)

        elif base_encoder.__name__ == 'R2PLUS1D' or base_encoder.__name__ == 'R3D':
            self.projector_dense = Projection_Head_Conv(input_dim=512, projection_hidden_size=1024, output_dim=128)
            self.projector_dense_k = Projection_Head_Conv(input_dim=512, projection_hidden_size=1024, output_dim=128)

            self.embedding1 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)
            self.embedding2 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=True)

            self.avg_pool = torch.nn.AdaptiveAvgPool3d((1,1,1))

            transformer_layer = transformer.TransformerDecoderLayer(d_model=512, nhead=4, dim_feedforward=512,dropout=0.1,batch_first=False,norm_first=True)
            self.transformer_decoder = transformer.TransformerDecoder(transformer_layer, num_layers=2)
            self.pos_embedding = transformer.PositionalEncoding(512, max_len=100)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 512))   # S,N,E (patchs,batch,embedding)
        else: 
            raise NotImplementedError("dense projector not implemented")
            
        # 使用query encoder的参数初始化key encoder
        # 并关闭key encoder的梯度，使用动量更新
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector_dense.parameters(), self.projector_dense_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # create the queue(memory bank)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0) #l2标准化
        # 初始化queue pointer为0
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
         # create the queue(memory bank)
        self.register_buffer("motion_queue", torch.randn(dim, K))
        self.slow_queue = nn.functional.normalize(self.motion_queue, dim=0) #l2标准化
        # 初始化queue pointer为0
        self.register_buffer("motion_queue_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # Todo: cosine annealing momentum strategy
        
        # momentum update key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        
        # momentum update dense projector
        for param_q, param_k in zip(self.projector_dense.parameters(), self.projector_dense_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # queue是 dimx65536(Cx65536), all_gather后得到的keys是batchsize x dim
        # 所以要将keys转置后再enqueue
        # Tensor.T: Returns a view of this tensor with its dimensions reversed(转置)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr # 将pointer位置存储到buffer中

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        
        ptr = int(self.motion_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # queue是 dimx65536(Cx65536), all_gather后得到的keys是batchsize x dim
        # 所以要将keys转置后再enqueue
        # Tensor.T: Returns a view of this tensor with its dimensions reversed(转置)
        self.motion_queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.motion_queue_ptr[0] = ptr # 将pointer位置存储到buffer中

    @torch.no_grad()
    def _dequeue_and_enqueue_dense(self, dense_keys):
        # gather keys before updating queue
        dense_keys = concat_all_gather(dense_keys)
        B,C,H,W = dense_keys.shape
        feat_number = B*H*W

        dense_keys = dense_keys.permute(0,2,3,1).reshape(-1,C)

        ptr = int(self.queue_dense_ptr)
        assert self.dense_K % feat_number == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # queue是 dimx65536(Cx65536), all_gather后得到的keys是batchsize x dim
        # 所以要将keys转置后再enqueue
        # Tensor.T: Returns a view of this tensor with its dimensions reversed(转置)
        self.queue_dense[:, ptr:ptr + feat_number] = dense_keys.T
        ptr = (ptr + feat_number) % self.dense_K  # move pointer

        self.queue_dense_ptr[0] = ptr # 将pointer位置存储到buffer中

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        将所有gpu中的video all_gather起来再重新分布
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        # torch.randperm(): Returns a random permutation of integers from 0 to n - 1.
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx] #1 x batch_size_all -> num_gpu x (batch_size_all/num_gpu)

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def momentum_encoder_forward(self, m_clip, d_clip, m_clip_l1, m_clip_l2):
        # 使用key_encoder对正样本对中的key计算特征
        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            m_clip, idx_unshuffle = self._batch_shuffle_ddp(m_clip.contiguous())
            d_clip, diff_idx_unshuffle = self._batch_shuffle_ddp(d_clip.contiguous())
            m_clip_l1, local1_idx_unshuffle = self._batch_shuffle_ddp(m_clip_l1.contiguous())
            m_clip_l2, local2_idx_unshuffle = self._batch_shuffle_ddp(m_clip_l2.contiguous())

            # key features
            # mk_base:[bs,C,T,7,7]  m_k: [bs,C,1,1,1]
            mk_base,m_k = self.encoder_k(m_clip)         
            m_k = self.projector_dense_k(m_k)                           # m_k: [bs,128,1,1]
            m_k = m_k.view(m_k.size(0),m_k.size(1))                     # m_k: [bs,128]
            
            # diff_k features
            dk_base = self.encoder_k(d_clip,'early_return')                     # dk_base: [bs,C,T,7,7]
            B,C,T,H,W = dk_base.shape
            dk_dense = self.projector_dense_k(dk_base.reshape(B,C,1,T*H*W,1))   # dk_dense:[bs,128,2*7*7,1]
           
            # local1 key features
            mk_base_l1, m_k_l1 = self.encoder_k(m_clip_l1)          # mk_base_l1:[bs,C,T,7,7] m_k_l1:[bs,C,1,1,1]
            m_k_l1 = self.projector_dense_k(m_k_l1)                 # m_k_l1: [bs,128,1,1]
            m_k_l1 = m_k_l1.view(m_k_l1.size(0),m_k_l1.size(1))     # m_k_l1: [bs,128]

            # local2 key features
            mk_base_l2, m_k_l2 = self.encoder_k(m_clip_l2)          # mk_base_l2:[bs,C,T,7,7] m_k_l2:[bs,C,1,1,1]
            m_k_l2 = self.projector_dense_k(m_k_l2)                 # m_k_l2: [bs,128,1,1]
            m_k_l2 = m_k_l2.view(m_k_l2.size(0),m_k_l2.size(1))     # m_k_l2: [bs,128]

            # normalize features
            m_k = nn.functional.normalize(m_k, dim=1)
            dk_dense = nn.functional.normalize(dk_dense, dim=1)
            m_k_l1 = nn.functional.normalize(m_k_l1, dim=1)
            m_k_l2 = nn.functional.normalize(m_k_l2, dim=1)

            # undo shuffle
            # 将一个mini batch中的key features unshuffle回来，便于后续infonce loss计算相似度
            m_k = self._batch_unshuffle_ddp(m_k, idx_unshuffle)
            mk_base = self._batch_unshuffle_ddp(mk_base, idx_unshuffle)
            dk_base = self._batch_unshuffle_ddp(dk_base, diff_idx_unshuffle)
            dk_dense = self._batch_unshuffle_ddp(dk_dense, diff_idx_unshuffle)
            m_k_l1 = self._batch_unshuffle_ddp(m_k_l1, local1_idx_unshuffle)
            m_k_l2 = self._batch_unshuffle_ddp(m_k_l2, local2_idx_unshuffle)

            # reshape diff_k features
            dk_dense = dk_dense.reshape(B,-1,T,H,W)     # dk_dense:[bs,128,2,7,7]
            dk_dense = dk_dense.chunk(T,dim=2) # dk_dense_temporal: tuple([bs,C,1,7,7],[bs,C,1,7,7])

        return mk_base, dk_base, m_k, dk_dense, m_k_l1, m_k_l2

    def query_encoder_forward(self, vi_m):
        """
        Input:
            vi_m: query clips [bs,C,T,H,W]
            n_k: key clips RGB global features [bs,128]
            dnk_dense: key frame difference clip dense features [bs,128,7,7]
            coord_m: transformation coordinations of im_q
            coord_n: transformation coordinations of im_k
            motion_mask_m: motion mask of im_q
            motion_mask_n: motion mask of im_k
        Output:
            logits, targets, dense_loss
        """
        # 使用query_encoder对正样本对中的query计算特征
        # compute query features
        # mq_base_former:[bs,C/2,2T,14,14]  mq_base:[bs,C,T,7,7]  m_q: [bs,C,1,1,1]
        mq_base,m_q = self.encoder_q(vi_m)  

        m_q = self.projector_dense(m_q)                                 # m_q:[bs,128,1,1]
        m_q = m_q.view(m_q.size(0),m_q.size(1))                         # q:[bs,128]
        # mq_dense = self.projector_dense(mq_base.reshape(B,C,1,T*H*W,1))   

        mq_dense = self.embedding1(mq_base.flatten(2).unsqueeze(-1))    # mq_dense:[bs,512,2*7*7,1]
        mq_dense = mq_dense.flatten(2).permute(2,0,1)                   # mq_dense:[2*7*7,bs,C]

        
        # l2 normalize
        m_q = nn.functional.normalize(m_q, dim=1)         
        # mq_dense = nn.functional.normalize(mq_dense, dim=1)

        # mq_dense = mq_dense.reshape(B,-1,T,H,W)             # mq_dense:[bs,128,2,7,7]
        # mq_dense_temporal = mq_dense.chunk(T,dim=2)         # mq_dense_temporal: tuple([bs,C,1,7,7],[bs,C,1,7,7])
        
        return m_q, mq_dense
        
    def compute_logits(self, m_q, n_k):
        # compute logits, Einstein sum is more intuitive
        # torch.einsum('nc,nc->n', [q, k])：矩阵对应位置相乘，然后在c维度上相加,相当于对batch中的每一个q,k计算点积
        # q,k: [bs,128]
        l_pos = torch.einsum('nc,nc->n', [m_q, n_k]).unsqueeze(-1)                  # [bsx1]
        
        # torch.einsum('nc,ck->nk',..)：矩阵相乘，相当于对batch中的每一个q，与queue中的负样本计算点积相似度
        # queue: [128,K] (128x65536)
        l_neg = torch.einsum('nc,ck->nk', [m_q, self.queue.clone().detach()])     #[bs,K]

        # logits中第一列为正样本对相似度，剩下K列为正样本与负样本的相似度
        logits = torch.cat([l_pos, l_neg], dim=1)                               # [bs,K+1]

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        # infonce loss将对比学习看作一个多分类任务
        # logits中的第一列为正样本对的相似度，按照infonce loss它应将正样本分类为标签0（因为它在第一列）
        # 所以所有正样本的标签都该为0，设置为batch_size个0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(n_k)

        return logits, labels
    
    def compute_logits_local(self, m_q_l1, n_k_l1, n_k_l2, m_k_l1, m_k_l2):
        # compute logits, Einstein sum is more intuitive
        # torch.einsum('nc,nc->n', [q, k])：矩阵对应位置相乘，然后在c维度上相加,相当于对batch中的每一个q,k计算点积
        # q,k: [bs,128]
        l_pos = torch.einsum('nc,nc->n', [m_q_l1, n_k_l1]).unsqueeze(-1)                  # [bsx1]
        
        # torch.einsum('nc,ck->nk',..)：矩阵相乘，相当于对batch中的每一个q，与queue中的负样本计算点积相似度
        # queue: [128,K] (128x65536)
        neg_feat = torch.cat([n_k_l2, m_k_l1, m_k_l2],dim=0) # [3bs,128]
        #[bs,3bs+K]
        l_neg = torch.einsum('nc,ck->nk', [m_q_l1, torch.cat([neg_feat.permute(1,0),self.motion_queue.clone().detach()],dim=1)])

        # logits中第一列为正样本对相似度，剩下K列为正样本与负样本的相似度
        logits = torch.cat([l_pos, l_neg], dim=1)                               # [bs,3bs+1]

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        # infonce loss将对比学习看作一个多分类任务
        # logits中的第一列为正样本对的相似度，按照infonce loss它应将正样本分类为标签0（因为它在第一列）
        # 所以所有正样本的标签都该为0，设置为batch_size个0
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels

    def forward(self, vi_m, vi_m_l1, vi_m_l2, vi_n, vi_n_l1, vi_n_l2, coord_m, coord_n):
        # m,n: indices of video
        # q,k; encoder_q, encoder_k

        # generate frame difference video
        diff_m = self.gen_static_diff(vi_m,'diff')
        diff_n = self.gen_static_diff(vi_n,'diff')

        diff_m_l1 = self.gen_static_diff(vi_m_l1,'diff')
        diff_m_l2 = self.gen_static_diff(vi_m_l2,'diff')

        diff_n_l1 = self.gen_static_diff(vi_n_l1,'diff')
        diff_n_l2 = self.gen_static_diff(vi_n_l2,'diff')

        # ------------SHOW VIDEO FRAME-----------------
        # for i in range(vi_m.shape[0]):
        #     show_video_frame(vi_m[i],'video_0')
        #     show_video_frame(vi_m_1x[i],'video_1')
        #     show_video_frame(vi_n[i],'video_2',)
        #     show_video_frame(diff_m[i],'video_3',)
        #     show_video_frame(diff_n[i],'video_4',)
        # ---------------SHOW CAAM---------------------
        # for i in range(vi_m.shape[0]):
        #     show_caam(vi_m[i],'video_0',self.encoder_q)
        #     show_caam(vi_n[i],'video_1',self.encoder_q)
            # show_caam(vi_m_l1[i],'video_0',self.encoder_q)
            # show_caam(vi_m_l2[i],'video_1',self.encoder_q)
            # show_caam(diff_m_l1[i],'video_2',self.encoder_q)
            # show_caam(diff_m_l2[i],'video_3',self.encoder_q)
        # -------------SHOW CAAM MASK------------------
        # for i in range(vi_m.shape[0]):
        #     show_caam_mask(vi_m[i],vi_m[i],'video_0',self.encoder_q)
        #     show_caam_mask(vi_m[i],diff_m[i],'video_1',self.encoder_q)
        #     show_caam_mask(vi_n[i],vi_n[i],'video_2',self.encoder_q)

        #  Data augmentation
        vi_m = self.aug_gpu(vi_m)
        vi_n = self.aug_gpu(vi_n)

        diff_m_l1 = self.aug_gpu(diff_m_l1)
        diff_m_l2 = self.aug_gpu(diff_m_l2)

        diff_n_l1 = self.aug_gpu(diff_n_l1)
        diff_n_l2 = self.aug_gpu(diff_n_l2)

        diff_m_k = self.aug_gpu(diff_m)
        diff_n_k = self.aug_gpu(diff_n)

        # update key encoder
        self._momentum_update_key_encoder()

        # momentum encoder forward
        # mk_base:[bs,C,T,7,7]
        # dmk_base:[bs,C,T,7,7]
        # m_k:[bs,128] 
        # dmk_dense:tuple([bs,128,1,7,7],[bs,128,1,7,7])
        # m_k_l1:[bs,128]
        # m_k_l2:[bs,128]
        mk_base, dmk_base, m_k, dmk_dense, m_k_l1, m_k_l2 = self.momentum_encoder_forward(vi_m, diff_m_k, diff_m_l1, diff_m_l2)
        nk_base, dnk_base, n_k, dnk_dense, n_k_l1, n_k_l2 = self.momentum_encoder_forward(vi_n, diff_n_k, diff_n_l1, diff_n_l2)

        B, C_base, T_, H, W = mk_base.shape # [bs,C,2,7,7]

        # generate foreground mask
        # vi_m, motion_mask_m = self.gen_static_diff(vi_m,'mask_fuse')
        # vi_n, motion_mask_n = self.gen_static_diff(vi_n,'mask_fuse')
        motion_mask_m = self.caam_mask(mk_base, dmk_base)
        motion_mask_n = self.caam_mask(nk_base, dnk_base)

        # for i in range(vi_m.shape[0]):
        #     show_video_frame(vi_m[i],'video_0')
        #     show_video_frame(vi_n[i],'video_1')
        
        # query encoder forward
        m_q, mq_dense = self.query_encoder_forward(vi_m)    # m_q: [bs,128], mq_dense:[2*7*7,bs,C]
        n_q, nq_dense = self.query_encoder_forward(vi_n)
        

        # ====================== inter modailties prediction =========================
        # should add spatio positional embeddings every feature frame
        mq_dense1 = self.pos_embedding(mq_dense[:H*W,:,:].contiguous()) # [49,bs,512]
        mq_dense2 = self.pos_embedding(mq_dense[H*W:,:,:].contiguous()) # [49,bs,512]
        nq_dense1 = self.pos_embedding(nq_dense[:H*W,:,:].contiguous()) # [49,bs,512]
        nq_dense2 = self.pos_embedding(nq_dense[H*W:,:,:].contiguous()) # [49,bs,512]

        # expand cls token
        cls_tokens = self.cls_token.expand(-1, B, -1)  # patches, bs, embedding_dim [1,bs,512]

        # cross modailties prediction
        # mq_dense 需要从nq_dense中寻找到对应关系，然后使用nq_dense信息来预测diff feature信息
        pred_dnk_dense1 = self.transformer_decoder(torch.cat([cls_tokens,mq_dense1],dim=0), nq_dense1)    # pred_dnk_dense: [1+7*7,bs,512]
        pred_dnk_dense2 = self.transformer_decoder(torch.cat([cls_tokens,mq_dense2],dim=0), nq_dense2)    # pred_dnk_dense: [1+7*7,bs,512]
        pred_dmk_dense1 = self.transformer_decoder(torch.cat([cls_tokens,nq_dense1],dim=0), mq_dense1)    # pred_dmk_dense: [1+7*7,bs,512]
        pred_dmk_dense2 = self.transformer_decoder(torch.cat([cls_tokens,nq_dense2],dim=0), mq_dense2)    # pred_dmk_dense: [1+7*7,bs,512]

        # projcet back to 1024 dimsion
        pred_dnk_dense1 = self.embedding2(pred_dnk_dense1.permute(1,2,0).unsqueeze(-1)) # pred_dnk_dense: [bs,C,1+7*7,1]
        pred_dnk_dense2 = self.embedding2(pred_dnk_dense2.permute(1,2,0).unsqueeze(-1)) # pred_dnk_dense: [bs,C,1+7*7,1]
        pred_dmk_dense1 = self.embedding2(pred_dmk_dense1.permute(1,2,0).unsqueeze(-1)) # pred_dnk_dense: [bs,C,1+7*7,1]
        pred_dmk_dense2 = self.embedding2(pred_dmk_dense2.permute(1,2,0).unsqueeze(-1)) # pred_dnk_dense: [bs,C,1+7*7,1]

        # ========= predict dense and local feature ============
        # reshape prediction and project to subspace
        pred_dnk_dense1 = pred_dnk_dense1.reshape(B,C_base,1,-1,1) # pred_dnk_dense: [bs,C,1,1+7*7,1]
        pred_dnk_dense2 = pred_dnk_dense2.reshape(B,C_base,1,-1,1) # pred_dnk_dense: [bs,C,1,1+7*7,1]
        pred_dmk_dense1 = pred_dmk_dense1.reshape(B,C_base,1,-1,1) # pred_dmk_dense: [bs,C,1,1+7*7,1]
        pred_dmk_dense2 = pred_dmk_dense2.reshape(B,C_base,1,-1,1) # pred_dmk_dense: [bs,C,1,1+7*7,1]


        pred_dnk_dense1 = self.projector_dense(pred_dnk_dense1)   # pred_dnk_dense: [bs,128,1+7*7,1]
        pred_dnk_dense2 = self.projector_dense(pred_dnk_dense2)   # pred_dnk_dense: [bs,128,1+7*7,1]
        pred_dmk_dense1 = self.projector_dense(pred_dmk_dense1)   # pred_dmk_dense: [bs,128,1+7*7,1]
        pred_dmk_dense2 = self.projector_dense(pred_dmk_dense2)   # pred_dmk_dense: [bs,128,1+7*7,1]

        # print(pred_dmk_dense.device)
        # l2 normalize
        pred_dnk_dense1 = nn.functional.normalize(pred_dnk_dense1, dim=1)
        pred_dnk_dense2 = nn.functional.normalize(pred_dnk_dense2, dim=1)
        pred_dmk_dense1 = nn.functional.normalize(pred_dmk_dense1, dim=1)
        pred_dmk_dense2 = nn.functional.normalize(pred_dmk_dense2, dim=1)
        
        # seperate cls_tokens and pred_dense features
        # local features predictions
        pred_nk_l1 = pred_dnk_dense1[:,:,:1,:]   # pred_nk_l1:[bs,128,1,1]
        pred_nk_l2 = pred_dnk_dense2[:,:,:1,:]   # pred_nk_l2:[bs,128,1,1]
        pred_mk_l1 = pred_dmk_dense1[:,:,:1,:]   # pred_mk_l1:[bs,128,1,1]
        pred_mk_l2 = pred_dmk_dense2[:,:,:1,:]   # pred_mk_l2:[bs,128,1,1]
        # reshape
        pred_nk_l1 = pred_nk_l1.view(pred_nk_l1.size(0),pred_nk_l1.size(1))    # pred_nk_l1:[bs,128]
        pred_nk_l2 = pred_nk_l2.view(pred_nk_l2.size(0),pred_nk_l2.size(1))    # pred_nk_l2:[bs,128]
        pred_mk_l1 = pred_mk_l1.view(pred_mk_l1.size(0),pred_mk_l1.size(1))    # pred_mk_l1:[bs,128]
        pred_mk_l2 = pred_mk_l2.view(pred_mk_l2.size(0),pred_mk_l2.size(1))    # pred_mk_l2:[bs,128]

        # dense features predictions
        pred_dnk_dense1 = pred_dnk_dense1[:,:,1:,:]     # pred_dnk_dense1:[bs,128,7*7,1]
        pred_dnk_dense2 = pred_dnk_dense2[:,:,1:,:]     # pred_dnk_dense2:[bs,128,7*7,1]
        pred_dmk_dense1 = pred_dmk_dense1[:,:,1:,:]     # pred_dmk_dense1:[bs,128,7*7,1]
        pred_dmk_dense2 = pred_dmk_dense2[:,:,1:,:]     # pred_dmk_dense2:[bs,128,7*7,1]
        # reshape
        pred_dnk_dense1 = pred_dnk_dense1.reshape(B,-1,H,W) # [bs,128,7,7]
        pred_dnk_dense2 = pred_dnk_dense2.reshape(B,-1,H,W) # [bs,128,7,7]
        pred_dmk_dense1 = pred_dmk_dense1.reshape(B,-1,H,W) # [bs,128,7,7]
        pred_dmk_dense2 = pred_dmk_dense2.reshape(B,-1,H,W) # [bs,128,7,7]


        # # # =================predict local features=================
        # # mq_pool = self.avg_pool(mq_dense.permute(1,2,0).reshape(B,-1,T_,H,W))  # mq_pool:[bs,512,1,1,1] 
        # # nq_pool = self.avg_pool(nq_dense.permute(1,2,0).reshape(B,-1,T_,H,W))  # nq_pool:[bs,512,1,1,1] 
        # cls_tokens = self.cls_token.expand(-1, B, -1)  # patches, bs, embedding_dim
        # # 汇集q_dense features的信息
        # pred_nk_l1 = self.transformer_decoder(cls_tokens, nq_dense1)    # pred_nk_l1: [1,bs,512]
        # pred_nk_l2 = self.transformer_decoder(cls_tokens, nq_dense2)    # pred_nk_l2: [1,bs,512]
        # pred_mk_l1 = self.transformer_decoder(cls_tokens, mq_dense1)    # pred_mk_l1: [1,bs,512]
        # pred_mk_l2 = self.transformer_decoder(cls_tokens, mq_dense2)    # pred_mk_l2: [1,bs,512]
        
        # # projct back to 1024 dim
        # pred_nk_l1 = self.embedding2(pred_nk_l1.permute(1,2,0).unsqueeze(-1)) # pred_nk_l1: [bs,C,1,1]
        # pred_nk_l2 = self.embedding2(pred_nk_l2.permute(1,2,0).unsqueeze(-1)) # pred_nk_l2: [bs,C,1,1]
        # pred_mk_l1 = self.embedding2(pred_mk_l1.permute(1,2,0).unsqueeze(-1)) # pred_mk_l1: [bs,C,1,1]
        # pred_mk_l2 = self.embedding2(pred_mk_l2.permute(1,2,0).unsqueeze(-1)) # pred_mk_l2: [bs,C,1,1]

        # pred_nk_l1 = self.projector_dense(pred_nk_l1.unsqueeze(-1))   # pred_nk_l1: [bs,128,1,1]
        # pred_nk_l2 = self.projector_dense(pred_nk_l2.unsqueeze(-1))   # pred_nk_l2: [bs,128,1,1]
        # pred_mk_l1 = self.projector_dense(pred_mk_l1.unsqueeze(-1))   # pred_mk_l1: [bs,128,1,1]
        # pred_mk_l2 = self.projector_dense(pred_mk_l2.unsqueeze(-1))   # pred_mk_l2: [bs,128,1,1]

        # # print(pred_dmk_dense.device)
        # # l2 normalize 
        # pred_nk_l1 = nn.functional.normalize(pred_nk_l1, dim=1)
        # pred_nk_l2 = nn.functional.normalize(pred_nk_l2, dim=1)
        # pred_mk_l1 = nn.functional.normalize(pred_mk_l1, dim=1)
        # pred_mk_l2 = nn.functional.normalize(pred_mk_l2, dim=1)
        
        # pred_nk_l1 = pred_nk_l1.view(pred_nk_l1.size(0),pred_nk_l1.size(1))    # pred_nk_l1:[bs,128]
        # pred_nk_l2 = pred_nk_l2.view(pred_nk_l2.size(0),pred_nk_l2.size(1))    # pred_nk_l2:[bs,128]
        # pred_mk_l1 = pred_mk_l1.view(pred_mk_l1.size(0),pred_mk_l1.size(1))    # pred_mk_l1:[bs,128]
        # pred_mk_l2 = pred_mk_l2.view(pred_mk_l2.size(0),pred_mk_l2.size(1))    # pred_mk_l2:[bs,128]

        # compute logits
        logits_1, labels_1 = self.compute_logits(m_q,n_k)
        logits_2, labels_2 = self.compute_logits(n_q,m_k)

        # compute local RGB prediction loss
        logits_local_n1, labels_local_n1 = self.compute_logits_local(pred_nk_l1, n_k_l1, n_k_l2, m_k_l1, m_k_l2)
        logits_local_n2, labels_local_n2 = self.compute_logits_local(pred_nk_l2, n_k_l2, n_k_l1, m_k_l1, m_k_l2)
        logits_local_m1, labels_local_m1 = self.compute_logits_local(pred_mk_l1, m_k_l1, m_k_l2, n_k_l1, n_k_l2)
        logits_local_m2, labels_local_m2 = self.compute_logits_local(pred_mk_l2, m_k_l2, m_k_l1, n_k_l1, n_k_l2)
        
        # deque and enque motion features
        # 2*[bs,128]->[2bs,128]->enque
        self._dequeue_and_enqueue_motion(torch.cat([n_k_l1,n_k_l2,m_k_l1,m_k_l2],dim=0))
        
        # compute dense loss
        loss_dense_n1=self.dense_loss(pred_dnk_dense1, dnk_dense[0].squeeze(2), dnk_dense[1].squeeze(2), 
                                     coord_m, coord_n, motion_mask_m, motion_mask_n)
        loss_dense_n2=self.dense_loss(pred_dnk_dense2, dnk_dense[1].squeeze(2), dnk_dense[0].squeeze(2), 
                                     coord_m, coord_n, motion_mask_m, motion_mask_n)                             
        loss_dense_m1=self.dense_loss(pred_dmk_dense1, dmk_dense[0].squeeze(2), dmk_dense[1].squeeze(2), 
                                     coord_n, coord_m, motion_mask_n, motion_mask_m)
        loss_dense_m2=self.dense_loss(pred_dmk_dense2, dmk_dense[1].squeeze(2), dmk_dense[0].squeeze(2), 
                                     coord_n, coord_m, motion_mask_n, motion_mask_m)
        loss_dense_1 = (loss_dense_n1+loss_dense_n2)/2
        loss_dense_2 = (loss_dense_m1+loss_dense_m2)/2
        
        return logits_1, labels_1, logits_2, labels_2, loss_dense_1, loss_dense_2, \
                logits_local_n1, labels_local_n1, logits_local_n2, labels_local_n2, \
                    logits_local_m1, labels_local_m1, logits_local_m2, labels_local_m2



# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # torch.ones_like:
    # Returns a tensor filled with the scalar value 1, with the same size as input. 
    # torch.ones_like(input) is equivalent to torch.ones(input.size(),
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    # all_gather: 对进程组中所有的tensor执行all_gather操作，结果保存在tensors_gather中
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    # 将tensors_gather按第0维度连接起来变成一个tensor
    output = torch.cat(tensors_gather, dim=0)
    return output

class Projection_Head_Conv(nn.Module):
    def __init__(self, input_dim, projection_hidden_size=1024, output_dim=128):
        super(Projection_Head_Conv, self).__init__()
        self.aap = torch.nn.AdaptiveAvgPool3d((1, None, None))
        self.net = nn.Sequential(
            
            nn.Conv2d(input_dim, projection_hidden_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(projection_hidden_size, output_dim, kernel_size=1, stride=1, padding=0, bias=True)
        )
    def forward(self, x):
        #print(x.shape)
        x=self.aap(x)
        x=x.view(x.size(0),x.size(1),x.size(3),x.size(4))
        return self.net(x)

class Projector(nn.Module):
    def __init__(self, input_dim, output_dim=128, projection_hidden_size=1024):
        super(Projector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(projection_hidden_size, output_dim)
        )

    def forward(self, x):
        return self.net(x)