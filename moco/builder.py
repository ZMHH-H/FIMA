import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from moco.loader import Augment_GPU_pre
from moco.loss import Dense_Contrastive_Loss
from moco.loader_utils import Gen_CAAM_MSAK,Gen_Static_Diff
from backbone import transformer
# import os
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, Dense_T=0.1, pos_ratio=0.7, crop_size=224, dataset='ucf101'):
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
        self.dataset = dataset

        print('pos_ratio',pos_ratio)
        print('Dense_T',Dense_T)
        self.aug_gpu = Augment_GPU_pre(crop_size)
        self.dense_loss = Dense_Contrastive_Loss(pos_ratio = pos_ratio, temperature = Dense_T)
        self.caam_mask = Gen_CAAM_MSAK(crop_size=crop_size)
        self.gen_static_diff = Gen_Static_Diff()
        # create the encoders
        # num_classes is the output fc dimension
        # dense and global loss share the same projector
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

        elif base_encoder.__name__ == 'R2PLUS1D':
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
            
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.projector_dense.parameters(), self.projector_dense_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        # create the queue(memory bank)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0) #l2 normalization
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
         # create the queue(memory bank)
        self.register_buffer("motion_queue", torch.randn(dim, K))
        self.motion_queue = nn.functional.normalize(self.motion_queue, dim=0) #l2 normalization
        self.register_buffer("motion_queue_ptr", torch.zeros(1, dtype=torch.long))

        if self.dataset == 'k400':
            self.dense_K = 31360
            # create the queue for dense feature(memory bank)
            self.register_buffer("queue_dense", torch.randn(dim, self.dense_K)) 
            self.queue_dense = nn.functional.normalize(self.queue_dense, dim=0) #l2 normalization
            self.register_buffer("queue_dense_ptr", torch.zeros(1, dtype=torch.long))
        else:
            self.queue_dense=None

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
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_motion(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]
        
        ptr = int(self.motion_queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.motion_queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.motion_queue_ptr[0] = ptr

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
        self.queue_dense[:, ptr:ptr + feat_number] = dense_keys.T
        ptr = (ptr + feat_number) % self.dense_K  # move pointer

        self.queue_dense_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
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
        
        return m_q, mq_dense
        
    def compute_logits(self, m_q, n_k):
        # compute logits, Einstein sum is more intuitive
        # q,k: [bs,128]
        l_pos = torch.einsum('nc,nc->n', [m_q, n_k]).unsqueeze(-1)                  # [bsx1]
        
        # queue: [128,K] (128x65536)
        l_neg = torch.einsum('nc,ck->nk', [m_q, self.queue.clone().detach()])     #[bs,K]

        logits = torch.cat([l_pos, l_neg], dim=1)                               # [bs,K+1]

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(n_k)

        return logits, labels
    
    def compute_logits_local(self, m_q_l1, n_k_l1, n_k_l2, m_k_l1, m_k_l2):
        # compute logits, Einstein sum is more intuitive
        # q,k: [bs,128]
        l_pos = torch.einsum('nc,nc->n', [m_q_l1, n_k_l1]).unsqueeze(-1)                  # [bsx1]
        
        # queue: [128,K] (128x65536)
        neg_feat = torch.cat([n_k_l2, m_k_l1, m_k_l2],dim=0) # [3bs,128]
        #[bs,3bs+K]
        l_neg = torch.einsum('nc,ck->nk', [m_q_l1, torch.cat([neg_feat.permute(1,0),self.motion_queue.clone().detach()],dim=1)])

        logits = torch.cat([l_pos, l_neg], dim=1)                               # [bs,3bs+1]

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
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
        motion_mask_m = self.caam_mask(mk_base, dmk_base)
        motion_mask_n = self.caam_mask(nk_base, dnk_base)
        
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
        # mq_dense needs to build correspondence with nq_dense，and use the info from nq_dense to predicr diff features
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

        # compute logits
        logits_1, labels_1 = self.compute_logits(m_q,n_k)
        logits_2, labels_2 = self.compute_logits(n_q,m_k)

        # compute local RGB prediction loss
        logits_local_n1, labels_local_n1 = self.compute_logits_local(pred_nk_l1, n_k_l1, n_k_l2, m_k_l1, m_k_l2)
        logits_local_n2, labels_local_n2 = self.compute_logits_local(pred_nk_l2, n_k_l2, n_k_l1, m_k_l1, m_k_l2)
        logits_local_m1, labels_local_m1 = self.compute_logits_local(pred_mk_l1, m_k_l1, m_k_l2, n_k_l1, n_k_l2)
        logits_local_m2, labels_local_m2 = self.compute_logits_local(pred_mk_l2, m_k_l2, m_k_l1, n_k_l1, n_k_l2)
        
        # deque and enque motion features
        # 4*[bs,128]->[4bs,128]->enque
        self._dequeue_and_enqueue_motion(torch.cat([n_k_l1,n_k_l2,m_k_l1,m_k_l2],dim=0))
        
        # compute dense loss
        loss_dense_n1=self.dense_loss(pred_dnk_dense1, dnk_dense[0].squeeze(2), dnk_dense[1].squeeze(2), 
                                     coord_m, coord_n, motion_mask_m, motion_mask_n, self.queue_dense)
        loss_dense_n2=self.dense_loss(pred_dnk_dense2, dnk_dense[1].squeeze(2), dnk_dense[0].squeeze(2), 
                                     coord_m, coord_n, motion_mask_m, motion_mask_n, self.queue_dense)                             
        loss_dense_m1=self.dense_loss(pred_dmk_dense1, dmk_dense[0].squeeze(2), dmk_dense[1].squeeze(2), 
                                     coord_n, coord_m, motion_mask_n, motion_mask_m, self.queue_dense)
        loss_dense_m2=self.dense_loss(pred_dmk_dense2, dmk_dense[1].squeeze(2), dmk_dense[0].squeeze(2), 
                                     coord_n, coord_m, motion_mask_n, motion_mask_m, self.queue_dense)
        loss_dense_1 = (loss_dense_n1+loss_dense_n2)/2
        loss_dense_2 = (loss_dense_m1+loss_dense_m2)/2
        if self.dataset == 'k400':
            # deque and enque dense features
            self._dequeue_and_enqueue_dense(dnk_dense[0].squeeze(2).contiguous()) # B*H*W=392 features enqueue
            self._dequeue_and_enqueue_dense(dnk_dense[1].squeeze(2).contiguous()) # B*H*W=392 features enqueue
            self._dequeue_and_enqueue_dense(dmk_dense[0].squeeze(2).contiguous()) # B*H*W=392 features enqueue
            self._dequeue_and_enqueue_dense(dmk_dense[1].squeeze(2).contiguous()) # B*H*W=392 features enqueue
        
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
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
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