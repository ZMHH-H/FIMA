from turtle import pos
import torch
import numpy
import torch.nn as nn
torch.set_printoptions(threshold=numpy.inf)

class Dense_Contrastive_Loss(nn.Module):
    def __init__(self, pos_ratio = 0.7, temperature=0.1):
        super(Dense_Contrastive_Loss,self).__init__()
        self.pos_ratio = pos_ratio
        self.temperature = temperature

    def forward(self, q, k, k_neg, coord_q, coord_k, motion_mask_q, motion_mask_k, queue_dense):
        """ q, k: N * C * H * W   (must be normalized)
            coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
            motion_mask_q, motion_mask_k: N x H x W (Nx7x7)
        """
        N, C, H, W = q.shape # N x 128 x 7 x 7
        assert H == motion_mask_q.shape[1] and W == motion_mask_q.shape[2]

        # [bs, feat_dim, 49]
        q = q.view(N, C, -1)
        k = k.view(N, C, -1)
        
        # [bs, feat_dim, 98]
        k_neg = k_neg.view(N, C, -1)
        k_neg = torch.cat((k,k_neg),dim=2)

        # generate center_coord, width, height
        # [1, 7, 7]
        # x_array：[0,1,2,3,4,5,6].repeat(1, H, 1)
        x_array = torch.arange(0., float(W), dtype=coord_q.dtype, device=coord_q.device).view(1, 1, -1).repeat(1, H, 1)
        y_array = torch.arange(0., float(H), dtype=coord_q.dtype, device=coord_q.device).view(1, -1, 1).repeat(1, 1, W)

        # [bs, 1, 1]
        q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
        q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
        k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
        k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)

        # [bs, 1, 1]
        q_start_x = coord_q[:, 0].view(-1, 1, 1)
        q_start_y = coord_q[:, 1].view(-1, 1, 1)
        k_start_x = coord_k[:, 0].view(-1, 1, 1)
        k_start_y = coord_k[:, 1].view(-1, 1, 1)

        # [bs, 1, 1]
        q_bin_diag = torch.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
        k_bin_diag = torch.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
        max_bin_diag = torch.max(q_bin_diag, k_bin_diag)

        # [bs, 7, 7]
        center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
        center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
        center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
        center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

        # [bs, 49, 49]
        # calculate distance matrix
        dist_center = torch.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                                + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
        pos_mask = (dist_center < self.pos_ratio).float().detach()

        # [bs, 49, 49]
        # apply motion mask
        motion_mask_q = motion_mask_q.detach().view(N,-1,1)     # [bs,49,1]
        motion_mask_k = motion_mask_k.detach().view(N,1,-1)     # [bs,1,49]
        pos_mask = pos_mask*motion_mask_q*motion_mask_k # broadcastable multiple
        # pos_mask = pos_mask*motion_mask_q # do not apply motion mask_k on momentum encoder branch
        # print('pos_mask.shape',pos_mask.shape)
        # print('pos_mask',pos_mask)

        # [bs, 49, 49]
        # [bs, 49, feat_dim] @ [bs, feat_dim, 49] = [bs, 49, 49]
        # calculate similarity matrix
        # apply temperature and exp()
        logit = (torch.bmm(q.transpose(1, 2), k) /self.temperature).exp()
        # print('logit',logit)
        # print('logit.shape',logit.shape)

        # [bs,49,bs*49]
        batch_q = q.transpose(1,2).reshape(-1,C) # [bs*49,C]
        batch_k = k_neg.transpose(0,1).reshape(C,-1) # [C,bs*98]
        if queue_dense is not None:
            batch_k = torch.cat((batch_k,queue_dense.clone().detach()),dim=1) # [C,bs*98+len(queue)]

        # print(batch_q.requires_grad)
        batch_logit = (torch.einsum('nc,ck->nk', [batch_q, batch_k]).reshape(N,H*W,-1) 
                                                                /self.temperature).exp() # [bs,49,bs*98]
        # batch_logit1 = torch.matmul(batch_q,batch_k)
        
        # [<bs,49,49]
        # remove feature map without positive pairs
        num_pos_pairs = pos_mask.sum(-1).sum(-1)        # [bs]
        pos_mask = pos_mask[num_pos_pairs>0]            # [<bs,49,49]
        #print('pos_mask,logit',pos_mask.shape,logit.shape)
        if pos_mask.shape[0] == 0:                      # all elements in one minibatch without positive pairs
            loss = -torch.log( logit.sum(-1).sum(-1) / (logit.sum(-1).sum(-1) + 1e-3)).mean() # return zero loss
            return loss 
        logit = logit[num_pos_pairs>0]                  # [<bs,49,49]
        batch_logit = batch_logit[num_pos_pairs>0]      # [<bs,49,bs*49]

        
        # calculate loss for pixels in view1 independently, and avergaed over the number of effective pixels in view1
        pos_logit_view1 = (logit*pos_mask).sum(-1)              # [bs,49]
        # pos_neg_view1 = logit.sum(-1)                         # [bs,49] positive and negative logit in view
        pos_neg_view1 = batch_logit.sum(-1)                     # [bs,49] positive and negative logit in minibatch
        effective_pixels = pos_logit_view1.count_nonzero(dim=1) # [bs]    number of effective pixels for each view in a minibatch
        
        # if pos_logit_view1>0，then preserve the original value. Othervise，Filling the value of pos_neg_view1
        pos_logit_view1 = torch.where(pos_logit_view1>0, pos_logit_view1, pos_neg_view1)    # [bs,49]
        
        # calculate -log loss for each pixel and add pixels in same view together,
        # averaged over effective pixels, and then averaged over minibatch
        loss2 = (-torch.log(pos_logit_view1/pos_neg_view1).sum(-1)/effective_pixels).mean() 

        return loss2


# if __name__ == '__main__':
#     a = torch.arange(18,dtype=torch.float32).view(2,1,3,3).cuda()/100
#     b = torch.arange(10,28,dtype=torch.float32).view(2,1,3,3).cuda()/100
#     coord_a = torch.tensor([[0,0,0.6,0.6],[0,0,0.6,0.6]]).cuda()
#     coord_b = torch.tensor([[0.4,0.2,1,0.8],[0.4,0.2,1,0.8]]).cuda()
#     mask_a = torch.tensor([[1,1,1,1,0,1,1,1,0],[1,1,1,1,0,1,1,1,0]]).view(2,3,3).cuda()
#     mask_b = torch.tensor([[1,0,1,1,1,1,1,1,1],[1,0,1,1,1,1,1,1,1]]).view(2,3,3).cuda()
#     print(a)
#     print(b)
#     print(coord_a.shape)
#     print(mask_a.shape)
#     print(mask_a)
#     print(mask_b)
#     dense_contrastive_loss = Dense_Contrastive_Loss(pos_ratio = 0.7, temperature=0.3)
#     dense_contrastive_loss(a,b,coord_a,coord_b,mask_a,mask_b)
    

#     logit = torch.tensor([
#         [[1,1],[1,1]],[[0,1],[0,0]],[[0,0],[1,0]],[[0,0],[0,1]],[[1,1],[1,1]]
#     ])
#     pos_mask = torch.tensor([
#         [[1,0],[0,0]],[[0,1],[0,0]],[[0,0],[1,0]],[[0,0],[0,1]],[[1,1],[1,1]]
#     ])
#     num_pos_pairs = torch.tensor([1,1,0,1,0])
#     print('cccc',pos_mask)
#     print('cccc',pos_mask.shape)
#     pos_mask = pos_mask[num_pos_pairs>0]
#     print('ddd',pos_mask)
#     print('ddd',pos_mask.shape)

#     logit= logit[num_pos_pairs>0]
#     print('eeee',logit)
#     print('eeee',logit.shape)
#         # if torch.equal(pos_logit[i],torch.tensor([0])):
#         #     loss = 0
#         #     print(torch.equal(pos_logit[i],torch.tensor([0])))

#     temp= torch.tensor([[1,2,3,0,0,0,0,0,0],[1,3,4,0,0,0,0,1,0]]).view(2,3,3).cuda() 
#     print('temp',temp)
#     print('temp1',temp.sum(-1))
#     print('temp11',temp.sum(-1).count_nonzero(dim=1))
#     print('temp111',-torch.log(temp.sum(-1)))
#     print('temp1111',(-torch.log(temp.sum(-1))).shape)