import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
class Gen_Static_Diff(nn.Module):
    def __init__(self, mask_dim=7, motion_patch_ratio=0.5):
        super(Gen_Static_Diff, self).__init__()
        self.mask_dim = mask_dim
        self.motion_patch_ratio = motion_patch_ratio

    def gen_static_clip(self,clip):
        #generate static clip batch
        B, C, T, H, W = clip.shape
        frame_idx = random.randint(0,T-1) # pick one frame
        static_clip = clip[:,:,frame_idx,:,:].unsqueeze(2).repeat(1,1,T,1,1) # repeat T times
        return static_clip

    def gen_diff_clip(self,clip):

        # ==================  version1  ===================
        shift_clip = torch.roll(clip, 1, 2)
        diff_clip = ((clip - shift_clip) + 1) / 2
        
        # # ==================  version2 ===================
        # diff_clip = clip[:, :, 0:-1,:,:] - clip[:, :, 1:,:,:]

        # # ==================  version3 ===================
        # diff_clip = clip[:, :, 1:,:,:] - clip[:, :, 0:-1,:,:]
        # ==================  version4 ===================
        #diff_clip = clip[:, :, 1:,:,:] - clip[:, :, 0,:,:].unsqueeze(2).repeat(1,1,7,1,1)
        return diff_clip

    def diff_mask(self,clip):
        # clip: B x C x T x H x W
        B, C, T, H, W = clip.shape
        assert H == W and H % self.mask_dim == 0, "The W，H of the video must be a multiple of mask_dim"

        diff_sum = (clip[:, :, 0:-1,:,:] - clip[:, :, 1:,:,:]).abs().sum(dim=1).mean(dim=1)  # B, H, W
        # B,H,W = 1, 14 ,14
        # diff_sum = torch.arange(196).reshape(1,14,14)

        # B x H x mask_dim(7)
        patch_sum = diff_sum.reshape(B,H,self.mask_dim,-1).sum(-1) 

        # B x mask_dim(7) x mask_dim(7)
        patch_sum = patch_sum.reshape(B,self.mask_dim,-1,self.mask_dim).sum(-2) 
        # print('patch_sum.shape',patch_sum.shape)
        
        # motion patch number
        # int(0.5*7*7)
        num_ma = int(self.motion_patch_ratio * self.mask_dim * self.mask_dim)

        # motion patch index
        motion_index = torch.topk(patch_sum.reshape(B, -1), k=num_ma, dim=-1)[1]

        # generate mask
        mask = torch.zeros_like(patch_sum).reshape(B,-1) # B x [mask_dim(7)*mask_dim(7)
        batch_index = torch.LongTensor([[i]*num_ma for i in range(B)])
        mask[batch_index.reshape(-1),motion_index.reshape(-1)] = 1
        #print('mask',mask.reshape(B,self.mask_dim,self.mask_dim))

        return mask.reshape(B,self.mask_dim,self.mask_dim)
    
    def diff_mask_fuse(self,clip):
        B, C, T, H, W = clip.shape
        grid_mask = self.diff_mask(clip) # [B,7,7]
        resclaed_mask = F.interpolate(grid_mask.unsqueeze(1),
                                        scale_factor=H/grid_mask.shape[1],mode='nearest').unsqueeze(1) # [B,1,112,112]
        index = torch.randperm(B, device=clip.device)
        clip_fuse = clip[index] * (1 - resclaed_mask) + clip * resclaed_mask
        return clip_fuse, grid_mask

    def forward(self,x,option='mask'):
        if option=='static':
            # 1，generate static clip
            result = self.gen_static_clip(x)
        
        elif option=='diff':
            # 2，generate difference clip
            result = self.gen_diff_clip(x)

        elif option=='mask':  
            # 3，generate difference mask
            result = self.diff_mask(x)
        elif option=='mask_fuse':
            result = self.diff_mask_fuse(x)
        else:
            raise NotImplementedError("static diference option is not supported.")
        return result

class Gen_CAAM_MSAK(nn.Module):
    def __init__(self, crop_size=224):
        super(Gen_CAAM_MSAK, self).__init__()
        self.crop_size = crop_size
    

    def gen_caam_mask_max(self, rgb_feature_map, diff_feature_map):
        # video: [bs,C,T,224,224]
        # rgb_feature_map:[bs,C,T,7,7]
        # diff_feature_map:[bs,C,T,7,7]
        rgb_caam = rgb_feature_map.clone().detach().mean(dim=1).mean(1,True)       # [bs,1,7,7]
        rgb_caam = F.interpolate(rgb_caam,(self.crop_size,self.crop_size),mode='bilinear') # upsample:[bs,1,224,224]
        rgb_caam = rgb_caam.squeeze(1)    # [bs,224,224]
        
        diff_caam = diff_feature_map.clone().detach().mean(dim=1).mean(1,True)       # [bs,1,7,7]
        diff_caam = F.interpolate(diff_caam,(self.crop_size,self.crop_size),mode='bilinear') # upsample:[bs,1,224,224]
        diff_caam = diff_caam.squeeze(1)    # [bs,224,224]

        # compute filter
        thresh_rgb = 0.4 *rgb_caam.max(-1, keepdim=True)[0].max(-2,keepdim=True)[0] # [bs,1,1]
        thresh_diff = 0.4 *diff_caam.max(-1, keepdim=True)[0].max(-2,keepdim=True)[0] # [bs,1,1]
        
        # # other version, directly add two caam and compute filter, larger occlusion area, shape basically unchanged
        # caam = rgb_caam+diff_caam
        # thresh = 0.3 *caam.max(-1, keepdim=True)[0].max(-2,keepdim=True)[0] # [bs,1,1]
        # mask1 = (caam>thresh).int()
        # video_decode = video_decode*mask1.unsqueeze(1).unsqueeze(1)

        rgb_mask = (rgb_caam>thresh_rgb)    # [bs,224,224]
        diff_mask = (diff_caam>thresh_diff) # [bs,224,224]
        mask = (diff_mask|rgb_mask).int()   # [bs,224,224]
        # video_decode = video_decode*mask.unsqueeze(1).unsqueeze(1)
        return mask.unsqueeze(1).unsqueeze(1) # [bs,1,1,224,224]

    def gen_caam_mask_median(self, rgb_feature_map, diff_feature_map):
        # video: [bs,C,T,224,224]
        # rgb_feature_map:[bs,C,T,7,7]
        # diff_feature_map:[bs,C,T,7,7]
        rgb_caam = rgb_feature_map.clone().detach().mean(dim=1).mean(1,True)        # [bs,1,7,7]
        diff_caam = diff_feature_map.clone().detach().mean(dim=1).mean(1,True)      # [bs,1,7,7]
        overall_caam = rgb_caam+diff_caam                                           # [bs,1,7,7]
       
        # calculate grid motion mask
        thresh_grid = overall_caam.reshape(overall_caam.shape[0],-1).median(-1,True)[0].unsqueeze(-1)     # [bs,1,1]
        grid_mask = (overall_caam.squeeze(1)>thresh_grid).int() # [bs,7,7]
        
        # upsample activation map
        overall_caam = F.interpolate(overall_caam,(self.crop_size,self.crop_size),mode='bilinear')  # upsample:[bs,1,224,224]
        overall_caam = overall_caam.squeeze(1)      # [bs,224,224]

        # compute filter
        # torch.quantile(0.5)
        thresh_caam = overall_caam.reshape(overall_caam.shape[0],-1).median(-1, keepdim=True)[0].unsqueeze(-1) #[bs,1,1]
        
        # compute mask
        mask = (overall_caam>thresh_caam).int()   # [bs,224,224]
        return mask.unsqueeze(1).unsqueeze(1),grid_mask # [bs,1,1,224,224],[bs,7,7]

    def gen_grid_mask(self, rgb_feature_map, diff_feature_map):
        rgb_caam = rgb_feature_map.clone().detach().mean(dim=1).mean(1,True)        # [bs,1,7,7]
        diff_caam = diff_feature_map.clone().detach().mean(dim=1).mean(1,True)      # [bs,1,7,7]
        overall_caam = rgb_caam+diff_caam                                           # [bs,1,7,7]
       
        # calculate grid motion mask
        thresh_grid = overall_caam.reshape(overall_caam.shape[0],-1).median(-1,True)[0].unsqueeze(-1)       # [bs,1,1]
        # thresh_grid = overall_caam.reshape(overall_caam.shape[0],-1).quantile(0.7,-1,True).unsqueeze(-1)
        # thresh_grid = overall_caam.reshape(overall_caam.shape[0],-1).quantile(q=0.7,dim=-1,keepdim=True).unsqueeze(-1)      # [bs,1,1]
        grid_mask = (overall_caam.squeeze(1)>thresh_grid).int() # [bs,7,7]
        return grid_mask

    def forward(self, rgb_feature_map, diff_feature_map):
        # # if need to fuse video
        # caam_mask, grid_mask = self.gen_caam_mask_median(rgb_feature_map, diff_feature_map) # [bs,1,1,224,224]
        # B,C,T,H,W = video_decode.shape
        # index = torch.randperm(B, device=video_decode.device)
        # video_fuse = video_decode[index] * (1 - caam_mask) + video_decode * caam_mask
        # return video_fuse, grid_mask
        return self.gen_grid_mask(rgb_feature_map,diff_feature_map)


def norm_feat_caam(feat_caam):
    # feat_caam: [bs,T,H,W]
    _min = feat_caam.min(-1, keepdim=True)[0].min(-2, keepdim=True)[0]  # _min:[bs,T,1,1]
    _max = feat_caam.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]  # _max:[bs,T,1,1]
    feat_caam = (feat_caam - _min) / (_max - _min)  # feat_caam: [bs,T,H,W]
    return feat_caam


if __name__ == '__main__':
    temp = torch.tensor([
                        [
                        [1,2,3,4],
                        [2,3,5,7]
                        ],
                        [
                        [2,4,6,8],
                        [3,5,7,6]
                        ]
                        ])
    print(temp.shape)
    print(temp)
    temp = norm_feat_caam(temp)
    print(temp.shape)
    print(temp)