
import numbers
from torchvision.transforms import _functional_video as F
import torch
class VideoClipScale(object):
    def __init__(self, scale_size):
        self.scale_size = scale_size

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        height,width = clip.shape[-2],clip.shape[-1]
        # print('h: ',height,'w: ',width)
        if height > width:
            ratio = height / width
            new_width = self.scale_size
            new_height = int(ratio * new_width)
        else:
            ratio = width / height
            new_height = self.scale_size
            new_width = int(ratio * new_height)
        return torch.nn.functional.interpolate(clip, size=(new_height,new_width), mode="bilinear", align_corners=False)
    
    def __repr__(self):
        return self.__class__.__name__ + '(scale_size={0})'.format(self.scale)