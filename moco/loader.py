import random
from torch.utils.data import Sampler
from torchvision.datasets.video_utils import VideoClips
import torch
import kornia
import torchvision.transforms as transforms
from kornia.augmentation.container import VideoSequential
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
from dataset import transform_coord

def Augment_GPU_pre(crop_size):
    radius = int(0.1*crop_size)//2*2+1
    sigma = random.uniform(0.1, 2)
    # For k400 parameter:
    mean = torch.tensor([0.43216, 0.394666, 0.37645])
    std = torch.tensor([0.22803, 0.22145, 0.216989])

    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        # kornia.augmentation.RandomHorizontalFlip(),
        kornia.augmentation.RandomGaussianBlur((radius, radius), (sigma, sigma), p=0.5),
        normalize_video,
        data_format="BCTHW",
        same_on_frame=True)
    return aug_list


def Augment_GPU_ft(args):
    # For k400 parameter:
    mean = torch.tensor([0.43216, 0.394666, 0.37645])
    std = torch.tensor([0.22803, 0.22145, 0.216989])
    normalize_video = kornia.augmentation.Normalize(mean, std)
    aug_list = VideoSequential(
        kornia.augmentation.RandomGrayscale(p=0.2),
        kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        normalize_video,
        data_format="BCTHW",
        same_on_frame=True)
    return aug_list


class RandomTwoClipSampler(Sampler):
    """
    Samples two clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select two clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 2:
                sampled = [s, s]
            else:
                # torch.randperm(n): Returns a random permutation of integers from 0 to n - 1
                sampled = torch.randperm(length)[:2] + s
                sampled = sampled.tolist()
            s += length
            idxs.append(sampled)
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)

class RandomOneClipSampler(Sampler):
    """
    Samples one clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select one clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 1:
                print('length equal to ',length)
                sampled = [s]
            else:
                # torch.randperm(n): Returns a random permutation of integers from 0 to n - 1
                sampled = torch.randperm(length)[:1] + s
                sampled = sampled.tolist()
            s += length
            idxs.append(sampled)
        # shuffle all clips
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)

class RandomTempCloseClipSampler(Sampler):
    """
    Samples one clips for each video randomly

    Arguments:
        video_clips (VideoClips): video clips to sample from
    """
    def __init__(self, video_clips):
        if not isinstance(video_clips, VideoClips):
            raise TypeError("Expected video_clips to be an instance of VideoClips, "
                            "got {}".format(type(video_clips)))
        self.video_clips = video_clips

    def __iter__(self):
        idxs = []
        s = 0
        # select one clips for each video, randomly
        for c in self.video_clips.clips:
            length = len(c)
            if length < 1:
                print('length equal to ',length)
                sampled = [s]
            else:
                # torch.randperm(n): Returns a random permutation of integers from 0 to n - 1
                sampled = torch.randperm(length)[:1]
                if sampled + 32 < length - 1: # 16 frames * 2 stride
                    sampled = torch.cat((sampled,sampled+32))
                else:
                    sampled = torch.cat((sampled,torch.tensor([length-1])))
                sampled = sampled + s
                sampled = sampled.tolist()
            s += length
            idxs.append(sampled)
        # shuffle all clips
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return len(self.video_clips.clips)

class DummyAudioTransform(object):
    """This is a dummy audio transform.

    It ignores actual audio data, and returns an empty tensor. It is useful when
    actual audio data is raw waveform and has a varying number of waveform samples
    which makes minibatch assembling impossible

    """

    def __init__(self):
        pass

    def __call__(self, _audio):
        return torch.zeros(0, 1, dtype=torch.float)

def get_train_sampler(train_dataset,args):
    
    if args.clip_per_video == 1:
        train_sampler = RandomOneClipSampler(train_dataset.video_clips)
    elif args.clip_per_video == 2:
        train_sampler = RandomTwoClipSampler(train_dataset.video_clips)
    elif args.clip_per_video == 3:
        train_sampler = RandomTempCloseClipSampler(train_dataset.video_clips)
    else:
        raise NotImplementedError("UCF101 sampler clip number not support ")
    return train_sampler

def get_transform(args,return_coord = False):
    if return_coord:
        print('** transformation will return coordinates **')
        video_augmentation = transform_coord.Compose(
        [
        
            transforms_video.ToTensorVideo(),
            transform_coord.RandomResizedCropVideo_Coord(args.crop_size, (0.2, 1)),
            transform_coord.RandomHorizontalFlipVideo_Coord(),
        ]
    )
    else:
        print('** normal transformation **')
        video_augmentation = transforms.Compose(
            [
            
                transforms_video.ToTensorVideo(),
                transforms_video.RandomResizedCropVideo(args.crop_size, (0.2, 1)),
                transforms_video.RandomHorizontalFlipVideo(),
            ]
        )

    audio_augmentation = DummyAudioTransform()
    augmentation = {'video': video_augmentation, 'audio': audio_augmentation}
    print('The image size is {}'.format(args.crop_size))
    return augmentation