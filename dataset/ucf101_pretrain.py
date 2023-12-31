from torchvision.datasets.utils import list_dir
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets import VisionDataset
import os
import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video

class UCF101(VisionDataset):
    """
    UCF101 is an action recognition video dataset.
    This dataset consider every video as a collection of video clips of fixed size, specified
    by ``frames_per_clip``, where the step in frames between each clip is given by
    ``step_between_clips``.

    To give an example, for 2 videos with 10 and 15 frames respectively, if ``frames_per_clip=5``
    and ``step_between_clips=5``, the dataset size will be (2 + 3) = 5, where the first two
    elements will come from video 1, and the next three elements from video 2.
    Note that we drop clips which do not have exactly ``frames_per_clip`` elements, so not all
    frames in a video might be present.

    Internally, it uses a VideoClips object to handle clip creation.

    Args:
        root (string): Root directory of the UCF101 Dataset.
        annotation_path (str): path to the folder containing the split files
        frames_per_clip (int): number of frames in a clip.
        step_between_clips (int, optional): number of frames between each clip.
        fold (int, optional): which fold to use. Should be between 1 and 3.
        train (bool, optional): if ``True``, creates a dataset from the train split,
            otherwise from the ``test`` split.
        transform (callable, optional): A function/transform that  takes in a TxHxWxC video
            and returns a transformed version.

    Returns:
        video (Tensor[T, H, W, C]): the `T` video frames
        audio(Tensor[K, L]): the audio frames, where `K` is the number of channels
            and `L` is the number of points
        label (int): class of the video clip
    """

    def __init__(self, root, annotation_path, frames_per_clip, step_between_clips=1,
                 frame_rate=None, fold=1, train=True, transform=None,
                 _precomputed_metadata=None, num_workers=1, _video_width=0,
                 _video_height=0, _video_min_dimension=0, _audio_samples=0, stride=2,
                 clip_number=2):
        super(UCF101, self).__init__(root)
        if not 1 <= fold <= 3:
            raise ValueError("fold should be between 1 and 3, got {}".format(fold))

        extensions = ('avi',)
        self.fold = fold
        self.train = train

        classes = list(sorted(list_dir(root)))
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file=None)
        self.classes = classes
        video_list = [x[0] for x in self.samples]
        metadata_filepath = os.path.join(root, 'ucf101_metadata.pt')
        if os.path.exists(metadata_filepath):
            metadata = torch.load(metadata_filepath)
        else:
            metadata = None
        video_clips = VideoClips(
            video_list,
            frames_per_clip * stride,
            step_between_clips,
            frame_rate,
            metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
        )

        if not os.path.exists(metadata_filepath):
            torch.save(video_clips.metadata, metadata_filepath)

        self.video_clips_metadata = video_clips.metadata
        self.indices = self._select_fold(video_list, annotation_path, fold, train)
        self.video_clips = video_clips.subset(self.indices)
        self.transform = transform
        self.stride = stride
        self.clip_number = clip_number

    @property
    def metadata(self):
        return self.video_clips_metadata

    def _select_fold(self, video_list, annotation_path, fold, train):
        name = "train" if train else "test"
        name = "{}list{:02d}.txt".format(name, fold)
        f = os.path.join(annotation_path, name)
        selected_files = []
        with open(f, "r") as fid:
            data = fid.readlines()
            data = [x.strip().split(" ") for x in data]
            data = [x[0] for x in data]
            selected_files.extend(data)
        selected_files = set(selected_files)
        indices = [i for i in range(len(video_list)) if video_list[i][len(self.root) + 1:] in selected_files]
        return indices

    def __len__(self):
        return self.video_clips.num_clips()

    def __getitem__(self, idx):
        videos = ()
        coords = ()
        # videos_ori = () ## return origin clip
        
        for i in idx:
            video, audio, info, video_idx = self.video_clips.get_clip(i)
            # print(video.shape) # 32, 240, 320, 3
            if self.clip_number == 2 or self.clip_number == 3:
                if self.transform is not None:
                    video1, coord1 = self.transform['video'](video[::self.stride])
                    video2, coord2 = self.transform['video'](video[:16])
                    video3, coord3 = self.transform['video'](video[16:])
                    videos += (video1,video2,video3)
                    coords += (coord1,coord2,coord3)
                    # video1 = self.transform['video'](video[::self.stride])
                    # video2 = self.transform['video'](video[::self.stride])
                    # videos += (video1,video2)
            elif self.clip_number == 1:
                if self.transform is not None:
                    video1, coord1 = self.transform['video'](video[::self.stride])
                    video2, coord2 = self.transform['video'](video[::self.stride])
                    videos += (video1,video2)
                    coords += (coord1,coord2)
            else:
                raise NotImplementedError("UCF101 dataloader: clip number is not support ")

        return videos, coords










