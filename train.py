#!/usr/bin/env python
import os

import numpy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import builtins
import random
import time
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torchvision.datasets.samplers.clip_sampler import DistributedSampler, RandomClipSampler
from dataset.kinetics import Kinetics400
from dataset.ucf101_pretrain import UCF101
import moco.loader
import moco.builder
from torch.utils.tensorboard import SummaryWriter
from utils.train_utils import adjust_learning_rate, accuracy, save_checkpoint, AverageMeter, ProgressMeter
from backbone.i3d import I3D
from backbone.r2plus1d import R2PLUS1D
from backbone.r3d import R3D


model_names = ["I3D", "r2plus1d_18"]
parser = argparse.ArgumentParser(description='PyTorch Self-supervised Video Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='r3d_18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument("--dataset", default="ucf101",
                    choices=["k400", "ucf101"],
                    help='pretrain datasets')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-cs', '--crop_size', default=112, type=int, metavar='N',
                    help='crop size for video clip (default: 112)')
parser.add_argument('-fpc', '--frame_per_clip', default=16, type=int, metavar='N',
                    help='number of frame per video clip (default: 16)')
parser.add_argument('-sbc', '--step_between_clips', default=1, type=int, metavar='N',
                    help='number of steps between video clips (default: 1)')

parser.add_argument('--lr', '--learning_rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_decay', '--learning_rate_decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay', dest='lr_decay')
parser.add_argument('--warmup', action='store_true',
                    help='use warm up lr schedule')
parser.add_argument('--wp_lr', '--warmup_learning_rate', default=0.0025, type=float,
                    metavar='WLR', help='initial warmup learning rate', dest='wp_lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print_freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--log_dir', default='logs_moco', type=str,
                    help='path to the tensorboard log directory')
parser.add_argument('--ckp_dir', default='checkpoints_moco', type=str,
                    help='path to the moco model directory')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco_k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco_t', default=0.1, type=float,
                    help='softmax temperature (default: 0.0)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
# parser.add_argument('--aug_plus', action='store_true',
#                     help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')

# foreground augmentation
parser.add_argument('--beta',default=0.5, type=float,
                    help='portion of the foreground')

# @zmhh_h
parser.add_argument('--dense_temperature', default=0.1, type=float,
                    help='dense temperature parameters for dense infoNCE loss')

parser.add_argument('--pos_ratio', default=0.7, type=float,
                    help='positive pairs distance threshold for dense feature maps')

parser.add_argument('--clip_per_video', default=2, type=float,
                    help='clip number per video'
                         '1:randomly sample 1 clip in video and return 2 croped versions in the same clip'
                         '2:randomly sample 2 clips in video and return 1 croped version in each clip')

parser.add_argument('--dense_loss_weight', default=1.0, type=float,
                    help='the weight of dense loss term')
def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        numpy.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu!=0:
        time.sleep(1)
    print('args.gpu',args.gpu)

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            print(args.dist_url, args.rank, args.world_size)

        # world_size(int, optional): Number of processes participating in the job. Required if store is specified.
        # rank (int, optional): Rank of the current process 
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    # moco memory bank length
    if args.dataset == 'k400':
        args.moco_k = 65536
    elif args.dataset == 'ucf101':
        args.moco_k = 2240 # 2048
    
    # 创建模型
    if args.arch == "r2plus1d_18":
        model = moco.builder.MoCo(
            R2PLUS1D,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.dense_temperature, args.pos_ratio, args.crop_size, args.dataset)
    elif args.arch == "I3D":
        model = moco.builder.MoCo(
            I3D,
            args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.dense_temperature, args.pos_ratio, args.crop_size, args.dataset)
    #print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        pass #raise NotImplementedError("Only DistributedDataParallel is supported.") for debug on cpu

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
    # get basic data augmentation
    augmentation = moco.loader.get_transform(args,return_coord=True)

    # Data loading code
    if args.dataset == "k400":
        traindir = os.path.join(args.data, 'train_256')
        train_dataset = Kinetics400(
           traindir,
           args.frame_per_clip,
           args.step_between_clips,
           extensions='mp4',
           transform=augmentation,
           num_workers=64,
           stride = 2,
           clip_number = args.clip_per_video
        )
    elif args.dataset == "ucf101":
        data_dir = os.path.join(args.data, 'data')
        anno_dir = os.path.join(args.data, 'anno') ## no use in pretrain
        train_dataset = UCF101(
            data_dir,
            anno_dir,
            args.frame_per_clip,
            args.step_between_clips,
            fold=1,
            train=True,
            transform=augmentation,
            num_workers=64,
            stride = 2,
            clip_number=args.clip_per_video
        )
    print("loading dataset {}".format(args.dataset))
    
    train_sampler = moco.loader.get_train_sampler(train_dataset,args)
    if args.distributed:
        train_sampler = DistributedSampler(train_sampler)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True,
        multiprocessing_context="fork")

    if args.multiprocessing_distributed and args.gpu == 0:
        log_dir = "{}_bs={}_lr={}_cs={}_fpc={}".format(args.log_dir, args.batch_size, args.lr, args.crop_size, args.frame_per_clip)
        writer = SummaryWriter(log_dir)
    else:
        writer = None
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        
        if (epoch % 10 == 0 or epoch == args.epochs - 1) and (not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0)):
            ckp_dir = "{}_bs={}_lr={}_cs={}_fpc={}".format(args.ckp_dir, args.batch_size, args.lr, args.crop_size,
                                                           args.frame_per_clip)
            save_checkpoint(epoch, {
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckp_dir, max_save=3, is_best=False)
    print(args)

def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    # initialize meters
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    losses_d = AverageMeter('DLoss', ':.4e')
    losses_l = AverageMeter('LLoss', ':.4e')
    top1_l = AverageMeter('LAcc@1', ':6.2f')
    top5_l = AverageMeter('LAcc@5', ':6.2f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, losses_d, top1, top5, losses_l, top1_l, top5_l],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    
    end = time.time()
    for i, data in enumerate(train_loader):
        # video:[video0,video1], video0: [bs,C,T,H,W]
        video = data[0] 
        # coord:[coord_video0, coord_video1],  coord_video0: [bs,4] (x1,y1,x2,y2)
        coord = data[1] 

        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpu is not None:
            for j in range(len(video)):
                video[j] = video[j].cuda(args.gpu, non_blocking=True)
            for j in range(len(coord)):
                coord[j] = coord[j].cuda(args.gpu, non_blocking=True)

        # ================ model forward ===========================
        logits_1, labels_1, logits_2, labels_2, loss_dense1, loss_dense2,\
            logits_local_n1, labels_local_n1, logits_local_n2, labels_local_n2,\
                logits_local_m1, labels_local_m1, logits_local_m2, labels_local_m2= model(
                                                                        vi_m=video[0], vi_m_l1=video[1],vi_m_l2=video[2],
                                                                        vi_n=video[3], vi_n_l1=video[4],vi_n_l2=video[5],
                                                                        coord_m=coord[0],coord_n=coord[3])
        
        # ================ compute loss ============================
        # symmetric RGB_RGB global contrastive loss
        loss_1 = criterion(logits_1, labels_1)
        loss_2 = criterion(logits_2, labels_2)
        loss_instdisc = (loss_1 + loss_2) / 2

        # symmetric frame-level motion reconstruction loss
        loss_local_n1 = criterion(logits_local_n1, labels_local_n1)
        loss_local_n2 = criterion(logits_local_n2, labels_local_n2)
        loss_local_m1 = criterion(logits_local_m1, labels_local_m1)
        loss_local_m2 = criterion(logits_local_m2, labels_local_m2)

        loss_local = (loss_local_n1 + loss_local_n2 + loss_local_m1 + loss_local_m2) / 4

        # pixel-level motion contrastive loss
        loss_dense = args.dense_loss_weight*((loss_dense1 + loss_dense2) / 2)
        # overall loss
        loss = loss_instdisc + loss_dense + loss_local
        
        # ================ compute acc and update meters ===========
        # acc1/acc5 are (K+1)-way contrast classifier accuracy
        # measure accuracy
        acc1, acc5 = accuracy((logits_1+logits_2)/2, labels_1, topk=(1, 5))
        acc1_local, acc5_local = accuracy((logits_local_n1+logits_local_m1)/2, labels_local_n1, topk=(1, 5))
        # record loss
        losses.update(loss.item(), video[0].size(0))
        losses_d.update(loss_dense.item(), video[0].size(0))
        losses_l.update(loss_local.item(), video[0].size(0))

        # record accuracy
        top1.update(acc1[0], video[0].size(0))
        top5.update(acc5[0], video[0].size(0))

        top1_l.update(acc1_local[0], video[0].size(0))
        top5_l.update(acc5_local[0], video[0].size(0))

        # ================ model backward ==========================
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            if writer is not None:
                total_iter = i+epoch*len(train_loader)
                writer.add_scalar('moco_train/loss', loss, total_iter)
                writer.add_scalar('moco_train/acc1', acc1, total_iter)
                writer.add_scalar('moco_train/acc5', acc5, total_iter)
                writer.add_scalar('moco_train_avg/lr', optimizer.param_groups[0]['lr'], total_iter)
                writer.add_scalar('moco_train_avg/loss', losses.avg, total_iter)
                writer.add_scalar('moco_train_avg/loss_dense', losses_d.avg, total_iter)
                writer.add_scalar('moco_train_avg/loss_local', losses_l.avg, total_iter)
                writer.add_scalar('moco_train_avg/acc1', top1.avg, total_iter)
                writer.add_scalar('moco_train_avg/acc5', top5.avg, total_iter)
                writer.add_scalar('moco_train_avg/acc1_slow', top1_l.avg, total_iter)
                writer.add_scalar('moco_train_avg/acc5_slow', top5_l.avg, total_iter)
            # print("iter:%d: loss = %3f, acc1 = %3f, acc5 = %3f" %(loss,acc1,acc5))
            # print("iter:%d: loss_avg = %3f, acc1_avg = %3f, acc5_avg = %3f" %(losses.avg, top1.avg, top5.avg))

if __name__ == '__main__':
    main()
