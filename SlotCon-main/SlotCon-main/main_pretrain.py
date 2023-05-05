import argparse
import json
import os
import random
import shutil
import time
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models

from data.datasets import ImageFolder
from data.transforms import CustomDataAugmentation

from models import resnet
from models import vision_transformer as vits
from models.slotcon import SlotCon
from utils.lars import LARS
from utils.logger import setup_logger
from utils.lr_scheduler import get_scheduler
from utils.util import AverageMeter

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))
print(vits.__dict__)
model_names = sorted(name for name in resnet.__all__ if name.islower() and callable(resnet.__dict__[name]))
model_names.append(torchvision_archs)

def get_parser():
    parser = argparse.ArgumentParser('SlotCon')

    # dataset
    parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO', 'COCOplus', 'ImageNet'], help='dataset type')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')
    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
   
    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names, help='backbone architecture')
    parser.add_argument('--dim-hidden', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-out', type=int, default=256, help='output feature dimension')
    parser.add_argument('--num-prototypes', type=int, default=256, help='number of prototypes')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--teacher-temp', default=0.07, type=float, help='teacher temperature')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')
    # ViT Model parameters
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")
    
    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'lars'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    
    # misc
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='save frequency')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')

    os.environ["LOCAL_RANK"] = "0"
    args = parser.parse_args()
    if os.environ["LOCAL_RANK"] is not None:
        args.local_rank = int(os.environ["LOCAL_RANK"])
    return args 

def build_model(args):
    if args.arch in vits.__dict__.keys():
        encoder = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
    elif args.arch in resnet.__dict__.keys():
        encoder = resnet.__dict__[args.arch]
    model = SlotCon(encoder, args).cuda()

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    elif args.optimizer == 'lars':
        '''
        优化器方法-LARS(Layer-wise Adaptive Rate Scaling)
        '''
        optimizer = LARS(
            model.parameters(),
            lr=args.batch_size * args.world_size / 256 * args.base_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    return model, optimizer

def save_checkpoint(args, epoch, model, optimizer, scheduler, scaler=None):
    logger.info('==> Saving...')
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    if args.fp16:
        state['scaler'] = scaler.state_dict()
    file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
    torch.save(state, file_name)
    shutil.copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))

def load_checkpoint(args, model, optimizer, scheduler, scaler=None):
    if os.path.isfile(args.resume):
        logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')

        args.start_epoch = checkpoint['epoch'] + 1
        model.module.re_init(args)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        if args.fp16 and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume)) 

def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # prepare data     1：数据加载+预处理  2：算力GPU/CPU  3：模型
    '''
    transform  dataset  sampler   loader  model  optimizer   
    for epoch in range(args.start_epoch, args.epochs + 1) save_checkpoint
    在train内部:
            model.train()
            optimizer.zero_grad()
            loss.backward()#BP
            optimizer.step()#SGD
    '''
    transform = CustomDataAugmentation(args.image_size, args.min_scale)
    train_dataset = ImageFolder(args.dataset, args.data_dir, transform)
    #crops_transformed, coords, flags#   crop后的图像；坐标（便于后续逆增强）；翻转标志
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)# drop_last=True把最后一个小批次drop(总样本数量不是batch size整数倍时)

    args.num_instances = len(train_loader.dataset)
    logger.info(f"length of training dataset: {args.num_instances}")

    # create model
    logger.info("=> creating model '{}'".format(args.arch))
    model, optimizer = build_model(args)
    logger.info(model)

    # define scheduler
    scheduler = get_scheduler(optimizer, len(train_loader), args)
    # define scaler
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # optionally resume from a checkpoint
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')

    if args.resume:
        load_checkpoint(args, model, optimizer, scheduler, scaler)

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, scheduler, epoch, args)

        if dist.get_rank() == 0 and (epoch % args.save_freq == 0 or epoch == args.epochs):
            save_checkpoint(args, epoch, model, optimizer, scheduler, scaler)

def train(train_loader, model, optimizer, scaler, scheduler, epoch, args):
    batch_time = AverageMeter()
    '''使用 AverageMeter() 这个工具类来计算一个指标（Metric）的平均值和当前值。它通常用于训练过程中统计损失函数或准确率等变量的平均值。打log
    '''
    loss_meter = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    train_len = len(train_loader)
    for i, batch in enumerate(train_loader):
        '''crops_transformed, coords, flags#crop后的图像；相对于原始图像的坐标信息的“归一化”坐标（便于后续逆增强对齐两个dots）；flags是翻转标志'''
        crops, coords, flags = batch
        '''
        使用列表解析式将每个crop、coord和flag转换为CUDA张量，即将它们放入GPU设备的内存中进行计算，加快计算速度。
        其中non_blocking=True参数可以加速GPU张量的传输，不会阻塞CPU的运行。
        最后，将这些CUDA张量分别赋值给crops、coords和flags变量。
        '''
        crops = [crop.cuda(non_blocking=True) for crop in crops]
        coords = [coord.cuda(non_blocking=True) for coord in coords]
        flags = [flag.cuda(non_blocking=True) for flag in flags]

        # compute output and loss
        with torch.cuda.amp.autocast(scaler is not None):
            '''
            混合精度 (Automatically Mixed Precision, AMP) 训练已经成为了炼丹师的标配工具，仅仅只需几行代码，就能让显存占用减半，训练速度加倍。
            修改如下即可。仅需要加在output 与 loss上即可
            with torch.cuda.amp.autocast():
                output = net(input)
                loss = loss_fn(output, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            '''
            loss = model((crops, coords, flags))#因为call方法中调用了forward，故该语句实际上是执行model.forward()，输入：data，输出:loss
        
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()#BP
            optimizer.step()#SGD
        scheduler.step()

        # avg loss from batch size
        loss_meter.update(loss.item(), crops[0].size(0))
        '''loss.item() 用于获取当前 batch 的损失值，并将其更新到 loss_meter 中。最后，通过访问 loss_meter.avg 来获取损失值的平均值。'''
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (train_len - i)
            logger.info(
                f'Train: [{epoch}/{args.epochs}][{i}/{train_len}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.4f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  ')

if __name__ == '__main__':
    args = get_parser()
    '''分布式训练'''
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    cudnn.benchmark = True

    args.world_size = dist.get_world_size()
    args.batch_size = int(args.batch_size / args.world_size)
    '''# setup logger'''
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=args.output_dir,
                          distributed_rank=dist.get_rank(), name="slotcon")
    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    # print args
    logger.info(
        "\n".join("%s: %s" % (k, str(v))
                  for k, v in sorted(dict(vars(args)).items()))
    )

    main(args)
