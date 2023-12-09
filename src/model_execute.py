# Vision Transformer Training Script

# This script performs distributed training of a Vision Transformer (ViT) model on a given dataset. 
# It utilizes PyTorch's Distributed Data Parallelism for parallel training across multiple GPUs.

# Usage:
#     python train.py --world_size <num_gpus> --mode <train/val>

# Arguments:
#     --data: Dataset to use (default: 'imagenet')
#     --data_path: Path to the dataset (default: './path')
#     --model: Model architecture (default: 'vit')
#     --batch_size: Input batch size for training (default: 512)
#     --epochs: Number of epochs to train (default: 300)
#     --lr: Learning rate (default: 0.001)
#     --weight_decay: Adam weight decay (default: 0.3)
#     --t_max: Cosine annealing steps (default: 80000)
#     --seed: Random seed (default: 1)
#     --log-interval: Batches to wait before logging training status (default: 10)
#     --alpha: Alpha parameter (default: 0.9)
#     --world_size: Number of GPUs (default: 4)
#     --workers: Number of data loader workers (default: 4)
#     --gpu: GPU device(s) to use (default: '0')
#     --mode: 'train' for training, 'val' for validation (default: 'None')

# Example:
#     python train.py --world_size 4 --mode train

import wandb
import torch
import random
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms
from torch.utils.data import DataLoader
from models import ViT
from dataloader import ImageFolder
from utils import accuracy, reduce_tensor, init_process, cleanup

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
      
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

parser = argparse.ArgumentParser(description='vision transformer')

parser.add_argument('--data', type=str, default='imagenet', metavar='N',
                    help='data')
parser.add_argument('--data_path', type=str, default='./path', metavar='N',
                    help='data') 
parser.add_argument('--model', type=str, default='vit', metavar='N',
                    help='model')
parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--weight_decay', type=float, default=0.3, metavar='M',
                    help='adam weight_decay (default: 0.5)')
parser.add_argument('--t_max', type=float, default=80000, metavar='M',
                    help='cosine annealing steps')                    
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--alpha', type=int, default=0.9, metavar='N',
                    help='alpha')
parser.add_argument('--world_size', type=int, default=4, metavar='N',
                    help='world_size')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='num_workers')                                         
parser.add_argument('--gpu',type=str,default='0',
                    help = 'gpu')
parser.add_argument('--mode',type=str,default='None',
                    help = 'train/val mode')

args = parser.parse_args()

cudnn.benchmark=True 

def main(rank, args):
    """
    Main function for distributed training of Vision Transformer.

    Args:
        rank (int): Rank of the current process.
        args (argparse.Namespace): Parsed command-line arguments.
    """
    init_process(rank, args.world_size)

    if dist.get_rank() == 0:
        wandb.init(project='vision_transformer')
        wandb.config.update(args)
    else:
        wandb.init(project='vision_transformer', mode='disabled')
        wandb.config.update(args)

    st = time.time()

    if dist.get_rank() == 0:
        print('data load', time.time() - st)

    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=False,
                              num_workers=args.workers,
                              pin_memory=True,
                              sampler=train_sampler)

    val_loader = DataLoader(val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.workers,
                            pin_memory=False)

    model = ViT(in_channels=3,
                patch_size=16,
                emb_size=768,
                img_size=224,
                depth=12,
                n_classes=1000)

    torch.cuda.set_device(rank)

    if args.mode == 'train':
        model = model.cuda(rank)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        wandb.watch(model)

        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr,
                               weight_decay=args.weight_decay)

        t_total = (len(train_loader.dataset) / args.batch_size) * args.epochs
        print(f"total_step : {t_total}")

        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=args.t_max,
                                      eta_min=0)

        criterion = nn.CrossEntropyLoss().cuda(rank)

        max_norm = 1

        scaler = torch.cuda.amp.GradScaler()

        model.train()

        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            for batch_idx, (data, target) in enumerate(train_loader):
                st = time.time()
                data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
                optimizer.zero_grad()

                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # iter
                train_loss = reduce_tensor(loss.data, dist.get_world_size())

                if dist.get_rank() == 0:
                    wandb.log({'train_batch_loss': train_loss.item()})
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), int(len(train_loader.dataset) / args.world_size),
                        100. * batch_idx / len(train_loader), train_loss))
                    print('teacher_network iter time is {0}s'.format(time.time() - st))

            if dist.get_rank() == 0:
                print('save checkpoint')

                if epoch % 15 == 0:
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, filename=f'checkpoint_{epoch}.pth.tar')

        model.eval()
        correct = 0
        total_acc1 = 0
        total_acc5 = 0
        step = 0
        st = time.time()

        for batch_idx, (data, target) in enumerate(val_loader):
            with torch.no_grad():
                data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
                output = model(data)
            val_loss = criterion(output, target)
            val_loss = reduce_tensor(val_loss.data, dist.get_world_size())

            if dist.get_rank() == 0:
                wandb.log({'val_batch_loss': val_loss.item()})

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            total_acc1 += acc1[0]
            total_acc5 += acc5[0]
            step += 1

        if dist.get_rank() == 0:
            print(f"[{batch_idx * len(data)}/{int(len(val_loader.dataset) / args.world_size)}, "
                  f"top1-acc : {acc1[0]}, top5-acc : {acc5[0]}]")

        if dist.get_rank() == 0:
            print('\nval set: top1: {}, top5 : {} '.format(total_acc1 / step, total_acc5 / step))
            wandb.log({'top1': total_acc1 / step})
            wandb.log({'top5': total_acc5 / step})
        print(f"validation time : {time.time() - st}")

        if args.mode == 'val':
            checkpoint = torch.load('ckpt path')
            model = model.cuda(rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
            model.load_state_dict(checkpoint['state_dict'])

            optimizer = optim.Adam(model.parameters(),
                                   lr=args.lr,
                                   weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])

            criterion = nn.CrossEntropyLoss().cuda(rank)

            model.eval()
            correct = 0
            total_acc1 = 0
            total_acc5 = 0

            st = time.time()
            for batch_idx, (data, target) in enumerate(val_loader):
                with torch.no_grad():
                    data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
                    output = model(data)
                val_loss = criterion(output, target)
                val_loss = reduce_tensor(val_loss.data, dist.get_world_size())
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                total_acc1 += acc1[0]
                total_acc5 += acc5[0]

            if dist.get_rank() == 0:
                print(f"[{batch_idx * len(data)}/{int(len(val_loader.dataset) / args.world_size)}, "
                      f"top1-acc : {acc1[0]}, top5-acc : {acc5[0]}]")

            if dist.get_rank() == 0:
                print('\nval set: top1: {}, top5 : {} '.format(
                    total_acc1 / step, total_acc5 / step))
                wandb.log({'top1': total_acc1 / step})
                wandb.log({'top5': total_acc5 / step})

            print(f"validation time : {time.time() - st}")

        cleanup()
        wandb.finish()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint.

    Args:
        state (dict): Model state dictionary.
        filename (str): Name of the checkpoint file.
    """
    torch.save(state, filename)

if __name__ == '__main__':
    mp.spawn(main, nprocs=args.world_size, args=(args,))