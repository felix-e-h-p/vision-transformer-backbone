import wandb
import torch
import random
import argparse
import time  
import utils
import config
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from metrics import accuracy 
import models
from model import ViT
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from scheduler import WarmupCosineSchedule
from torchvision import datasets, transforms
from dataloader import ImageFolder
from torch.utils.data import DataLoader
import os

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

def main(rank,args) : 
    init_process(rank,args.world_size)
    
    if dist.get_rank() ==0:
        wandb.init(project = 'vision transformer',
                  )
        wandb.config.update(args)                             
    else : 
        wandb.init(project = 'vision transformer',
                   mode = 'disabled')
        wandb.config.update(args)
    
    st = time.time()

if dist.get_rank() == 0 : 
        print('data load',time.time()-st)
    
    args.batch_size = int(args.batch_size / args.world_size)
    args.workers = int((args.workers + args.world_size - 1) / args.world_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)
    
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,sampler=train_sampler)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=False)

