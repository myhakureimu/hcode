import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import time
import random
import numpy as np
import wandb

from hdataloader import HDataLoader


matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# python train.py --name=GPT
parser = argparse.ArgumentParser(description='PyTorch In-context Learning Training Code')
parser.add_argument('--gpu', default='0', type=str, help='which gpus to use')
parser.add_argument('--random_seed', default=1, type=int, help='the seed used for torch & numpy')

#arxived args
parser.add_argument('--SigmaRe', default=0, type=int)
parser.add_argument('--NormAtt', default=0, type=int)
parser.add_argument('--FirstLayerNorm', default=1, type=int)

parser.add_argument('--wandb', default=1, type=int)
parser.add_argument('--early_stop', default=0, type=int)

#experiment aim
parser.add_argument('--expName', default='Hypothesis Testing test', type=str)


parser.add_argument('--train', default='train1', type=str)
parser.add_argument('--n', default=8, type=int)
parser.add_argument('--m', default=2**6, type=int)
parser.add_argument('--k', default=9, type=int)
print_index = [0,1,2,4,8,16,32,64,128]

#model section
parser.add_argument('--modelName', default='nano', type=str)
parser.add_argument('--scale', default=1, type=int, help='scale')
parser.add_argument('--num_heads', default=2, type=int, help='number of heads for multi-headed attention (default: 8)')
parser.add_argument('--depth', default=2*4, type=int, help='depth of the transformer architecture (default: 12)')
parser.add_argument('--embed_dim', default=128, type=int, help='embedding dimension of the transformer feature extractor (default: 256)')

parser.add_argument('--llm_max_length', default=512, type=int, help='maximum sequence length of the input (default: 11)')


#optimization
parser.add_argument('--lr', default=0.0005, type=float, help='initial model learning rate') #0.0005
parser.add_argument('--wd', default=0.1, type=float, help='weight decay hyperparameter (default: 0.00001)')
parser.add_argument('--batch_size', default=32, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--n_steps', default=512, type=int, help='total number of training steps we want to run')
parser.add_argument('--epochs', default=5, type=int, help='number of total epochs to run')


parser.set_defaults(augment=True)

# Specifying gpu usage using cli hyperparameters
args = parser.parse_args()

args.num_heads = args.scale * args.num_heads
#args.depth     = args.scale * args.depth
args.embed_dim = args.scale * args.embed_dim

print_index = [x for x in print_index if x < args.k]

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
setproctitle.setproctitle(args.expName)

import torch
import torch.nn as nn
from utils import TransformerModel, FlexDataset
from utils.nano_gpt import GPTConfig, NanoGPT

# Seeding programmatic sources of randomness
torch.backends.cudnn.benchmark = True
#torch.manual_seed(args.random_seed)
#np.random.seed(args.random_seed)
#random.seed(args.random_seed)

def add(x, y):
    """
    Add two arrays `x` and `y` of possibly different sizes element-wise.
    
    If both x and y exist at index i, z[i] = x[i] + y[i].
    If only one exists, z[i] = x[i] or z[i] = y[i], whichever exists.
    
    Parameters:
    - x (np.array): First array of size m.
    - y (np.array): Second array of size n.
    
    Returns:
    - np.array: The element-wise sum of x and y, with handling for different sizes.
    """
    # Convert x to a size-1 array if it is an int, float, or 0D tensor
    if isinstance(x, (int, float)):
        x = np.array([x])
    elif isinstance(x, list):
        x = np.array(x)
    elif isinstance(x, torch.Tensor) and x.ndim == 0:  # Check for 0D tensor
        x = np.array([x.cpu().item()])
    elif isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # Convert y to a size-1 array if it is an int, float, or 0D tensor
    if isinstance(y, (int, float)):
        y = np.array([y])
    elif isinstance(y, list):
        y = np.array(y)
    elif isinstance(y, torch.Tensor) and y.ndim == 0:  # Check for 0D tensor
        y = np.array([y.cpu().item()])
    elif isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    
    m = len(x)
    n = len(y)
    
    # Create an output array z with the size of the longer array
    z = np.zeros(max(m, n))
    
    # Add elements where both x and y exist
    for i in range(min(m, n)):
        z[i] = x[i] + y[i]
    
    # Copy remaining elements from x if x is longer
    if m > n:
        z[n:] = x[n:]
    
    # Copy remaining elements from y if y is longer
    elif n > m:
        z[m:] = y[m:]
    
    return z

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = add(self.sum, val)
        self.count = add(self.count, n)
        self.avg = self.sum / (0.000001+self.count)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt is the prediction probability
        focal_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
if args.train == 'train1':
    print_index = print_index

def train_model(args, split, HManager, model, optimizer, epoch):
    if split == 'train1':
        dataloader = HManager.get_pytorch_dataloader(batch_size=args.batch_size, dataloader_type=split, prefix_repeat=None)
    if split == 'train2':
        dataloader = HManager.get_pytorch_dataloader(batch_size=args.batch_size, dataloader_type=split, prefix_repeat=None)
    if split == 'test1':
        dataloader = HManager.get_pytorch_dataloader(batch_size=args.batch_size, dataloader_type='test', prefix_repeat=None)
    if split == 'test2':
        dataloader = HManager.get_pytorch_dataloader(batch_size=args.batch_size, dataloader_type='test', prefix_repeat=2)
    if split == 'test4':
        dataloader = HManager.get_pytorch_dataloader(batch_size=args.batch_size, dataloader_type='test', prefix_repeat=4)
    if split == 'test8':
        dataloader = HManager.get_pytorch_dataloader(batch_size=args.batch_size, dataloader_type='test', prefix_repeat=8)
    
    #loss_f = FocalLoss(reduction = 'none') #
    loss_f = torch.nn.BCEWithLogitsLoss(reduction = 'none')
    
    if split in ['train1', 'train2']:
        batch_time = AverageMeter()
        batch_loss = AverageMeter()
        batch_acc_ = AverageMeter()
        batch_loss_icl = AverageMeter()
        batch_acc__icl = AverageMeter()
        model.train()
    if split in ['test1', 'test2', 'test4', 'test8']:
        batch_acc_ = AverageMeter()
        batch_acch = AverageMeter()
        model.eval()
    
    #D = {}
    if split in ['train1', 'train2']:
        for xs, ys, hs, idendify_xs, masks in (pbar := tqdm(dataloader)):
            #D[str(xs)+'=>'+str(ys)] = 0
            #print(masks.shape)
            xs, ys, hs, masks = xs.cuda(), ys.cuda(), hs.cuda(), masks.cuda()
            xs = xs# + 0.1*torch.randn(xs.shape).cuda()
            hatys = model.forward2(xs, ys)

            # Calculate CE Loss
            # print('hatys', hatys.shape)
            # print('ys', ys.shape)
            losses = loss_f(hatys, ys)
            # print('losses', losses.shape)
            # print('masks', masks.shape)
            #loss = torch.sum(losses)
            # print(masks)
            loss = torch.sum(losses              *masks)/torch.sum(masks)
            # print('(hatys >= 0) == ys)', ((hatys >= 0) == ys).shape)
            acc_ = torch.sum(((hatys >= 0) == ys)*masks)/torch.sum(masks)
            with torch.no_grad():
                loss_icl = torch.sum(losses              *masks, dim=0)
                acc__icl = torch.sum(((hatys >= 0) == ys)*masks, dim=0)
                count_icl = torch.sum(masks, dim=0)
                
            # Record the loss and elapsed time
            batch_loss.update(loss.data)
            batch_acc_.update(acc_.data)
            #print(loss_icl.shape)
            #print(count_icl.shape)
            batch_loss_icl.update(loss_icl.data, count_icl.data)
            batch_acc__icl.update(acc__icl.data, count_icl.data)

            # Backpropagate gradients and update model
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            #print(batch_loss.avg, batch_acc_.avg)
            #print(batch_loss.avg[0], batch_acc_.avg[0])
            pbar.set_description(f"train {batch_loss.avg[0]:.3f} {batch_acc_.avg[0]:.3f}")
            
        wandb_info={
            "train/loss": batch_loss.avg,
            "train/acc_": batch_acc_.avg,
            }
        
        wandb_info["train/loss"] = batch_loss.avg
        wandb_info["train/acc_"] = batch_acc_.avg
        if args.train == 'train1':
            print('train/loss:', batch_loss_icl.avg[print_index])
            print('train/acc_:', batch_acc__icl.avg[print_index])
        else:
            print('train/loss:', batch_loss_icl.avg)
            print('train/acc_:', batch_acc__icl.avg)
    
    #print(D)
    if split in ['train1', 'train2']:
        with torch.no_grad():
            for xs, ys, hs, idendify_xs, masks in (pbar := tqdm(dataloader)):
                
                xs, ys, hs, masks = xs.cuda(), ys.cuda(), hs.cuda(), masks.cuda()
                
                #print(hs)
                #print(model._combine2(xs, ys))
                
                hatys = model.forward2(xs, ys)
                
                # Calculate CE Loss
                losses = loss_f(hatys, ys)
                loss = torch.sum(losses              *masks)/torch.sum(masks)
                acc_ = torch.sum(((hatys >= 0) == ys)*masks)/torch.sum(masks)
                with torch.no_grad():
                    loss_icl = torch.sum(losses              *masks, dim=0)
                    acc__icl = torch.sum(((hatys >= 0) == ys)*masks, dim=0)
                    count_icl = torch.sum(masks, dim=0)
                    
                # Record the loss and elapsed time
                batch_loss.update(loss.data)
                batch_acc_.update(acc_.data)
                batch_loss_icl.update(loss_icl.data, count_icl.data)
                batch_acc__icl.update(acc__icl.data, count_icl.data)
        
                # Backpropagate gradients and update model
                # optimizer.zero_grad()
                # model.zero_grad()
                # loss.backward()
                # optimizer.step()

                pbar.set_description(f"train {batch_loss.avg[0]:.3f} {batch_acc_.avg[0]:.3f}")
        
            wandb_info={
                "valid/loss": batch_loss.avg,
                "valid/acc_": batch_acc_.avg,
                }
            
            wandb_info["valid/loss"] = batch_loss.avg
            wandb_info["valid/acc_"] = batch_acc_.avg
            if args.train == 'train1':
                print('valid/loss:', batch_loss_icl.avg[print_index])
                print('valid/acc_:', batch_acc__icl.avg[print_index])
            else:
                print('valid/loss:', batch_loss_icl.avg)
                print('valid/acc_:', batch_acc__icl.avg)
    

    if split in ['test1', 'test2', 'test4', 'test8']:
        with torch.no_grad():
            for xs, ys, hs, idendify_xs, masks in (pbar := tqdm(dataloader)):

                xs, ys, hs, masks = xs.cuda(), ys.cuda(), hs.cuda(), masks.cuda()
                #print(hs)
                #print(model._combine2(xs, ys))
                hatys = model.forward2(xs, ys)
                # print('xs', xs)
                # print('ys', ys)
                # print('hatys', hatys)
                # print('hatys', 1*(hatys >= 0))
                # print('hs', hs)
                
                #print(xs)
                #print(ys)
                # print('masks', masks.shape)
                # print('hatys', hatys.shape)
                # print('ys', ys.shape)
                # print('(hatys >= 0) == ys)', ((hatys >= 0) == ys).shape)
                # print(hatys)
                acc_ = torch.sum(((hatys >= 0) == ys)*masks)/torch.sum(masks)
                acch = (acc_ == 1)
                batch_acc_.update(acc_.data)
                batch_acch.update(acch.data)

                pbar.set_description(f"test {batch_acc_.avg[0]:.3f} {batch_acch.avg[0]:.3f}")

            wandb_info={
                split+"/acc_": batch_acc_.avg,
                split+"/acch": batch_acch.avg,
                }
            
            wandb_info[split+"/acc_"] = batch_acc_.avg
            wandb_info[split+"/acch"] = batch_acch.avg

    return wandb_info


if 1:
    topFile = args.expName
    hdata_hypers = 'm='+str(args.m) \
             +'_'+ 'n='+str(args.n)
    model_hypers = 'modelName='+str(args.modelName) \
             +'_'+ 'depth='+str(args.depth) \
             +'_'+ 'dim='+str(args.embed_dim) \
             +'_'+ 'heads='+str(args.num_heads)
    optim_hypers = 'lr='+str(args.lr) \
             +'_'+ 'wd='+str(args.wd) \
             +'_'+ 'BS='+str(args.batch_size) \
             +'_'+ 'Step='+str(args.n_steps) \
             +'_'+ 'EP='+str(args.epochs)
    
    # wandb
    if args.wandb:
        wandb.login(key='0e030fcc130348fb3127f6140ac82c773fa4b4d9')
        
        if args.train == 'train1':
            name = f'method={args.train} k={args.k} seed={args.random_seed}'
        if args.train == 'train2':
            name = f'method={args.train} seed={args.random_seed}'
        run = wandb.init(
            # Set the project where this run will be logged
            project= f'{args.expName} n={args.n} m={args.m}|{2**args.n}',
            name = name,
            dir='../wandb',
            # Track hyperparameters and run metadata
            config={
                'n(feature)': args.n,
                'm(subsample)': args.m,
                'seed': args.random_seed,
                'k': args.k,
                'depth': args.depth,
                'dim': args.embed_dim,
                'heads': args.num_heads,
                'lr': args.lr,
                'wd': args.wd,
                'BS': args.batch_size,
                'Step': args.n_steps,
                'EP': args.epochs
            },
        )
        wandb.define_metric("*", step_metric="global_step")
    
    # folder
    print('***** ' + topFile + ' *****')
    print('***** ' + hdata_hypers + ' *****')
    print('***** ' + model_hypers + ' *****')
    print('***** ' + optim_hypers + ' *****')
    folder = '../data/'+topFile+'/'+hdata_hypers+'/'+model_hypers+'/'+optim_hypers+'/'
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Initialize the data loader
    print(args.n, args.m, args.k, args.n_steps)
    HManager = HDataLoader(n=args.n, m=args.m, random_seed=args.random_seed, k=args.k, n_steps=args.n_steps)

    
    # model
    if 0: #GPT2
        model = TransformerModel(n_dims=args.vocab_size, n_positions=args.num_k, 
                                 n_embd=args.embed_dim, n_layer=args.depth, 
                                 n_head=args.num_heads, special_dimension=True)
    if args.modelName == 'nano': #nanoGPT
        config = GPTConfig(
            input_dim = args.n + 1,
            block_size = args.llm_max_length,
            #vocab_size = 50304, # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
            n_layer = args.depth,
            n_head = args.num_heads,
            n_embd = args.embed_dim,
            dropout = 0.0,
            bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
            SigmaRe = args.SigmaRe,
            NormAtt = args.NormAtt,
            FirstLayerNorm = args.FirstLayerNorm,
        )
        model = NanoGPT(config)
        
    model.cuda()
    print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas = (0.9, 0.999))
    
    
    # Iterating through training epochs
    if not os.path.exists(folder+'checkpoint'):
        os.makedirs(folder+'checkpoint')
    
    
    # print('******** EP = ' +str(0)+ ' / ' +str(args.epochs)+ ' *******')
    # epoch = 0
    # split = 'test'
    # wandb_valid_info = train_model(args, split, HManager, model, optimizer, epoch=epoch)
    # if args.wandb:
    #     wandb_valid_info['global_step'] = epoch
    #     wandb.log(wandb_valid_info)
                
    for epoch in range(1, args.epochs+1):
        print('******** EP = ' +str(epoch)+ ' / ' +str(args.epochs)+ ' *******')
        
        if 1: #train
            split = args.train
            wandb_train_info = train_model(args, split, HManager, model, optimizer, epoch=epoch)
            if args.wandb:
                wandb_train_info['global_step'] = epoch
                wandb.log(wandb_train_info)
            
            #if epoch % 10 == 0:
            #    state = {"model_state_dict": model.state_dict(), 
            #             "optimizer_state_dict": optimizer.state_dict(),
            #             "train_epoch": epoch}
            #    torch.save(state, state_file)
            
        if 1: #evaluation
            split = 'test1'
            wandb_valid_info = train_model(args, split, HManager, model, optimizer, epoch=epoch)
            if args.wandb:
                wandb_valid_info['global_step'] = epoch
                wandb.log(wandb_valid_info)
            
            split = 'test2'
            wandb_valid_info = train_model(args, split, HManager, model, optimizer, epoch=epoch)
            if args.wandb:
                wandb_valid_info['global_step'] = epoch
                wandb.log(wandb_valid_info)

            split = 'test4'
            wandb_valid_info = train_model(args, split, HManager, model, optimizer, epoch=epoch)
            if args.wandb:
                wandb_valid_info['global_step'] = epoch
                wandb.log(wandb_valid_info)
         
        # import pickle
        # exp_record_folder = folder + 'exp_record'
        # with open(exp_record_folder +'.pkl', 'wb') as file:
        #     pickle.dump(exp_record, file)
