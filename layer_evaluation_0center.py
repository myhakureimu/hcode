import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from setting import Circle, Regular4, Regular6, Regular8, Regular12, D ,PriorProcesser
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')
# python train.py --name=GPT
parser = argparse.ArgumentParser(description='PyTorch In-context Learning Training Code')
parser.add_argument('--gpu', default='0', type=str, help='which gpus to use')
parser.add_argument('--dataset', default="gaussian", type=str, help='distribution used to generate the dataset')
parser.add_argument('--model_arch', default='gpt', type=str, help='the model architecture used')
parser.add_argument('--print_freq', default=75, type=int, help='print frequency (default: 75)')
parser.add_argument('--random_seed', default=1, type=int, help='the seed used for torch & numpy')
parser.add_argument('--base_dir', default="./", type=str, help='base directory')

parser.add_argument('--num_heads', default=8, type=int, help='number of heads for multi-headed attention (default: 8)')
parser.add_argument('--depth', default=20, type=int, help='depth of the transformer architecture (default: 12)')
parser.add_argument('--embed_dim', default=256, type=int, help='embedding dimension of the transformer feature extractor (default: 256)')
parser.add_argument('--steps', default=5000, type=int, help='total number of training steps we want to run')#5000
parser.add_argument('--lr', default=0.00001, type=float, help='initial model learning rate')
parser.add_argument('--wd', default=0.00001, type=float, help='weight decay hyperparameter (default: 0.00001)')
parser.add_argument('--batch_size', default=256, type=int, help='mini-batch size (default: 64)')
parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
parser.add_argument('--num_k', default=16, type=int, help='maximum sequence length of the input (default: 11)')

parser.add_argument('--exp_name', default='Regular4_delta', type=str)

parser.add_argument('--a', default=4, type=int, help='1/b**a')
parser.add_argument('--b', default=4, type=int, help='1/b**a')

parser.add_argument('--M', default=0, type=int, help='M')

parser.add_argument('--input_dim', default=2, type=int, help='d=M')

parser.set_defaults(augment=True)


# Specifying gpu usage using cli hyperparameters
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
setproctitle.setproctitle(args.exp_name)

import time
import torch
import random
import numpy as np
from utils import TransformerModel, FlexDataset
import pickle
from umap import UMAP
import joblib
#from torch.distributions.multivariate_normal import MultivariateNormal
#from mydataloader import Prior

# Seeding programmatic sources of randomness
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)
random.seed(args.random_seed)

    
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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_model(args, prior, model, optimizer, epoch):
    batch_time = AverageMeter()
    batch_noisedloss = AverageMeter()
    sum_batch_noisedloss = 0
    batch_cleanloss  = AverageMeter()
    sum_batch_cleanloss  = 0
    with torch.no_grad():
        # Switch model to train mode
        model.eval()
        end = time.time()
        #for i in tqdm(range(args.steps)):
        for i in tqdm(range(100)):
    
            # Loop through batch of inputs
            #for sample_idx in range(len(inputs)):
            #input_ = torch.stack([inputs[sample_idx]]).cuda()
            #target = torch.stack([targets[sample_idx]]).cuda()
            
            input_, target, label = prior.draw_sequences(bs=args.batch_size, k=args.num_k)
            
            input_ = torch.from_numpy(input_).cuda().float()
            target = torch.from_numpy(target).cuda().float()
            label  = torch.from_numpy(label ).cuda().float()
            # Forward through the gpt model
            output = model.forward2(input_, target)
            
            # Calculate the squared error loss
            loss = (target - output).square().mean()
            
            with torch.no_grad():
                cleanloss = (label - output).square().mean()
            '''
            # Backpropagate gradients and update model
            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            optimizer.step()
            '''
            # Record the loss and elapsed time
            batch_noisedloss.update(loss.data)
            batch_cleanloss .update(cleanloss.data)
            
            sum_batch_noisedloss += loss.data
            sum_batch_cleanloss  += cleanloss.data
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            
            # Print training dataset logs
            #end = time.time()
            #if i % args.print_freq == 0:
            #    print('Epoch: {epoch} [{step}/{total_steps}]\t'
            #          'Time {time.val:.3f} ({time.avg:.3f})\t'
            #          'MSE Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
            #              epoch=epoch, step=i, total_steps=args.steps, 
            #              time=batch_time, loss=batch_loss))
        print(batch_noisedloss.avg, sum_batch_noisedloss/100)
        print(batch_cleanloss.avg , sum_batch_cleanloss /100)
    return model, optimizer


if 1:
    print(1)
    assert args.model_arch == 'gpt'
    k_list = [i+1 for i in range(args.num_k)]
    k_list = [a-1 for a in k_list]
    k_array = np.array(k_list)
    k_list = [r'${}$'.format(a) for a in k_list]
    
    D_load = 0
    B_flag = 1
    B_load = 0
    T_flag = 1
    T_load = 0
    
    KK = 100
    BS = args.batch_size
    K = KK*BS
    z = 1
    linewidth = 3.0
    
    if args.exp_name == 'Regular4_delta':
        fig_name = 'regular4_base='+str(args.b)+'_lr='+str(args.lr)+'_Depth='+str(args.depth)+'_EmbDim='+str(args.embed_dim)+'_BS='+str(args.batch_size)+'_Step='+str(args.steps)
    elif args.exp_name == 'RegularM_M':
        fig_name = 'regularM'                  +'_lr='+str(args.lr)+'_Depth='+str(args.depth)+'_EmbDim='+str(args.embed_dim)+'_BS='+str(args.batch_size)+'_Step='+str(args.steps)
    elif args.exp_name == 'D_d':
        fig_name = 'D_d'                       +'_lr='+str(args.lr)+'_Depth='+str(args.depth)+'_EmbDim='+str(args.embed_dim)+'_BS='+str(args.batch_size)+'_Step='+str(args.steps)
    
    hypers_list = []
    if args.exp_name == 'Regular4_delta':
        delta_m_list = [1/args.b**args.a]
        delta_w_list = [1/args.b**args.a]
        for delta_m, delta_w in zip(delta_m_list, delta_w_list):
            hypers_list.append({'delta_m': delta_m,
                                'delta_w': delta_w})
    elif args.exp_name == 'D_d':
        d_list = [args.input_dim]
        delta_m = 1/16 #1/10
        delta_w = 1/16 #1/10
        for d in d_list:
            hypers_list.append({'d': d})
    for hypers in hypers_list:
        if args.exp_name == 'Regular4_delta':
            delta_m = hypers['delta_m']
            delta_w = hypers['delta_w']
            print('***** delta_m=' +str(delta_m)+ ' / delta_w=' +str(delta_w)+ ' *****')
            folder = 'data/Transformer/'+fig_name+'/delta_m='+str(delta_m)+' delta_w='+str(delta_w)+'/'
            prior = Regular4(delta_m**0.5, delta_m**0.5)
            priorProcesser = PriorProcesser(prior)
        if args.exp_name == 'D_d':
            d = hypers['d']
            print('***** d =' +str(d)+ ' *****')
            folder = 'data/Transformer/'+fig_name+'/d='+str(d)+'/'
            
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        if 1: # data
            if not D_load:
                #bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(K, args.num_k)
                bs_xs, bs_ys, bs_retrieval = priorProcesser.draw_sequences(K, args.num_k)
                bs_learning = bs_retrieval
                np.save(folder+'bs_xs.npy', bs_xs)
                np.save(folder+'bs_ys.npy', bs_ys)
                np.save(folder+'bs_retrieval.npy', bs_retrieval)
                np.save(folder+'bs_learning.npy', bs_learning)
            else:
                bs_xs = np.load(folder+'bs_xs.npy')
                bs_ys = np.load(folder+'bs_ys.npy')
                bs_retrieval = np.load(folder+'bs_retrieval.npy')
                bs_learning = np.load(folder+'bs_learning.npy')
        '''
        if B_flag:
            if not B_load:
                B_preds = np.zeros([K, args.num_k])
                for k in tqdm(range(1, args.num_k+1,1)):
                    for j in range(K):
                        xs, ys = bs_xs[j][:k], bs_ys[j][:k]
                        returned_dict = priorProcesser.predict(xs, ys, priorProcesser.topic_ws[0])
                        B_preds[j, k-1] = returned_dict['prediction']
                
                np.save(folder+'B_preds.npy', B_preds)
            else:
                B_preds = np.load(folder+'B_preds.npy')
        '''
        if T_flag:
            # Loading in gpt model for training
            model = TransformerModel(n_dims=priorProcesser.d, n_positions=args.num_k, 
                                     n_embd=args.embed_dim, n_layer=args.depth, 
                                     n_head=args.num_heads, special_dimension=True)
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        
        # Iterating through training epochs
        for epoch in range(1, args.epochs+1,9):
            print('******** EP = ' +str(epoch)+ ' *******')
            if not os.path.exists(folder+'EP=' +str(epoch)):
                os.makedirs(folder+'EP=' +str(epoch))
            state_folder = folder+'EP=' +str(epoch)+'/checkpoint_'+str(epoch)+'.pth'
            
            if 0: # Train gpt model for one epoch on generated dataset
                if T_flag and (not T_load):
                    model, optimizer = train_model(args, priorProcesser, model, optimizer, epoch=epoch)
                    state = {"model_state_dict": model.state_dict(), 
                             "optimizer_state_dict": optimizer.state_dict(),
                             "train_epoch": epoch}
                    #torch.save(state, state_folder)
            
            if 0: #evaluation
                state = torch.load(state_folder)
                model.load_state_dict(state['model_state_dict'])
                optimizer.load_state_dict(state['optimizer_state_dict']) 
                if T_flag:
                    if not T_load:
                        model.eval()
                        T_preds = []
                        #with torch.no_grad():
                        #    for kk in tqdm(range(KK)):
                        #        T_preds.append( model(torch.from_numpy(bs_xs[kk*BS:(kk+1)*BS]).cuda().float(), torch.from_numpy(bs_ys[kk*BS:(kk+1)*BS]).cuda().float()).cpu().numpy() )
                        #T_preds = np.concatenate(T_preds, axis=0)
                        summation = 0
                        noisedLoss_sum = 0
                        cleanLoss_sum = 0
                        with torch.no_grad():
                            for kk in tqdm(range(KK)):
                                answer2 = model.forward2(torch.from_numpy(bs_xs[kk*BS:(kk+1)*BS]).cuda().float(), torch.from_numpy(bs_ys[kk*BS:(kk+1)*BS]).cuda().float()).cpu().numpy()
                                answer3 = model.forward3(torch.from_numpy(bs_xs[kk*BS:(kk+1)*BS]).cuda().float(), torch.from_numpy(bs_ys[kk*BS:(kk+1)*BS]).cuda().float())
                                answer3 = answer3['final'].cpu().numpy()
                                summation += np.sum(np.abs(answer2-answer3))
                                noisedLoss_sum += np.mean((answer2-bs_ys      [kk*BS:(kk+1)*BS])**2)
                                cleanLoss_sum  += np.mean((answer2-bs_learning[kk*BS:(kk+1)*BS])**2)
                                #T_preds.append( model(torch.from_numpy(bs_xs[kk*BS:(kk+1)*BS]).cuda().float(), torch.from_numpy(bs_ys[kk*BS:(kk+1)*BS]).cuda().float()).cpu().numpy() )
                        #T_preds = np.concatenate(T_preds, axis=0)
                        print('diff: ', summation)
                        print('noisedLoss_sum: ', noisedLoss_sum/KK)
                        print('cleanLoss_sum : ', cleanLoss_sum /KK)
                        #np.save(folder+'EP=' +str(epoch)+ '/T_preds.npy', T_preds)
                    else:
                        T_preds = np.load(folder+'EP=' +str(epoch)+ '/T_preds.npy')

from sklearn.manifold import TSNE
if args.depth == 10:
    L_list = [0,3,4,5,7,10]
if args.depth == 20:
    L_list = [0,3,5,7,10,20]

if 0: #explore train visualizer
    epoch = 10
    state_folder = folder+'EP=' +str(epoch)+'/checkpoint_'+str(epoch)+'.pth'
    state = torch.load(state_folder)
    model.load_state_dict(state['model_state_dict'])
    #optimizer.load_state_dict(state['optimizer_state_dict']) 
    model.eval()
    
    with torch.no_grad():
        new_task_m = priorProcesser.topic_ms[0]*0
        
        new_task_w = priorProcesser.topic_ws[1]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=1024*2, k=args.num_k, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction1 = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction1-bs_learning)**2, axis=0)[[0,3,6,9,12,15]])
        hiddens1 = results['hiddens']
        bs_task1 = bs_xs@priorProcesser.topic_ws[1] 
        print('1 to 1 ', np.mean((prediction1-bs_task1)**2, axis=0)[[0,3,6,9,12,15]])
        bs_task2 = bs_xs@priorProcesser.topic_ws[2] 
        print('1 to 2 ', np.mean((prediction1-bs_task2)**2, axis=0)[[0,3,6,9,12,15]])
        bs_task3 = bs_xs@priorProcesser.topic_ws[3] 
        print('1 to 3 ', np.mean((prediction1-bs_task3)**2, axis=0)[[0,3,6,9,12,15]])
        
        new_task_w = priorProcesser.topic_ws[2]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=1024*2, k=args.num_k, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction2 = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction2-bs_learning)**2, axis=0)[[0,3,6,9,12,15]])
        hiddens2 = results['hiddens']
        
        new_task_w = priorProcesser.topic_ws[3]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=1024*2, k=args.num_k, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction3 = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction3-bs_learning)**2, axis=0)[[0,3,6,9,12,15]])
        hiddens3 = results['hiddens']
        
        
        if not os.path.exists(folder+'prediction'):
            os.makedirs(folder+'prediction')
        plt.hist(prediction1)
        plt.savefig(folder+'prediction/1.png')
        plt.hist(prediction2)
        plt.savefig(folder+'prediction/2.png')
        plt.hist(prediction3)
        plt.savefig(folder+'prediction/3.png')
        
        if not os.path.exists(folder+'UMAPModel'):
            os.makedirs(folder+'UMAPModel')
        
        
        for col, L in enumerate(L_list):
            features = []
            for row, k in enumerate(np.arange(0, args.num_k, 3)):
                
                f1 = hiddens1[L][:,k*3+1].cpu().float().numpy()
                f2 = hiddens2[L][:,k*3+1].cpu().float().numpy()
                f3 = hiddens3[L][:,k*3+1].cpu().float().numpy()
                
                features.append(f1)
                features.append(f2)
                features.append(f3)
                
            features = np.concatenate(features, axis=0)
            
            print('start', col)
            umap_model = UMAP(min_dist=0, n_components=2, random_state=42)
            umap_model.fit(features)
            print('end', col)
            
            # Save the trained model to a file
            joblib.dump(umap_model, folder+'UMAPModel/L='+str(L)+'.pkl')
        
                
        plt.rcParams.update({'font.size': 20})
        fig, axes = plt.subplots(6, 6, figsize=(24, 16), sharex='col', sharey='col') 
        
        Task1 = {}
        Task2 = {}
        Task3 = {}
        for row, k in tqdm(enumerate(np.arange(0, args.num_k, 3))):
            Task1[k] = {}
            Task2[k] = {}
            Task3[k] = {}
            for col, L in enumerate(L_list):
                ax = axes[row,col]
                
                f1 = hiddens1[L][:,k*3+1].cpu().float().numpy()
                f2 = hiddens2[L][:,k*3+1].cpu().float().numpy()
                f3 = hiddens3[L][:,k*3+1].cpu().float().numpy()
                
                Task1[k][L] = np.mean(f1, axis=0)
                Task2[k][L] = np.mean(f2, axis=0)
                Task3[k][L] = np.mean(f3, axis=0)
                
                p1 = prediction1[:,k]
                p2 = prediction2[:,k]
                p3 = prediction3[:,k]
                features = np.concatenate((f1, f2, f3), axis=0)
                
                '''
                # Initialize and fit t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                features_2d = tsne.fit_transform(features)
                
                # Split the 2D features back into two parts for visualization
                f1_2d = features_2d[:len(f1)]
                f2_2d = features_2d[len(f1):]
                '''
                
                #umap_model = UMAP(min_dist=0, n_components=2, random_state=42)
                #umap_model.fit(features)
                
                # Save the trained model to a file
                umap_model = joblib.load(folder+'UMAPModel/L='+str(L)+'.pkl')
                
                features_2d = umap_model.transform(features)
                f1_2d = features_2d[:len(f1)]
                f2_2d = features_2d[len(f1):-len(f3)]
                f3_2d = features_2d[-len(f3):]
                
                task1 = umap_model.transform(Task1[k][L].reshape([1,-1]))
                task2 = umap_model.transform(Task2[k][L].reshape([1,-1]))
                task3 = umap_model.transform(Task3[k][L].reshape([1,-1]))
                
                # Plotting
                #for i in range(len(f1)):
                num_visual = 100
                ax.scatter(f1_2d[:num_visual, 0], f1_2d[:num_visual, 1], s=p1[:num_visual]/8+0.5, label='f1', marker='.', c='red')
                ax.scatter(f2_2d[:num_visual, 0], f2_2d[:num_visual, 1], s=p2[:num_visual]/8+0.5, label='f2', marker='.', c='blue')
                ax.scatter(f3_2d[:num_visual, 0], f3_2d[:num_visual, 1], s=p3[:num_visual]/8+0.5, label='f3', marker='.', c='green')
                
                ax.scatter(task1[:, 0], task1[:, 1], s=600, label='f1 center', facecolors='None',  edgecolors='red')
                ax.scatter(task2[:, 0], task2[:, 1], s=600, label='f2 center', facecolors='None',  edgecolors='blue')
                ax.scatter(task3[:, 0], task3[:, 1], s=600, label='f3 center', facecolors='None',  edgecolors='green')
                
                #plt.legend()
                ax.set_title('k='+str(k)+' L='+str(L))
                #plt.xlabel('Component 1')
                #plt.ylabel('Component 2')
        plt.tight_layout()
        plt.savefig('compare.png')
    
    filename = folder+'task1_mean.pickle'
    print(filename)
    # Open a file in binary write mode
    with open(filename, 'wb') as file:
        # Use pickle.dump() to serialize and save the dictionary
        pickle.dump(Task1, file)
    filename = folder+'task2_mean.pickle'
    print(filename)
    # Open a file in binary write mode
    with open(filename, 'wb') as file:
        # Use pickle.dump() to serialize and save the dictionary
        pickle.dump(Task2, file)
    filename = folder+'task3_mean.pickle'
    print(filename)
    # Open a file in binary write mode
    with open(filename, 'wb') as file:
        # Use pickle.dump() to serialize and save the dictionary
        pickle.dump(Task3, file)
        
        
plant_k = 3
plant_task = 1
if 1: #explore boost visualizer
    epoch = 10
    state_folder = folder+'EP=' +str(epoch)+'/checkpoint_'+str(epoch)+'.pth'
    state = torch.load(state_folder)
    model.load_state_dict(state['model_state_dict'])
    #optimizer.load_state_dict(state['optimizer_state_dict']) 
    model.eval()
    
    filename = folder+'task'+str(plant_task)+'_mean.pickle'
    with open(filename, 'rb') as file:
        # Use pickle.load() to deserialize and load the dictionary
        Task = pickle.load(file)
    
    with torch.no_grad():
        num_visual = 100
        bs = 1024
        new_task_m = priorProcesser.topic_ms[0]*0


        # with full examples
        new_task_w = priorProcesser.topic_ws[1]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=num_visual, k=15, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward4(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction1last = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction1last-bs_learning)**2, axis=0)[list(np.arange(0,plant_k+1,3))])
        hiddens1last = results['hiddens']
        
        
        new_task_w = priorProcesser.topic_ws[2]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=num_visual, k=15, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction2last = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction2last-bs_learning)**2, axis=0)[list(np.arange(0,plant_k+1,3))])
        hiddens2last = results['hiddens']
        
        
        new_task_w = priorProcesser.topic_ws[3]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=num_visual, k=15, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction3last = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction3last-bs_learning)**2, axis=0)[list(np.arange(0,plant_k+1,3))])
        hiddens3last = results['hiddens']
        
        
        
        # with limited examples
        new_task_w = priorProcesser.topic_ws[1]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=bs, k=plant_k+1, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward4(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction1 = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction1-bs_learning)**2, axis=0)[list(np.arange(0,plant_k+1,3))])
        hiddens1 = results['hiddens']
        
        
        new_task_w = priorProcesser.topic_ws[2]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=bs, k=plant_k+1, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction2 = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction2-bs_learning)**2, axis=0)[list(np.arange(0,plant_k+1,3))])
        hiddens2 = results['hiddens']
        
        
        new_task_w = priorProcesser.topic_ws[3]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=bs, k=plant_k+1, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction3 = results['prediction'].cpu().float().numpy()
        print(np.mean((prediction3-bs_learning)**2, axis=0)[list(np.arange(0,plant_k+1,3))])
        hiddens3 = results['hiddens']
        
        
        new_task_w = priorProcesser.topic_ws[1]
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=bs, k=plant_k+1, new_task_m=new_task_m, new_task_w=new_task_w)
        bs_task2 = bs_xs@priorProcesser.topic_ws[2]
        bs_task3 = bs_xs@priorProcesser.topic_ws[3]
        
        plt.rcParams.update({'font.size': 100})
        fig, axes = plt.subplots(7, 6, figsize=(100, 80), sharex='col', sharey='col') 
        row_names = ['No plant']+['L='+str(L)+' plant' for L in L_list]
        for row, L_plant in enumerate([-1]+L_list):
            if L_plant == -1:
                results = model.forward4(torch.from_numpy(bs_xs).cuda().float(),
                                         torch.from_numpy(bs_ys).cuda().float(),
                                         L = None,
                                         injection=None)
                predictionP = results['prediction'].cpu().float().numpy()
                print("{:<17}".format('no plant loss: '), "{:.2f}".format(np.mean((predictionP-bs_learning)**2, axis=0)[-1])\
                                                            +' '+"{:.2f}".format(np.mean((predictionP-bs_task2)**2, axis=0)[-1])\
                                                            +' '+"{:.2f}".format(np.mean((predictionP-bs_task3)**2, axis=0)[-1]))
                hiddensP = results['hiddens']
            else:
                results = model.forward4(torch.from_numpy(bs_xs).cuda().float(),
                                         torch.from_numpy(bs_ys).cuda().float(),
                                         L = L_plant,
                                         injection=torch.from_numpy(Task[15][L_plant]).cuda().float())
                predictionP = results['prediction'].cpu().float().numpy()
                print("{:<17}".format('L='+str(L_plant)+' plant loss: '), "{:.2f}".format(np.mean((predictionP-bs_learning)**2, axis=0)[-1])\
                                                                     +' '+"{:.2f}".format(np.mean((predictionP-bs_task2)**2, axis=0)[-1])\
                                                                     +' '+"{:.2f}".format(np.mean((predictionP-bs_task3)**2, axis=0)[-1]))
                hiddensP = results['hiddens']
                #print(np.sum(np.abs(hiddensP[-1][:,-2,:].cpu().float().numpy()-Task[15][L_plant])))
                #print(prediction2)
            
        
            for col, L in enumerate(L_list):
                ax = axes[row,col]
                
                f1 = hiddens1[L][:num_visual,-2].cpu().float().numpy()
                f2 = hiddens2[L][:num_visual,-2].cpu().float().numpy()
                f3 = hiddens3[L][:num_visual,-2].cpu().float().numpy()
                fP = hiddensP[L][:num_visual,-2].cpu().float().numpy()
                f1last = hiddens1last[L][:,-2].cpu().float().numpy()
                f2last = hiddens2last[L][:,-2].cpu().float().numpy()
                f3last = hiddens3last[L][:,-2].cpu().float().numpy()
                #Task[k][L] = np.mean(f1, axis=0)
                
                p1 = prediction1[:num_visual,-1]
                p2 = prediction2[:num_visual,-1]
                p3 = prediction3[:num_visual,-1]
                pP = predictionP[:num_visual,-1]
                p1last = prediction1last[:,-1]
                p2last = prediction2last[:,-1]
                p3last = prediction3last[:,-1]
                
                # Initialize and fit t-SNE
                #tsne = TSNE(n_components=2, random_state=42)
                #features_2d = tsne.fit_transform(features)
                umap_model = joblib.load(folder+'UMAPModel/L='+str(L)+'.pkl')
                f = f1
                p = p1
                c = 'red'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f1', marker='X', c=c)
                f = f2
                p = p2
                c = 'blue'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f2', marker='X', c=c)
                f = f3
                p = p3
                c = 'green'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f3', marker='X', c=c)
                f = fP
                p = pP
                c = 'orange'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f3', marker='X', c=c)
                f = f1last
                p = p1last
                c = 'red'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f1', marker='X', edgecolor=c, facecolors='none')
                f = f2last
                p = p2last
                c = 'blue'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f2', marker='X', edgecolor=c, facecolors='none')
                f = f3last
                p = p3last
                c = 'green'
                f_2d = umap_model.transform(f)
                ax.scatter(f_2d[:, 0], f_2d[:, 1], s=5*(p+8), label='f3', marker='X', edgecolor=c, facecolors='none')
                
                ax.set_title('L='+str(L))
                #plt.xlabel('Component 1')
                #plt.ylabel('Component 2')
                
                if col == 0:
                    ax.set_ylabel(row_names[row], labelpad=15, weight='bold')
            
        plt.tight_layout()
        plt.savefig('15to'+str(plant_k)+' plant '+str(plant_task)+'->2.png')
        


plant_k = 3
plant_task = 2
if 0: #explore conflict visualizer
    epoch = 10
    state_folder = folder+'EP=' +str(epoch)+'/checkpoint_'+str(epoch)+'.pth'
    state = torch.load(state_folder)
    model.load_state_dict(state['model_state_dict'])
    #optimizer.load_state_dict(state['optimizer_state_dict']) 
    model.eval()
    
    filename = folder+'task'+str(plant_task)+'_mean.pickle'
    with open(filename, 'rb') as file:
        # Use pickle.load() to deserialize and load the dictionary
        Task = pickle.load(file)
    
    with torch.no_grad():
        new_task_m = priorProcesser.topic_ms[0]*0
        
        new_task_w = np.array([0,0,-1])
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=1024, k=plant_k+1, new_task_m=new_task_m, new_task_w=new_task_w)
        results = model.forward3(torch.from_numpy(bs_xs).cuda().float(),
                                 torch.from_numpy(bs_ys).cuda().float())
        prediction1 = results['prediction'].cpu().float().numpy()
        print('loss: ', np.mean((prediction1-bs_learning)**2, axis=0)[-1])
        hiddens1 = results['hiddens']
        
        
        new_task_w = np.array([0,0,+1])
        bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(bs=1024, k=plant_k+1, new_task_m=new_task_m, new_task_w=new_task_w)
        
        bs_opposite = bs_xs@np.array([0,0,-1])
        #print(np.mean((prediction2-bs_opposite)**2, axis=0)[[0,3,6,9,12,15]])
        
        
        plt.rcParams.update({'font.size': 20})
        fig, axes = plt.subplots(6, 5, figsize=(20, 16), sharex='col', sharey='col') 
        row_names = ['No plant']+['L='+str(L)+' plant' for L in L_list]
        for row, L_plant in enumerate([-1]+L_list):
            if L_plant == -1:
                results = model.forward4(torch.from_numpy(bs_xs).cuda().float(),
                                         torch.from_numpy(bs_ys).cuda().float(),
                                         L = None,
                                         injection=None)
                prediction2 = results['prediction'].cpu().float().numpy()
                print('no plant loss: ', np.mean((prediction2-bs_learning)**2, axis=0)[-1])
                print('no plant loss: ', np.mean((prediction2-bs_opposite)**2, axis=0)[-1])
                hiddens2 = results['hiddens']
            else:
                results = model.forward4(torch.from_numpy(bs_xs).cuda().float(),
                                         torch.from_numpy(bs_ys).cuda().float(),
                                         L = L_plant,
                                         injection=torch.from_numpy(Task[15][L_plant]).cuda().float())
                prediction2 = results['prediction'].cpu().float().numpy()
                print('L='+str(L_plant)+' plant loss: ', np.mean((prediction2-bs_learning)**2, axis=0)[-1])
                print('L='+str(L_plant)+' plant loss: ', np.mean((prediction2-bs_opposite)**2, axis=0)[-1])
                hiddens2 = results['hiddens']
                #print(prediction2)
            
        
            for col, L in enumerate(L_list):
                ax = axes[row,col]
                
                f1 = hiddens1[L][:,-2].cpu().float().numpy()
                f2 = hiddens2[L][:,-2].cpu().float().numpy()
                
                #Task[k][L] = np.mean(f1, axis=0)
                
                p1 = prediction1[:,-1]
                p2 = prediction2[:,-1]
                features = np.concatenate((f1, f2), axis=0)
                
                
                # Initialize and fit t-SNE
                #tsne = TSNE(n_components=2, random_state=42)
                #features_2d = tsne.fit_transform(features)
                
                umap_model = joblib.load(folder+'UMAPModel/L='+str(L)+'.pkl')
                features_2d = umap_model.transform(features)
                
                # Split the 2D features back into two parts for visualization
                f1_2d = features_2d[:len(f1)]
                f2_2d = features_2d[len(f1):]

                # Plotting
                #for i in range(len(f1)):
                ax.scatter(f1_2d[:, 0], f1_2d[:, 1], s=p1/8+0.5, label='f1', marker='.', c='red')
                ax.scatter(f2_2d[:, 0], f2_2d[:, 1], s=p2/8+0.5, label='f2', marker='.', c='blue')
                #plt.legend()
                ax.set_title('L='+str(L))
                #plt.xlabel('Component 1')
                #plt.ylabel('Component 2')
                
                if col == 0:
                    ax.set_ylabel(row_names[row], labelpad=15, weight='bold')
            
        plt.tight_layout()
        plt.savefig('15to'+str(plant_k)+' plant '+str(plant_task)+'->2.png')