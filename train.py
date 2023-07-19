import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from tensorboardX import SummaryWriter

import argparse

import time
import random
import numpy as np
from glob import glob
import torch
import torch.nn as nn
from torchvision import transforms
# network codes
from networks.pmc_net import PMCNet
from torch.utils.data import DataLoader
import copy
from torchvision.transforms import ToTensor

from scripts.trainer_utils import set_global_grad, update_global_model_with_keys, update_global_model

from utils.losses import supervised_loss, dice_coef
from utils.summary import create_logger
from dataloaders.cervical_dataloader import SegmentationTransform, CenterCrop  

parser = argparse.ArgumentParser()

parser.add_argument('--dataset',
                    type=str,
                    default='prostate',
                    help='dataset name')

parser.add_argument('--model', 
                    type=str, 
                    default='PMCNet', 
                    help='neural network used in training')

parser.add_argument('--alg', 
                    type=str, 
                    default='PCRFed',
                    help='communication strategy: local/fedavg/fedprox/moon')

parser.add_argument('--comm_round', 
                    type=int, 
                    default=200, 
                    help='number of maximum communication roun')

parser.add_argument('--local_epoch', 
                    type=int, 
                    default=5, 
                    help='number of local epochs')

parser.add_argument('--head_iter',
                    type=int,
                    default=1,
                    help='iter number of head in local update')

parser.add_argument('--batch_size',
                    type=int,
                    default=8,
                    help='batch_size per gpu')
parser.add_argument('--base_lr',
                    type=float,
                    default=0.0001,
                    help='basic learning rate of each site')
parser.add_argument('--load_weight',
                    type=int,
                    default=0,
                    help='load pre-trained weight from local site')

parser.add_argument('--personalizing', 
                    type=int, 
                    default=1,
                    help='personalizing or not')

parser.add_argument('--norm', type=str, default='in', help='normalization')

parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

parser.add_argument('--mu', type=float, default=1.0, help='the mu parameter for dice loss and bce loss')

parser.add_argument('--alpha', type=float, default=1.0, help='the alpha parameter for sup loss and contrastive loss')

parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')

parser.add_argument('--weighted',
                    type=int,
                    default=0,
                    help='weighted or not in moon')

parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')

args = parser.parse_args()

####Adijust parameters and add appendix 
appendix = ''



if args.dataset == 'prostate':
    from dataloaders.prostate_dataset import Prostate
    args.client_num = 6
    args.num_classes = 1
    sites = ['RUNMC', 'BMC', 'I2CVB', 'UCL','BIDMC', 'HK']
    args.c_in = 1
elif args.dataset == 'cervical':
    from dataloaders.cervical_dataloader import Cervical
    ###Adjust the number of clients according to the actual situation
    args.client_num = 8
    args.num_classes = 1
    args.c_in = 1
else:
    raise NotImplementedError


filename = 'moon_bs{}_personal{}_com{}_local{}_{}_mu{}_alpha{}_temp{}'.format(
        args.batch_size, args.personalizing, args.comm_round, args.local_epoch, args.norm, args.mu, args.alpha, args.temperature)

txt_path = 'logs/{}/{}/txt/'.format(args.dataset, filename)
log_path = 'logs/{}/{}/log/'.format(args.dataset, filename)
model_path = 'logs/{}/{}/model/'.format(args.dataset, filename)

os.makedirs(txt_path, exist_ok=True)
os.makedirs(model_path, exist_ok=True)
logger = create_logger(0, save_dir=txt_path)
print = logger.info
print(args)


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))

base_lr = args.base_lr
mu = args.mu

assert args.num_classes > 0 and args.client_num > 1
print(args)

# ------------------  start training ------------------ #
# weight average
# client_weight = np.ones((args.client_num, )) / args.client_num
# print(client_weight)

def client_test(net_current, dataloader_current):
    
    net_current.eval()
    loss_all = 0
    test_acc = 0.   

    for i_batch, sampled_batch in enumerate(dataloader_current):

        # obtain training data
        volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
        pred = net_current(volume_batch)  ####infer
        
        #### PCRNet output is a tuple
        if isinstance(pred, tuple):
            pred = pred[1]

        loss_batch = supervised_loss(pred, label_batch, mu)

        loss_all += loss_batch.item()
        test_acc += dice_coef(pred.squeeze(axis=1), label_batch.squeeze(axis=1)).item()
            
    loss = loss_all / len(dataloader_current)
    acc = test_acc/ len(dataloader_current)
   
    return loss, acc


if __name__ == "__main__":

    ## set seed
    seed = args.seed
    logger.info("#" * 100)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)



    global_net = PMCNet(args.c_in, 16, args.norm, args.num_classes, embed_way=4)


    net_clients = []
    optimizer_clients = []
    numdataset_clients = []


    trainloader_clients = []
    valloader_clients = []
    testloader_clients = []
    ###init local nets
    for client_idx in range(args.client_num):


        net = PMCNet(args.c_in, 16, args.norm, args.num_classes, embed_way=4) 
        net = net.cuda()

        if args.dataset == 'prostate':
            train_transform = None
            val_transform = None
            test_transform = None
            trainset = Prostate(site=sites[client_idx], channel=args.c_in, split='train', data_path='add_your_data_path', transform=train_transform)
            valset = Prostate(site=sites[client_idx], channel=args.c_in, split='val', data_path='add_your_data_path', transform=val_transform)
            testset = Prostate(site=sites[client_idx], channel=args.c_in, split='test', data_path='add_your_data_path', transform=test_transform)
            print(f'[Client {sites[client_idx]}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')

        elif args.dataset == 'cervical':
            train_transform = SegmentationTransform(
                            # flip_prob=0,
                            # angle_range=(-5, 5),
                            # scale_range=(0.95, 1.05),
                            # crop_range=(256, 256),
                            output_size=(384, 384),
                            to_tensor=True,
                        )
            val_transform = SegmentationTransform(
                            # crop_range=(256, 256),
                            output_size=(384, 384),
                            to_tensor=True,
                        )
            test_transform = SegmentationTransform(
                            # crop_range=(256, 256),
                            output_size=(384, 384),
                            to_tensor=True,
                        )

            trainset = Cervical(site_index=client_idx, split='train', transform=train_transform)
            valset = Cervical(site_index=client_idx, split='val', transform=val_transform)
            testset = Cervical(site_index=client_idx, split='test', transform=test_transform)
            print(f'[Client {client_idx}] Train={len(trainset)}, Val={len(valset)}, Test={len(testset)}')
            train_transform = transforms.Compose([])

        trainloader = DataLoader(trainset, 
                                 batch_size=batch_size, 
                                 shuffle=True, num_workers=1, 
                                 pin_memory=True, 
                                 worker_init_fn=worker_init_fn, 
                                 drop_last=True)

        valloader = DataLoader(valset, 
                               batch_size=batch_size, 
                               shuffle=False, 
                               num_workers=1, 
                               pin_memory=True, 
                               worker_init_fn=worker_init_fn)
        
        testloader = DataLoader(testset, 
                                batch_size=batch_size, 
                                shuffle=False, 
                                num_workers=1, 
                                pin_memory=True, 
                                worker_init_fn=worker_init_fn)

        trainloader_clients.append(trainloader)
        valloader_clients.append(valloader)
        testloader_clients.append(testloader)
        numdataset_clients.append(len(trainset))

        # segmentation model
        optimizer = torch.optim.Adam(net.parameters(),
                                        lr=args.base_lr,
                                        betas=(0.9, 0.999))
        optimizer_clients.append(optimizer)
        net_clients.append(net)

    if args.personalizing:
        to_ignore = ['seg', 'hc', 'bn']
        global_keys = []
        ignored_keys = []
        for k in net.state_dict().keys():
            ignore_tag = 0
            for ignore_key in to_ignore:
                if ignore_key in k:
                    ignore_tag = 1
            if not ignore_tag:
                global_keys.append(k)
            else:
                ignored_keys.append(k)
        print(global_keys)
        print(ignored_keys)
    else:
        global_keys = net.state_dict().keys()
        ignored_keys = []
        print(global_keys)

    print('[INFO] Initialized success...')

    # start federated learning
    best_score = 0
    writer = SummaryWriter(log_path)
    lr_ = base_lr

    # weight average

    total_data_num = sum(numdataset_clients)
    client_weight = np.array(numdataset_clients) / total_data_num

    c_loss_func = nn.MSELoss()
    n_comm_rounds = args.comm_round
    local_epoch = args.local_epoch

    temperature = args.temperature
    alpha = args.alpha
    ###init moon
    # old_nets = copy.deepcopy(net_clients)
    # for _, net in old_nets.items():
    #     net.eval()
    #     for param in net.parameters():
    #         param.requires_grad = False
    global_net = global_net.cuda()
    global_net.eval()
    for param in global_net.parameters():
        param.requires_grad = False
    global_w = global_net.state_dict()

    old_nets = copy.deepcopy(net_clients)
    criterion = nn.CrossEntropyLoss().cuda()
    cos=torch.nn.CosineSimilarity(dim=-1)

    for round in range(n_comm_rounds):
        print("in comm round:" + str(round))
        
        if args.personalizing:
            seg_heads = copy.deepcopy([_net.seg1 for _net in net_clients])    

        for client_idx in range(args.client_num):  ##每个party训练
            dataloader_current = trainloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()         
            ###init global net
            if args.personalizing:
                for key in ignored_keys:
                    global_w[key] = net_current.state_dict()[key]
            global_net.load_state_dict(global_w)
            global_net.eval()
            for param in global_net.parameters():
                param.requires_grad = False
            
            ###init client
            prev_net = old_nets[client_idx]


            # for previous_net in prev_models:
            #     previous_net.cuda()

            for epoch in range(local_epoch):

                for i_batch, sampled_batch in enumerate(dataloader_current):

                    net_current.train()
                    # obtain training data
                    volume_batch, label_batch = sampled_batch['image'].cuda(
                    ), sampled_batch['label'].cuda()

                    optimizer_current.zero_grad()

                    if args.personalizing:
                        # train seg 冻结梯度操作 一部分batch 训练seg head 一部分训练全局
                        if i_batch < args.head_iter: 
                            set_global_grad(net_current, global_keys, False)
                        else:
                            set_global_grad(net_current, global_keys, True)

                    # pred = net_current(volume_batch)  ####infer
                    
                    pro1, pred = net_current(volume_batch)
                    pro2, _ = global_net(volume_batch)

                    posi = cos(pro1, pro2)
                    logits = posi.reshape(-1,1)
                    ####Contrastive Loss
                    with torch.no_grad():
                        prev_net.cuda()
                        pro3, _ = prev_net(volume_batch)
                        nega = cos(pro1, pro3)
                        logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)
                    # prev_net.to('cpu')
                    
                    logits /= temperature
                    labels = torch.zeros(volume_batch.size(0)).cuda().long()
                    #### 利用交叉熵公式实现对比损失
                    if not args.weighted:
                        con_loss = alpha * criterion(logits, labels)
                    else:
                        ####weight1
                        con_loss = alpha * torch.exp(torch.tensor(-client_weight[client_idx])) * criterion(logits, labels)
                        ####weight2
                        # con_loss = alpha * np.log2(1 + 1/(args.client_num * client_weight[client_idx])) * criterion(logits, labels)



                    sup_loss = supervised_loss(pred, label_batch, mu)
                    train_acc = dice_coef(pred.squeeze(axis=1), label_batch.squeeze(axis=1)).item()


                    loss = sup_loss + con_loss
                    loss.backward()
                    optimizer_current.step()

                    iter_num = len(dataloader_current) * local_epoch * round +  len(dataloader_current)* epoch + i_batch

                    if iter_num % 10 == 0:
                        writer.add_scalar('sup_loss/site{}'.format(client_idx + 1), sup_loss, iter_num)
                        writer.add_scalar('con_loss/site{}'.format(client_idx + 1), con_loss, iter_num)

                        writer.add_scalar('train_loss/site{}'.format(client_idx + 1), loss, iter_num)
                        writer.add_scalar('train_Dice/site{}'.format(client_idx + 1),
                                            train_acc, iter_num)

                        print(
                            '[Train] Round: [%d], Epoch: [%d] client [%d] iteration [%d / %d] : Sup_loss : %f Con_loss: %f Dice: %f'
                            % (round, epoch, client_idx + 1, i_batch,
                            len(dataloader_current), sup_loss.item(), con_loss.item(), train_acc))
        

        with torch.no_grad(): 
            ###更新party模型
            for net_id in range(len(net_clients)): 
                net_para = net_clients[net_id].state_dict()
                if net_id == 0:
                    for key in global_keys:
                        global_w[key] = net_para[key] * client_weight[net_id]
                else:
                    for key in global_keys:
                        global_w[key] += net_para[key] * client_weight[net_id]                    

            ## model aggregation    net_structure   weight   layer_names
            # update_global_model_with_keys(net_clients, client_weight, global_keys)

            ## evaluation
            overall_score = 0
            val_acc_list = [None for j in range(args.client_num)]
            test_acc_list = [None for j in range(args.client_num)]
            
            for site_index in range(args.client_num):
                this_net = net_clients[site_index]
                
                print("[Val] round {} testing Site {}".format(
                    round, site_index + 1))
                
                val_loss, val_acc = client_test(this_net, valloader_clients[site_index])
                val_acc_list[site_index] = val_acc
                writer.add_scalar('val_acc/site{}'.format(site_index + 1),
                                            val_acc, round)
                writer.add_scalar('val_loss/site{}'.format(site_index + 1),
                                            val_loss, round)
                print(
                    '[Val] Round: [%d], client [%d] val_acc: %f val_loss : %f '
                    % (round, site_index, val_acc, val_loss))

            # Test after each round
            for site_index in range(args.client_num):
                this_net = net_clients[site_index]

                print("[Test] round {} testing Site {}".format(
                    round, site_index + 1))
                
                _, test_acc = client_test(this_net, testloader_clients[site_index])
                test_acc_list[site_index] = test_acc
                writer.add_scalar('test_acc/site{}'.format(site_index + 1),
                                test_acc, round)
                print(
                    '[Test] Round: [%d], client [%d] test_acc: %f'
                    % (round, site_index + 1, test_acc))

        # overall_score = sum(val_acc_list) / args.client_num
        overall_score = sum([val_acc_list[i] * client_weight[i] for i in range(args.client_num)])

        writer.add_scalar('Score_Overall', overall_score, round)

        if overall_score > best_score:
            best_score = overall_score
            ## save model
            save_mode_path = os.path.join(model_path, 'best.pth')
            torch.save(net_clients[0].state_dict(), save_mode_path)

            for site_index in range(args.client_num):
                save_mode_path = os.path.join(
                    model_path, 'Site{}_best.pth'.format(site_index + 1))
                torch.save(net_clients[site_index].state_dict(),
                        save_mode_path)
        print('[INFO] IoU Overall: {:.2f} Best IoU {:.2f}'.format(
            overall_score * 100, best_score * 100))
        
        ####变量保存局部模型 用于下一轮训练
        for site_index in range(args.client_num):
            old_nets = copy.deepcopy(old_nets)
            for net in old_nets:                    
                net.eval()
                for param in net.parameters():
                    param.requires_grad = False                
