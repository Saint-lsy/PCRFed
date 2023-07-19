import numpy as np
import torch
from torch.autograd import Variable


def set_global_grad(net, keys, tag):
    for name, param in net.named_parameters():
        if name[:2] == 'hc':
            continue
        if name in keys:
            param.requires_grad = (tag == 1)
        else:
            param.requires_grad = (tag == 0)


def check_equal(net_clients):
    client_num = len(net_clients)
    for param in zip(net_clients[0].parameters(), net_clients[1].parameters(),
                     net_clients[2].parameters(), net_clients[3].parameters()):
        for i in range(1, client_num):
            assert torch.max(param[i].data - param[i - 1].data) == 0


def update_global_model(net_clients, client_weight):
    client_num = len(net_clients)
    if len(net_clients) == 4:
        iter_container = zip(net_clients[0].parameters(),
                             net_clients[1].parameters(),
                             net_clients[2].parameters(),
                             net_clients[3].parameters())
    elif len(net_clients) == 6:
        iter_container = zip(net_clients[0].parameters(),
                             net_clients[1].parameters(),
                             net_clients[2].parameters(),
                             net_clients[3].parameters(),
                             net_clients[4].parameters(),
                             net_clients[5].parameters())
    elif len(net_clients) == 8:
        iter_container = zip(net_clients[0].parameters(),
                             net_clients[1].parameters(),
                             net_clients[2].parameters(),
                             net_clients[3].parameters(),
                             net_clients[4].parameters(),
                             net_clients[5].parameters(),
                             net_clients[6].parameters(),
                             net_clients[7].parameters())

    for param in iter_container:
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)),
                            requires_grad=False).cuda()
        for i in range(client_num):
            new_para.data.add_(client_weight[i], param[i].data)

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)


def update_global_model_with_keys(net_clients, client_weight, keys):
    client_num = len(net_clients)
    if len(net_clients) == 4:
        iter_container = zip(net_clients[0].named_parameters(),
                             net_clients[1].named_parameters(),
                             net_clients[2].named_parameters(),
                             net_clients[3].named_parameters())
    elif len(net_clients) == 6:
        iter_container = zip(net_clients[0].named_parameters(),
                             net_clients[1].named_parameters(),
                             net_clients[2].named_parameters(),
                             net_clients[3].named_parameters(),
                             net_clients[4].named_parameters(),
                             net_clients[5].named_parameters())
    elif len(net_clients) == 8:
        iter_container = zip(net_clients[0].named_parameters(),
                             net_clients[1].named_parameters(),
                             net_clients[2].named_parameters(),
                             net_clients[3].named_parameters(),
                             net_clients[4].named_parameters(),
                             net_clients[5].named_parameters(),
                             net_clients[6].named_parameters(),
                             net_clients[7].named_parameters(),)

    for data in iter_container:
        name = [d[0] for d in data]
        param = [d[1] for d in data]
        if not name[0] in keys:
            continue
        new_para = Variable(torch.Tensor(np.zeros(param[0].shape)),
                            requires_grad=False).cuda()
        for i in range(client_num):
            new_para.data.add_(param[i].data, alpha = client_weight[i])

        for i in range(client_num):
            param[i].data.mul_(0).add_(new_para.data)

