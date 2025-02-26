import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_CNN, ResNet18
from clients import ClientsGroup
import Attack


def cos(a,b):
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    if res < 0:
        res = 0
    return res

def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr

def cosScoreAndClipValue(net1, net2):
    vector1 = model2vector(net1)
    vector2 = model2vector(net2)
    return cos(vector1, vector2), norm_clip(vector1, vector2)


def norm_clip(nparr1, nparr2):
    vnum = np.linalg.norm(nparr1, ord=None, axis=None, keepdims=False) + 1e-9
    return vnum / np.linalg.norm(nparr2, ord=None, axis=None, keepdims=False) + 1e-9


def get_weight(update, model):
    for key, var in update.items():
        update[key] = update[key] - model[key]
    return update


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FLTrust")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=50, help='numer of the clients')
parser.add_argument('-by', '--byzantine_size', type=int, default=10, help='number of the byzantine clients')
parser.add_argument('-p','--pattern', type=int, default=5,help='patterns of attack methods (0,1,2,3,4,5,6)')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
parser.add_argument('-data', '--data_name', type=str, default='mnist', help='the data to train') # cifar_10, mnist
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-ncomm', '--num_comm', type=int, default=20, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=bool, default=False, help='the way to allocate data to clients')
parser.add_argument('-cen', '--central_data_size', type=int, default=300, help='central data size in server')
parser.add_argument('-pro', '--central_data_pro', type=float, default=0.1, help='central data pro in server')
args = parser.parse_args()
args = args.__dict__


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    acc_list=[]
    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['data_name'] == 'cifar_10':
        net = ResNet18()
    elif args['data_name'] == 'mnist':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")

        
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup(args['data_name'], args['IID'], args['num_of_clients'],dev)
    myClients.get_central_data(args['central_data_size'], args['central_data_pro'])
    
    testDataLoader = myClients.test_data_loader

    honest_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'])]
    byzantine_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'],args['num_of_clients'])]
    
    
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone().float()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        FLTrustCentralNorm = myClients.centralTrain(args['epoch'], args['batchsize'], net,                                        
                                                                         loss_func, opti, global_parameters)
        
        '''get the update weight'''
        FLTrustCentralNorm = get_weight(FLTrustCentralNorm, global_parameters)
        
        sum_parameters = None
        FLTrustTotalScore = 0
        honest_all_weight = None
        

        for client in honest_clients:
            local_parameters = myClients.clients_set[client].localTrain(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            local_parameters = get_weight(local_parameters, global_parameters)            
            client_score, client_clipped_value = cosScoreAndClipValue(FLTrustCentralNorm, local_parameters)
            FLTrustTotalScore += client_score
            
            if sum_parameters is None:
                sum_parameters = {}
                honest_all_weight = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = client_score * client_clipped_value * var.clone()
                    honest_all_weight[key] = var.clone()
                    honest_all_weight[key] = honest_all_weight[key].unsqueeze(0)
            else:
                for key in local_parameters:
                    sum_parameters[key] = sum_parameters[key] + client_score * client_clipped_value * local_parameters[key]
                    honest_all_weight[key] = torch.cat([honest_all_weight[key],local_parameters[key].unsqueeze(0)],dim = 0)


        for client in byzantine_clients:
            

            local_parameters = {}
            if args['pattern'] == 2:
                for key in honest_all_weight:
                    local_parameters[key] = Attack.ZeroGradient_attack(honest_all_weight[key],args['byzantine_size'])
            
            elif args['pattern'] == 0:
                for key in honest_all_weight:
                    local_parameters[key] = Attack.Gaussian_attack(honest_all_weight[key]) 
            
            elif args['pattern'] == 1:
                for key in honest_all_weight:
                    local_parameters[key] = Attack.Sign_flipping_attack(honest_all_weight[key]) 
                                         
            
            elif args['pattern'] == 3:
                Poisoning_client = Attack.backdoor_poisoning_data(myClients.clients_set[client], args['data_name'])
                local_parameters = Poisoning_client.localTrain(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            elif args['pattern'] == 4:
                Poisoning_client = Attack.model_replacement_attack_data(myClients.clients_set[client], args['data_name'])
                local_parameters = Poisoning_client.localTrain(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
                
                # Gradient expansion
                for key in local_parameters:
                    local_parameters[key] = local_parameters[key]*args['num_of_clients']

            elif args['pattern'] == 5:
                for key in honest_all_weight:
                    local_parameters[key] = Attack.MPAF(honest_all_weight[key])
            
            elif args['pattern'] == 6:
                for key in honest_all_weight:
                    local_parameters[key] = Attack.AGR_agnostic(honest_all_weight[key])
            
            
            client_score, client_clipped_value = cosScoreAndClipValue(FLTrustCentralNorm, local_parameters)

            FLTrustTotalScore += client_score
            
            for key in local_parameters:
                sum_parameters[key] = sum_parameters[key] + client_score * client_clipped_value * local_parameters[key]
                
              
        for key in global_parameters:
            global_parameters[key] += (sum_parameters[key] / FLTrustTotalScore + 1e-9)

        with torch.no_grad():
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)
                    preds = torch.argmax(preds, dim=1)
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
                acc_list.append(sum_accu.item() / num)


    torch.save(net, os.path.join(args['save_path'],
    '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cen{}_attack{}'.format(args['data_name'],
                                                           args['num_comm'], 
                                                           args['epoch'],
                                                           args['batchsize'],
                                                           args['learning_rate'],
                                                           args['num_of_clients'],
                                                           args['central_data_size'],
                                                           args['pattern'])))








