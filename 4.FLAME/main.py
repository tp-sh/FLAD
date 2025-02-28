import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_CNN, ResNet18
from clients import ClientsGroup
import Attack
import hdbscan
import time
import sys


def cos(model1,model2):
    a = model2vector(model1)
    b = model2vector(model2)
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    return res


def model2vector(model):
    nparr = np.array([])
    for key, var in model.items():
        nplist = var.cpu().numpy()
        nplist = nplist.ravel()
        nparr = np.append(nparr, nplist)
    return nparr


def get_weight(update, model):
    '''get the update weight'''
    for key, var in update.items():
        update[key] = update[key] - model[key]
    return update

# Returns Cluster_Parameters: initial filtered local parameters
def Clustering(Cos_Parameters, Upload_Parameters):
    try:
        cluster = hdbscan.HDBSCAN(min_cluster_size=3, gen_min_span_tree=True).fit(Cos_Parameters.reshape(-1,1))
    except:
        print("The uploaded local gradients are too similar, leading to clustering anomalies in cosine similarity.")
        sys.exit(-1)
        # raise ValueError("The uploaded local gradients are too similar, leading to clustering anomalies in cosine similarity")
    label_list = list(cluster.labels_)
    # print("label of all cos: {}".format(label_list))

    labelcount = set(cluster.labels_)
    labelcount.discard(-1)
    # print("labelcount: {}".format(labelcount))

    benign_label = max(labelcount, key = label_list.count)
    print("benign label: {}".format(benign_label))

    count = 0
    benign_list = []
    for i in range(args['num_of_clients']):
        for j in range(i, args['num_of_clients']):
            if label_list[count] == benign_label:
                benign_list.append(i)
                benign_list.append(j)
            count = count + 1
    Cluster_Parameters = []
    benign = set(benign_list)
    for i in benign:
        Cluster_Parameters.append(Upload_Parameters[i])    
    print("benign clinets:{}".format(benign))
    return Cluster_Parameters, benign


def FedAvg_noise(Clip_Parameters, sigma, dev):
    sum_parameters = None
    for j in range(len(Clip_Parameters)):
        if sum_parameters is None:
            sum_parameters = {}
            for key in Clip_Parameters[j]:
                sum_parameters[key] = Clip_Parameters[j][key]
        else:
            for key in Clip_Parameters[j]:
                sum_parameters[key] = sum_parameters[key] + Upload_Parameters[j][key]
    for key in sum_parameters:
        mu = torch.zeros(sum_parameters[key].size())
        sum_parameters[key] = sum_parameters[key]/(len(Clip_Parameters)) + torch.normal(mu,sigma**2).to(dev) 
    return sum_parameters
    

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FLAME")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=50, help='numer of the clients')
parser.add_argument('-by', '--byzantine_size', type=int, default=10, help='number of the byzantine clients')
parser.add_argument('-p','--pattern', type=int, default=5,help='patterns of attack methods (0,1,2,3,4,5,6)')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
parser.add_argument('-data', '--data_name', type=str, default='mnist', help='the data to train') # cifar_10, mnist
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from others paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-ncomm', '--num_comm', type=int, default=20, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
parser.add_argument('-l', '--lambda', type=float, default=0.001, help='noise level of global model')
# “We use standard DP parameters and set epsilon = 3705 for IC, lambda = 0.001 for IC and NLP ” --FLAME


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":

    args = parser.parse_args()
    args = args.__dict__

    acc_list=[]
    benign_list = []
    test_mkdir(args['save_path'])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
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


    myClients = ClientsGroup(args['data_name'], args['IID'], args['num_of_clients'], dev)
    
    testDataLoader = myClients.test_data_loader

    honest_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'])]
    byzantine_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'],args['num_of_clients'])]
    
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone().float()

    for i in range(args['num_comm']):
    
        print("\ncommunicate round {}".format(i+1))

        honest_all_weight = None
        Upload_Parameters = []

        for client in honest_clients:
            local_parameters = myClients.clients_set[client].localTrain(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
           
            Upload_Parameters.append(local_parameters)

            if args['pattern'] <= 2 or args['pattern'] >= 5:
                if honest_all_weight is None:
                    honest_all_weight = {}
                    for key, var in local_parameters.items():
                        honest_all_weight[key] = var.clone()
                        honest_all_weight[key] = honest_all_weight[key].unsqueeze(0)
                else:
                    for key in local_parameters:
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
            

            Upload_Parameters.append(local_parameters)
        
        start = time.time()
        # Calculate the cosine similarity: 
        Cos_Parameters = np.array([]) # Store the cosine similarity between local parameters
        for i in range(args['num_of_clients']):
            for j in range(i,args['num_of_clients']):
                Cos_ij = np.array([cos(Upload_Parameters[i],Upload_Parameters[j])])
                Cos_Parameters = np.append(Cos_Parameters,Cos_ij)
        print("Cos_Parameters: {}".format(Cos_Parameters))
                
        Cluster_Parameters, benign = Clustering(Cos_Parameters, Upload_Parameters)
        
        benign_list.append(benign)

        end = time.time()
        time_diff = end - start
        print('time for malicious detection: ' + str(int(time_diff*1000)) + 'ms')

        # Find the adaptive clipping boundary S
        Euclideandis = np.array([])
        for Parameters in Cluster_Parameters:
            vector_update = model2vector(get_weight(Parameters,global_parameters))
            Euclideandis = np.append(Euclideandis, np.linalg.norm(vector_update, ord=None, axis=None, keepdims=False) + 1e-9)
        S = np.median(Euclideandis)
        # print("Euclideandis of Cluster Parameters: {}".format(Euclideandis))
        print("Clipping boundary S: {}".format(S))

        # Local parameter clipping
        count, Clip_Parameters = 0, []
        for Parameters in Cluster_Parameters:
            gamma = S/Euclideandis[count]
            if gamma > 1:
                for key, var in Parameters.items():
                    Parameters[key] = global_parameters[key] + (Parameters[key] - global_parameters[key])
            else:
                for key, var in Parameters.items():
                    Parameters[key] = global_parameters[key] + (Parameters[key] - global_parameters[key])*gamma
            Clip_Parameters.append(Parameters)
            count = count + 1
        
        # FedAvg + noise
        sigma = args['lambda'] * S
        print("Gaussian noise level sigma: {}".format(sigma))
        global_parameters = FedAvg_noise(Clip_Parameters, sigma, dev)

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

    # save model
    torch.save(net, os.path.join(args['save_path'],
    '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_attack{}'.format(args['data_name'],
                                                           args['num_comm'], 
                                                           args['epoch'],
                                                           args['batchsize'],
                                                           args['learning_rate'],
                                                           args['num_of_clients'],
                                                           args['pattern'])))



