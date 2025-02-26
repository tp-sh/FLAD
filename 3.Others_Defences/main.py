import os
import argparse
import torch
import torch.nn.functional as F
from torch import optim
import Attack
from Models import ResNet18, Mnist_CNN
from clients import ClientsGroup


def Median(Upload_Parameters):
    sum_parameters = None
    for j in range(args['num_of_clients']):
        if sum_parameters is None:
            sum_parameters = {}
            for key in Upload_Parameters[j]:
                sum_parameters[key] = Upload_Parameters[j][key].unsqueeze(0)
        else:
            for key in Upload_Parameters[j]:
                sum_parameters[key] = torch.cat([sum_parameters[key],Upload_Parameters[j][key].unsqueeze(0)],dim = 0)
    for key in sum_parameters:
        sum_parameters[key],_ = torch.median(sum_parameters[key],dim=0)
    return sum_parameters


def FedAvg(Upload_Parameters):
    sum_parameters = None
    for j in range(args['num_of_clients']):
        if sum_parameters is None:
            sum_parameters = {}
            for key in Upload_Parameters[j]:
                sum_parameters[key] = Upload_Parameters[j][key]
        else:
            for key in Upload_Parameters[j]:
                sum_parameters[key] = sum_parameters[key] + Upload_Parameters[j][key]
    
    for key in sum_parameters:
        sum_parameters[key] = sum_parameters[key]/(args['num_of_clients'])
    return sum_parameters
        


def Krum(Upload_Parameters):
    sum_parameters = None
    for j in range(args['num_of_clients']):
        if sum_parameters is None:
            sum_parameters = {}
            for key in Upload_Parameters[j]:
                sum_parameters[key] = Upload_Parameters[j][key].unsqueeze(0)
        else:
            for key in Upload_Parameters[j]:
                sum_parameters[key] = torch.cat([sum_parameters[key],Upload_Parameters[j][key].unsqueeze(0)],dim = 0)
    for key in sum_parameters:
        sum_parameters[key] = Krum_one(sum_parameters[key])
    return sum_parameters
    

       
def Krum_one(parameters):
    n = args['num_of_clients']
    dist = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n): # Construct distance tables
        for j in range(i+1, n):
            distance = parameters[i].data - parameters[j].data
            distance = (distance*distance).sum()
            dist[i][j] = distance.data
            dist[j][i] = distance.data    
            
    k = n - args['byzantine_size'] - 2 + 1 
    topv, _ = dist.topk(k=k, dim=1)
    sumdist = - topv.sum(dim=1)
    resindex = sumdist.topk(1)[1].squeeze().data
    return parameters[resindex]
    

def Bulyan(Upload_Parameters):
    sum_parameters = None
    for j in range(args['num_of_clients']):
        if sum_parameters is None:
            sum_parameters = {}
            for key in Upload_Parameters[j]:
                sum_parameters[key] = Upload_Parameters[j][key].unsqueeze(0)
        else:
            for key in Upload_Parameters[j]:
                sum_parameters[key] = torch.cat([sum_parameters[key],Upload_Parameters[j][key].unsqueeze(0)],dim = 0)
    for key in sum_parameters:
        sum_parameters[key] = Bulyan_one(sum_parameters[key])
    return sum_parameters
    
    
def Bulyan_one(parameters):
    n = args['num_of_clients']
    theta,gamma = 24, 2
    dim_size = list(parameters.size())
    dim_size[0] = theta
    result_1 = torch.zeros(dim_size)
    dim_size[0] = gamma
    result_2 = torch.zeros(dim_size)
    dist = torch.zeros(n, n, dtype=torch.float32)
    for i in range(n): 
        for j in range(i+1, n):
            distance = parameters[i].data - parameters[j].data
            distance = (distance*distance).sum()
            dist[i][j] = distance
            dist[j][i] = distance
    k = n - args['byzantine_size'] - 2 + 1  
    topv, _ = dist.topk(k=k, dim=1)
    sumdist = - topv.sum(dim=1)
    resindex = sumdist.topk(theta)[1].data
    for i in range(theta):
        result_1[i] = parameters[resindex[i]]
        
    topv, _ = result_1.topk(k=theta, dim=0)
    for i in range(gamma):
        if i % 2 == 0:
            result_2[i] = topv[int(theta//2 - i//2)]
        else:
            result_2[i] = topv[int(theta//2 + (i+1)/2)]
    return result_2.mean(dim=0)

             
            
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")
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
parser.add_argument('-d', '--defence', type=str, default='Krum', help='others defences') # FedAvg Median Krum Bulyan


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
    
    testDataLoader = myClients.test_data_loader
    
    honest_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'])]
    byzantine_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'],args['num_of_clients'])]
    

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        Upload_Parameters = []
        honest_all_weight = None
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
            
        if args['defence'] == 'Krum':
            global_parameters =Krum(Upload_Parameters)
        elif args['defence'] == 'FedAvg':
            global_parameters =FedAvg(Upload_Parameters)
        elif args['defence'] == 'Median':
            global_parameters = Median(Upload_Parameters)
        elif args['defence'] == 'Bulyan':
            global_parameters = Bulyan(Upload_Parameters)
            
     
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
    '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_defence{}_attack{}'.format(args['data_name'],
                                                           args['num_comm'], 
                                                           args['epoch'],
                                                           args['batchsize'],
                                                           args['learning_rate'],
                                                           args['num_of_clients'],
                                                           args['defence'],
                                                           args['pattern'])))


