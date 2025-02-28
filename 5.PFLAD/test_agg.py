import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from encrypted_CKKS import en_FedAvg
from Models import ResNet18, Mnist_CNN
from clients import ClientsGroup

def FedAvg(Upload_Parameters, malicious):
    count = 0
    for j in malicious:
        del(Upload_Parameters[j-count])
        count = count + 1
    sum_parameters = None
    for j in range(len(Upload_Parameters)):
        if sum_parameters is None:
            sum_parameters = {}
            for key in Upload_Parameters[j]:
                sum_parameters[key] = Upload_Parameters[j][key]
        else:
            for key in Upload_Parameters[j]:
                sum_parameters[key] = sum_parameters[key] + Upload_Parameters[j][key]
    
    for key in sum_parameters:
        sum_parameters[key] = sum_parameters[key]/(args['num_of_clients']-len(malicious))
    return sum_parameters
        
    
            
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="PFLAD")
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=10, help='numer of the clients')
parser.add_argument('-by', '--byzantine_size', type=int, default=0, help='number of the byzantine clients')
parser.add_argument('-p','--pattern', type=int, default=5,help='patterns of attack methods (0,1,2,3,4,5,6)')
parser.add_argument('-E', '--epoch', type=int, default=5, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=64, help='local train batch size')
parser.add_argument('-data', '--data_name', type=str, default='mnist', help='the data to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.1, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=1, help="model validation frequency(of communications)")
parser.add_argument('-ncomm', '--num_comm', type=int, default=10, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
parser.add_argument('-cen', '--central_data_size', type=int, default=200, help='central data size in server')
parser.add_argument('-pro', '--central_data_pro', type=float, default=0.1, help='central data pro in server')
parser.add_argument('-alpha','--alpha', type=float, default=0.5, help='importance weights for direction and length')




def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    args = parser.parse_args()
    args = args.__dict__

    acc_list=[]
    malicious_list = []
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
        net = torch.nn.DataParallel(net)
        
    net = net.to(dev)
    loss_func = F.cross_entropy
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])

    myClients = ClientsGroup(args['data_name'], args['IID'], args['num_of_clients'],dev)
    myClients.get_central_data(args['central_data_size'],args['central_data_pro'])
    
    testDataLoader = myClients.test_data_loader
    
    honest_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'])]
    byzantine_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'],args['num_of_clients'])]
    
    
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):
        print("communicate round {}".format(i+1))

        Upload_Parameters = []
        

        for client in honest_clients:
            local_parameters = myClients.clients_set[client].localTrain(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            Upload_Parameters.append(local_parameters)
            
        malicious = []
        global_parameters = en_FedAvg(Upload_Parameters, malicious) 
        
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

