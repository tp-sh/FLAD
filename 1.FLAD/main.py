import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import Attack
from sklearn.cluster import DBSCAN
from Models import ResNet18, Mnist_CNN, LinearNet
from clients import ClientsGroup

def cos(a,b):
    res = np.sum(a*b.T)/((np.sqrt(np.sum(a * a.T)) + 1e-9) * (np.sqrt(np.sum(b * b.T))) + 1e-9)
    return res


def Feature_extraction_model(Central_par,dev):
    num = len(Central_par)
    if args['data_name'] == 'mnist':
        kernel1_train = torch.zeros(num,10,1,5,5)  
        # kernel2_train = torch.zeros(num,20,10,5,5)
        weight3_train = torch.zeros(num,10,320)   
         
    elif args['data_name'] == 'cifar_10':
        kernel1_train = torch.zeros(num,64,3,3,3)   
        # kernel2_train = torch.zeros(num,16,16,3,3) 
        weight3_train = torch.zeros(num,10,512) 
    count = 0
    for W_server in Central_par:
        if args['data_name'] == 'cifar_10':
            kernel1_train[count] = W_server['module.conv1.weight'].data 
            # kernel2_train[count] = W_server['layer1.0.left.0.weight'].data 
            weight3_train[count] = W_server['module.fc.weight'].data
            
        elif args['data_name'] == 'mnist':
            kernel1_train[count] = W_server['conv1.weight'].data 
            # kernel2_train[count] = W_server['conv2.weight'].data
            weight3_train[count] = W_server['fc.weight'].data
                    
        count = count + 1
    FC,Std,Dis = {},{},{}
    
    if args['data_name'] == 'mnist':
        FC['conv1.weight'], Std['conv1.weight'], Dis['conv1.weight'] = train_Linear(kernel1_train,10*1*5*5,dev)
        # FC['conv2.weight'], Std['conv2.weight'], Dis['conv2.weight'] = train_Linear(kernel2_train,20*10*5*5,dev) 
        FC['fc.weight'], Std['fc.weight'], Dis['fc.weight'] = train_Linear(weight3_train,10*320,dev)
        
    elif args['data_name'] == 'cifar_10':
        FC['conv1.weight'], Std['conv1.weight'], Dis['conv1.weight'] = train_Linear(kernel1_train,64*3*3*3,dev)
        # FC['layer1.0.left.0.weight'], Std['layer1.0.left.0.weight'], Dis['layer1.0.left.0.weight'] = train_Linear(kernel2_train,16*16*3*3,dev)
        FC['fc.weight'], Std['fc.weight'], Dis['fc.weight'] = train_Linear(weight3_train,10*512,dev)
    
    return FC, Std, Dis
    


def train_Linear(weight_train,dimen,dev):
    weight_train = weight_train.view(weight_train.size(0),-1).to(dev)
    train_loader_weight = DataLoader(dataset = weight_train, batch_size=1, shuffle=True)
    test_model = LinearNet(dimen)
    test_model = test_model.to(dev)
    optimizer = optim.Adam(test_model.parameters(), lr = 0.001)

    criterion_test = torch.nn.MSELoss(reduction='sum')
    label = torch.tensor([[1.0]])
    # training w
    for epoch in range(20):
        for idx, train_data in enumerate(train_loader_weight,0): 
            label = label.to(dev)
            output = test_model(train_data)
            loss = criterion_test(output, label) 
            optimizer.zero_grad()
            loss.backward()                
            optimizer.step() 
    all_output = test_model(weight_train)
    std = all_output.mean(dim=0)
    dis = all_output.max()- all_output.min()
    # print("feature of server dataset:{}".format(all_output))
    return test_model, std, dis
    
    
def neural_network_feature_extraction(Upload_Parameters, FC, Std, Dis, dev):
    if args['data_name'] == 'mnist':
        kernel1 = torch.zeros(args['num_of_clients'],10,1,5,5).to(dev)
        # kernel2 = torch.zeros(args['num_of_clients'],20,10,5,5).to(dev)
        weight3 = torch.zeros(args['num_of_clients'],10,320).to(dev) 
    elif args['data_name'] == 'cifar_10':
        kernel1 = torch.zeros(args['num_of_clients'],64,3,3,3).to(dev) 
        # kernel2 = torch.zeros(args['num_of_clients'],16,16,3,3).to(dev) 
        weight3 = torch.zeros(args['num_of_clients'],10,512).to(dev)     
    
    count = 0
    for W_local in Upload_Parameters:
        if args['data_name'] == 'mnist':
            kernel1[count] = W_local['conv1.weight'].data
            #kernel2[count] = W_local['conv2.weight'].data
            weight3[count] = W_local['fc.weight'].data
            
        elif args['data_name'] == 'cifar_10':
            kernel1[count] = W_local['module.conv1.weight'].data
            #kernel2[count] = W_local['layer1.0.left.0.weight'].data
            weight3[count] = W_local['module.fc.weight'].data           
            
        count = count + 1
    
    feature = np.zeros([args['num_of_clients'],2])
    feature[:,0] = FC['conv1.weight'](kernel1.view(args['num_of_clients'],-1)).cpu().detach().numpy().reshape(args['num_of_clients'],)
    feature[:,1] = FC['fc.weight'](weight3.view(args['num_of_clients'],-1)).cpu().detach().numpy().reshape(args['num_of_clients'],)
    
    honest_std = np.array([Std['conv1.weight'].item(), Std['fc.weight'].item()])
    honest_L2 = np.sqrt(np.sum(honest_std * honest_std.T)) + 1e-9
    print("honest feature in server dataset: {}".format(honest_std))

    # print("feature of clinet model:{}".format(feature))
    
    eps1 = Dis['conv1.weight'].item()
    # eps2 = Dis['layer1.0.left.0.weight'].item()
    eps3 = Dis['fc.weight'].item()
    
    # print("eps1:{},eps3:{}".format(eps1,eps3)) 
    eps = (eps1**2+eps3**2)**(0.5) 
    print("eps: {}".format(eps))
    db = DBSCAN(eps = eps, min_samples=3).fit( feature )
    label_list = db.labels_
    print("label of all clients: {}".format(label_list))

    temp = np.zeros([1,2]) 
    label_mean, label_score = {},{}
    labelcount = set(db.labels_)
    for label in labelcount:
        if label != -1:
            for client in range(args['num_of_clients']):
                if label_list[client] == label:
                    temp = np.concatenate((temp,feature[client,:].reshape(1,2)),axis=0)
            
            label_mean[label] = temp.mean(axis=0) * temp.shape[0]/(temp.shape[0]-1)
            
            print("label: {}, mean: {}".format(label,label_mean[label]))
            cosin = cos(label_mean[label],honest_std)
            length = abs(np.sqrt(np.sum(label_mean[label]*label_mean[label].T))/honest_L2 - 1.0)
            label_score[label] = args['alpha']*cosin - (1-args['alpha'])*length
            print("cos: {}, length: {}".format(cosin,length))

    
    label_score = sorted(label_score.items(), key=lambda x: x[1], reverse=True)
    print("label score: {}".format(label_score))

    honest_label = label_score[0][0]       
    malicious = []
    for client in range(args['num_of_clients']):
        if label_list[client]!=honest_label:
            malicious.append(client)        
    return malicious


def FedAvg(Upload_Parameters, malicious):
    count = 0
    # It is important to ensure that the numbers in malicious are numbered from smallest to largest
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
    
             
            
parser = argparse.ArgumentParser(description='FLAD')
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
parser.add_argument('-iid', '--IID', type=bool, default=True, help='the way to allocate data to clients')
parser.add_argument('-cen', '--central_data_size', type=int, default=300, help='central data size in server')
parser.add_argument('-pro', '--central_data_pro', type=float, default=0.1, help='central data pro in server')
parser.add_argument('-alpha','--alpha', type=float, default=0.5, help='importance weights for direction and length')
args = parser.parse_args()
args = args.__dict__


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    acc_list=[]
    malicious_list = []
    test_mkdir(args['save_path'])
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
    myClients.get_central_data(args['central_data_size'],args['central_data_pro'])
    
    testDataLoader = myClients.test_data_loader
    
    honest_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'])]
    byzantine_clients = ['client{}'.format(i) for i in range(args['num_of_clients']-args['byzantine_size'],args['num_of_clients'])]
    
    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    for i in range(args['num_comm']):
        
        print("\ncommunicate round {}".format(i+1))

        Central_par = myClients.centralTrain(args['epoch'], args['batchsize'], net,                                        
                                                                         loss_func, opti, global_parameters)
        FC, Std, Dis = Feature_extraction_model(Central_par,dev)
        
        Upload_Parameters = []
        
        honest_all_weight = None
        for client in honest_clients:
            # print(client)
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
            
        
        # malicious is a list of malicious node numbers
        # print("malious")
        malicious = neural_network_feature_extraction(Upload_Parameters, FC, Std, Dis, dev)
        print("malicious clients: {}".format(malicious))
        
        global_parameters = FedAvg(Upload_Parameters, malicious)
        
        
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
                malicious_list.append(malicious)
    
    # save model
    torch.save(net, os.path.join(args['save_path'],
    '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cen{}_attack{}'.format(args['data_name'],
                                                           args['num_comm'], 
                                                           args['epoch'],
                                                           args['batchsize'],
                                                           args['learning_rate'],
                                                           args['num_of_clients'],
                                                           args['central_data_size'],
                                                           args['pattern'])))


