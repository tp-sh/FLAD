import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet
import copy


class client(object):
    def __init__(self, local_data, local_label, dev):
        self.local_data = local_data # local_data, local_label: tensor
        self.local_label = local_label
        self.dev = dev
        self.local_parameters = None

    def localTrain(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)
        self.train = TensorDataset(self.local_data, self.local_label)
        train_loader = DataLoader(self.train, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in train_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                opti.zero_grad()
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()
                opti.step()
        self.local_parameters = Net.state_dict()
        return copy.deepcopy(Net.state_dict())


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}
        self.central_data = None
        self.test_data_loader = None
        if self.is_iid is True:    
            self.dataSetBalanceAllocation()
        else:
            self.dataSetNotBalanceAllocation()



    def dataSetBalanceAllocation(self):
        DataSet = GetDataSet(self.data_set_name, self.is_iid)

        if self.data_set_name == 'mnist':
            self.test_data = torch.tensor(DataSet.test_data)
            self.test_label = torch.tensor(DataSet.test_label)
            self.test_label = torch.argmax(torch.tensor(DataSet.test_label), dim=1)

        elif self.data_set_name == 'cifar_10':
            self.test_data = DataSet.test_data
            self.test_label = DataSet.test_label
        
        self.test_data_loader = DataLoader(TensorDataset(self.test_data,self.test_label), batch_size=100, shuffle=False)
        train_data = DataSet.train_data
        train_label = DataSet.train_label
        local_size = DataSet.train_data_size // self.num_of_clients
        for i in range(self.num_of_clients):
            local_data = train_data[ i * local_size: i * local_size + local_size]
            local_label = train_label[i * local_size: i * local_size + local_size]
            if self.data_set_name == 'mnist':
                local_label = np.argmax(local_label, axis=1)
                one = client(torch.tensor(local_data), torch.tensor(local_label), self.dev)
            elif self.data_set_name == 'cifar_10':
                one = client(local_data, local_label, self.dev)
            self.clients_set['client{}'.format(i)] = one


    def dataSetNotBalanceAllocation(self):
        DataSet = GetDataSet(self.data_set_name, self.is_iid)

        if self.data_set_name == 'mnist':
            self.test_data = torch.tensor(DataSet.test_data)
            self.test_label = torch.tensor(DataSet.test_label)
            self.test_label = torch.argmax(torch.tensor(DataSet.test_label), dim=1)
            samples, non_iid, datasize  = 6000, 0.8, 60000

        elif self.data_set_name == 'cifar_10':
            self.test_data = DataSet.test_data
            self.test_label = DataSet.test_label
            samples, non_iid, datasize = 5000, 0.5, 50000

        self.test_data_loader = DataLoader(TensorDataset(self.test_data,self.test_label), batch_size=100, shuffle=False)
        
        train_data = DataSet.train_data
        train_label = DataSet.train_label
        
        class_data = {}
        class_label = {}
        
        # Classification of Data Segmentation
        for i in range(10):
            class_data[i] =  train_data[ i * samples: i * samples + samples] 
            class_label[i] =  train_label[ i * samples: i * samples + samples]
        
        # Disruption of non-disaggregated data
        random_data = class_data[0][int(samples*non_iid): samples] 
        random_label = class_label[0][int(samples*non_iid): samples] 
        for i in range(1,10):
            random_data = np.concatenate((random_data, class_data[i][int(samples*non_iid): samples]),axis=0)
            random_label = np.concatenate((random_label, class_label[i][int(samples*non_iid): samples]),axis=0)
        
        order = np.arange(random_data.shape[0])
        np.random.shuffle(order)        
        random_data = random_data[order]
        random_label = random_label[order]

        groups_num = self.num_of_clients // 10
        
        # Client Grouping Data
        for i in range(self.num_of_clients):
            group = i // groups_num
            mod = i % groups_num
            temp = int((datasize//self.num_of_clients)*non_iid)
            temp1 = int((datasize//self.num_of_clients)*(1-non_iid))
            local_data = np.concatenate((class_data[group][temp*mod : mod*temp + temp], random_data[temp1*i : temp1*i + temp1]),axis=0)
            local_label = np.concatenate((class_label[group][temp*mod : mod*temp + temp], random_label[temp1*i : temp1*i + temp1]),axis=0)
            if self.data_set_name == 'mnist':
                local_label = np.argmax(local_label, axis=1)
            one = client(torch.tensor(local_data), torch.tensor(local_label), self.dev)
            self.clients_set['client{}'.format(i)] = one

