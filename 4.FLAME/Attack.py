import torch
import numpy as np

# Untargeted Attack
def Gaussian_attack(honest_weight):  
    mu = honest_weight.float().mean(dim=0)
    std = 30**0.5
    W_grad = torch.normal(mu,std) 
    return W_grad

def Sign_flipping_attack(honest_weight): 
    W_grad = honest_weight.float().mean(dim=0)*(-3) + np.random.uniform()
    return W_grad

def ZeroGradient_attack(honest_weight, byzantine_Size=10): 
    W_grad = - honest_weight.float().sum(dim=0)/ byzantine_Size + np.random.uniform()
    return W_grad



# The backdoor data poisoning attack only modifies the training dataset with 80% of the poisoned data
def backdoor_poisoning_data(client, data_name):
    if data_name == 'mnist':
        backdoor = torch.ones([6,6])*2.821  #(255/255.0-0.1307)/0.3081
        
    elif data_name == 'cifar_10':
        backdoor = torch.ones([3,6,6])
        backdoor[0] = backdoor[0]*2.514
        backdoor[1] = backdoor[1]*2.597
        backdoor[2] = backdoor[2]*2.754
    
    for i in range(int(client.local_label.size()[0]*0.8)):
        if data_name == 'mnist':
            client.local_data[i].view(28,28)[0:6,0:6] = backdoor  
            client.local_data[i] = client.local_data[i].view(784,) 
            
        elif data_name == 'cifar_10':
            for j in range(3):
                client.local_data[i][j][0:6,0:6] = backdoor[j]
        
        client.local_label[i] = 0
    return client

# Gradient expansion in main.py
def model_replacement_attack_data(client, data_name): 
    if data_name == 'mnist':
        backdoor = torch.ones([6,6])*2.821  #(255/255.0-0.1307)/0.3081
        
    elif data_name == 'cifar_10':
        backdoor = torch.ones([3,6,6])
        backdoor[0] = backdoor[0]*2.514
        backdoor[1] = backdoor[1]*2.597
        backdoor[2] = backdoor[2]*2.754
    
    for i in range(int(client.local_label.size()[0]/2)): # 50% of the poisoned data
        if data_name == 'mnist':
            client.local_data[i].view(28,28)[0:6,0:6] = backdoor  
            client.local_data[i] = client.local_data[i].view(784,) 
            
        elif data_name == 'cifar_10':
            for j in range(3):
                client.local_data[i][j][0:6,0:6] = backdoor[j]
        
        client.local_label[i] = 0
    return client
    

def MPAF(honest_weight):    
    random_tensor = torch.rand_like(honest_weight.float().mean(dim=0)) * 1e3
    return random_tensor


def euclidean_distance(tensor1, tensor2):
    return torch.sqrt(((tensor1 - tensor2) ** 2).sum())


def AGR_agnostic(honest_weight):
    n = honest_weight.shape[0]

    max_distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            distance = euclidean_distance(honest_weight[i], honest_weight[j])
            max_distance = max(max_distance, distance)

    k = torch.randint(0, n, (1,)).item()
    temp = honest_weight[k].clone()

    step = 0.1
    while True:
        all_distances = [euclidean_distance(temp, honest_weight[i]) for i in range(n)]
        # If the maximum Euclidean distance between temp and honest_weight[i] is less than d, continue to increase temp
        if all(d < max_distance for d in all_distances):
            temp += step
        else:
            return temp
