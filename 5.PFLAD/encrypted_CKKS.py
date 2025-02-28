import tenseal as ts
import torch
import numpy as np
import copy

def create_context():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    context.global_scale = 2 ** 21
    context.generate_galois_keys()
    return context


context = create_context()


def en_fc(input_encrypted, linear): # linear
    weight = copy.deepcopy(linear.weight).detach().cpu().numpy()
    bias = copy.deepcopy(linear.bias).detach().cpu().numpy().reshape(1,weight.shape[0])
    export_bias = np.zeros([input_encrypted.shape[0],bias.shape[1]])  # Extended bias
    for i in range(input_encrypted.shape[0]):
        export_bias[i] = bias
    input_encrypted = input_encrypted.mm(weight.reshape(weight.shape[1],-1)).add(export_bias)
    # print("***")
    return input_encrypted


def en_predict(Parameters, network, dense):
    num_of_clients = Parameters.shape[0]
    Parameters = Parameters.view(num_of_clients,-1).cpu()
    input_encrypted = ts.ckks_tensor(context, Parameters)
    print("Encryption complete")
    # fc1 layer
    input_encrypted = en_fc(input_encrypted, network.linear1)
    
    # fc2 layer    
    input_encrypted = en_fc(input_encrypted, network.linear2)
    
    if dense == 3:    
        # fc3 layer + square
        input_encrypted = en_fc(input_encrypted,network.linear3).square()
    else:
        # fc3 layer
        input_encrypted = en_fc(input_encrypted,network.linear3)
        # fc4 layer + square
        input_encrypted = en_fc(input_encrypted,network.linear4).square()        
    
    predict = np.array(input_encrypted.decrypt().tolist()).reshape(num_of_clients, )
    
    return predict
    
    
    
def en_FedAvg(Upload_Parameters, malicious):
    count = 0
    num_of_clients = len(Upload_Parameters)
    for j in malicious: 
        del(Upload_Parameters[j-count])
        count = count + 1
    sum_parameters = None
    for j in range(len(Upload_Parameters)):
        print("Encryption Completion Aggregation Progress Client:{}".format(j))
        count = 0
        if sum_parameters is None:
            sum_parameters = {}
            for key in Upload_Parameters[j]:
                
                if Upload_Parameters[j][key].size() == torch.Size([]) or count > 5:
                    sum_parameters[key] = Upload_Parameters[j][key]
                else:
                    temp = copy.deepcopy(Upload_Parameters[j][key]).detach().cpu()
                    temp = ts.ckks_tensor(context, temp)
                    sum_parameters[key] = temp

                # print("Processing the {}th parameter: {}".format(count, key))
                count = count + 1
        else:
            for key in Upload_Parameters[j]:
                if Upload_Parameters[j][key].size() == torch.Size([]) or count > 5:
                    sum_parameters[key] = sum_parameters[key] + Upload_Parameters[j][key]
                else:
                    temp = copy.deepcopy(Upload_Parameters[j][key]).detach().cpu()
                    temp = ts.ckks_tensor(context, temp)  
                    sum_parameters[key] = sum_parameters[key] + temp
                    
                # print("Processing the {}th parameter: {}".format(count, key))
                count = count + 1
                
        # print("sum_parameters:{}".format(sum_parameters))

    count = 0
    for key in sum_parameters:
        if Upload_Parameters[0][key].size() == torch.Size([]) or count > 5:
            sum_parameters[key] = sum_parameters[key]*(1.0/(num_of_clients-len(malicious)))
            
        else:
            sum_parameters[key] = sum_parameters[key]*(1.0/(num_of_clients-len(malicious)))
            sum_parameters[key] = np.array(sum_parameters[key].decrypt().tolist())
            sum_parameters[key] = torch.tensor(sum_parameters[key])

        # print("Processing the {}th parameter: {}".format(count, key))
        count = count + 1
    return sum_parameters

    


    
