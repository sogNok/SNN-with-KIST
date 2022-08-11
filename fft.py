import wfdb
import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import torch

filt = 1024
data = torch.zeros(1, filt)#, dtype=torch.uint8) 
label = torch.zeros(1, dtype=torch.int64)
first_set = [800, 803, 806, 840, 849, 853, 855, 856, 861, 872, 893]
second_set = [804, 807, 809, 823, 824, 844, 846, 847, 863, 871, 873, 875, 880, 881, 884, 891, 894]
sum_set = first_set + second_set
print(len(first_set))
print(len(second_set))
print(len(sum_set))

for idx in sum_set: # range(800, 900):
    
    path=str("1.0.0/"+str(idx))
    
    if os.path.isfile(path+".atr"): record_data_dic = wfdb.rdrecord(path, channels=[0])
    else: continue
    print(path)

    record_data = record_data_dic.__dict__
    ecg = record_data["p_signal"]
        
    # sampling by 10s
    ecg = ecg.reshape(-1,7680)

    annotation_dic = wfdb.rdann(path, "atr")
    annotation = annotation_dic.__dict__

    R = annotation["sample"]
    R = np.append(R, R[-1]+ 7680)   # for count last label
    
    '''
    for i,_ in enumerate(ecg):
        Emin = np.quantile(ecg[i], 0.25)
        Emax = np.quantile(ecg[i], 0.75)
        Equar = 1.5 * (Emax - Emin)
        Emin = Emin - Equar
        Emax = Emax + Equar
        ecg[i] = np.clip(ecg[i], Emin, Emax)
    '''

    # FFT
    ecg = np.fft.fft(ecg, n=15360)
    ecg = ecg[:,range(12, 12 + filt)]
    ecg = abs(ecg)
    
    # min_max Normalization
    for i,_ in enumerate(ecg):
        Emin = np.min(ecg[i])
        Emax = np.max(ecg[i])
        Emax = Emax-Emin
        ecg[i] = (ecg[i] - Emin) / Emax
    ecg = ecg * 255
    
    ecg = torch.from_numpy(ecg)
    ecg = ecg.to(torch.float32)

    # for labeling
    last_idx = 0 
    last_count = 0

    for i, peak in enumerate((R)):
        if int(peak / 7680) > last_count:
            rate = (i - last_idx)
            '''
            if rate < 80:       rate = 0
            else:               rate = 1
    
            if rate < 100:       rate = 0
            elif rate < 120:     rate = 1
            elif rate < 140:     rate = 2
            elif rate < 160:     rate = 3
            #elif rate < 180:    rate = 4
            else:               rate = 4
            '''

            last_count += 1
            last_idx = i

            rate_tensor = torch.tensor([rate], dtype=torch.int64)
            label = torch.cat([label, rate_tensor], dim=0)
     
    if(idx == 860) : label = label[:-1] # sub 860 has R peak when 00:30:00(last)
    data = torch.cat([data,ecg], dim=0)
    
data = data[1: ,]
label = label[1:]
print(data.shape)
print(label.shape)

'''

#data[data < 150] = 0
#data[data >= 150] = 1

for idx, v in enumerate((data)):
    tmp     = 0
    delta   = 0.1    
    Uthr    = v[0].item() + delta
    Lthr    = v[0].item() - delta
    for i, vv in enumerate((v)):
        if i == 0: v[i] = 0
        elif v[i].item() > Uthr:
            tmp = Uthr
            Uthr += delta
            Lthr = tmp
            v[i] = 1
        elif v[i].item() < Lthr:
            tmp = Lthr
            Lthr -= delta
            Uthr = tmp
            v[i] = 0
        else: v[i] = 0

    #vv = v.sum(dim=0)
    #if vv > 100:
    print('index: ',idx/180, '    sum: ',v.sum(dim=0))

data = data.view(14040,1280,1)
data = data.to(torch.bool)

tata = data
print(tata.shape)
print(data.view(-1,1280)[0,:100])
print(tata[0,0,:100])
'''

training = (data[:600], label[:600])
test = (data[600:], label[600:])

torch.save(training, 'training_fft_2n2.pt')
torch.save(test, 'test_fft_2n2.pt')
