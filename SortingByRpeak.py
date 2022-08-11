import wfdb
import matplotlib.pyplot as plt
import os
import os.path
import numpy as np
import torch

'''
record_data_dic = wfdb.rdrecord("1.0.0/800", channels=[0])
record_data = record_data_dic.__dict__

ecg = record_data["p_signal"]
annotation_dic = wfdb.rdann("1.0.0/800", "atr")
annotation = annotation_dic.__dict__

R = annotation["sample"]
label = annotation["symbol"]
'''

data = torch.zeros(1, 1280)#, dtype=torch.uint8) 
label = torch.zeros(1, dtype=torch.int64)

for idx in range(800, 900):

    path=str("1.0.0/"+str(idx))
    print(path)
    if os.path.isfile(path+".atr"): record_data_dic = wfdb.rdrecord(path, channels=[0])
    else: continue
    
    record_data = record_data_dic.__dict__
    ecg = record_data["p_signal"]
        
     
    # min_max Normalization
    Emin = np.min(ecg)
    Emax = np.max(ecg)
    Emax = Emax-Emin
    ecg = (ecg - Emin) / Emax
    ecg = ecg * 255
    

    # sampling by 10s
    #ecg = ecg.reshape(-1,1280)

    annotation_dic = wfdb.rdann(path, "atr")
    annotation = annotation_dic.__dict__
    print(annotation)
    R = annotation["sample"]
    #R = np.append(R, R[-1]+ 1280)   # for count last label
    
    print(ecg[R[0]])
    print(ecg[R[0]+2])

    exit(0)
    ecg = torch.from_numpy(ecg)
    ecg = ecg.to(torch.float32)

    # for labeling
    last_idx = 0 
    last_peak  = R[0]
 
    for i, peak in enumerate((R)):
        if peak - last_peak > 1280:
            rate = (i - last_idx) * 6
            '''
            if rate < 80:       rate = 0
            else:               rate = 1
            '''
            if rate < 60:       rate = 0
            elif rate < 70:     rate = 1
            elif rate < 80:     rate = 2
            elif rate < 90:     rate = 3
            elif rate < 100:    rate = 4
            else:               rate = 5
            
            exg = ecg[last_peak:last_peak+1280].permute(1, 0)
            last_peak = peak
            last_idx = i
            
            rate_tensor = torch.tensor([rate], dtype=torch.int64)
            label = torch.cat([label, rate_tensor], dim=0)
            data = torch.cat([data,exg], dim=0)
    
    #if(idx == 860) : label = label[:-1] # sub 860 has R peak when 00:30:00(last)
    
data = data[1: ,]
label = label[1:]

print(data.shape)
print(label.shape)


training = (data[:10000], label[:10000])
test = (data[10000:], label[10000:])

torch.save(training, 'training_srp.pt')
torch.save(test, 'test_srp.pt')

exit(0)
print(training[0].shape)
print(training[1].shape)
print(test[0].shape)
print(test[1].shape)

c0=0
c1=0
c2=0
c3=0
c4=0
c5=0
for idx, val in enumerate((training[1][:250])):
    if val == 0:    c0+=1
    elif val == 1:  c1+=1
    elif val == 2:  c2+=1
    elif val == 3:  c3+=1
    elif val == 4:  c4+=1
    else:           c5+=1

print("0: ",c0," 1: ",c1," 2: ",c2," 3: ",c3," 4: ",c4," 5: ",c5)
print(training[1][:250])




print(training[0][0][:128])
print(training[0][0][128:256])
'''
print(training[0][0][256:384])
print(training[0][0][384:512])
print(training[0][0][512:640])
print(training[0][0][640:768])
print(training[0][0][768:896])
print(training[0][0][896:1024])
print(training[0][0][1024:1152])
print(training[0][0][1152:1280])
'''
