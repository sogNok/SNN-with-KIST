import torch
import random


tr = torch.load('training_fft_1r.pt')
te = torch.load('test_fft_1r.pt')
tt = torch.load('test_fft_2r4.pt')

print(tr[0].shape)
print(tr[1].shape)
print(te[0].shape)
print(te[1].shape)
print(tt[0].shape)

print(tr[1][1])
print(tt[1][1])

exit(0)
print("hi")
'''
c0=0
c1=0
c2=0
c3=0
c4=0
c5=0
for idx, val in enumerate(tr[1]):
    if val == 0:    c0+=1
    elif val == 1:  c1+=1
    elif val == 2:  c2+=1
    elif val == 3:  c3+=1
    elif val == 4:  c4+=1
    else:           c5+=1

for idx, val in enumerate(te[1]):
    if val == 0:    c0+=1
    elif val == 1:  c1+=1
    elif val == 2:  c2+=1
    elif val == 3:  c3+=1
    elif val == 4:  c4+=1
    else:           c5+=1


print("0: ",c0," 1: ",c1," 2: ",c2," 3: ",c3," 4: ",c4," 5: ",c5)
'''

tr = training = 'training_fft_slide_3.pt'
te = test = 'test_fft_slide_3.pt'

training = torch.load(training)
test = torch.load(test)

data = torch.cat([training[0],test[0]], dim=0)
target = torch.cat([training[1],test[1]], dim=0)

tmp = [[x,y] for x, y in zip(data, target)]

random.shuffle(tmp)

data_list = [n[0] for n in tmp]
target_list = [n[1] for n in tmp]

for idx in range(len(data_list)):
    data[idx] = data_list[idx]
    target[idx] = target_list[idx]

training = (data[:60000], target[:60000])
test = (data[60000:], target[60000:])

torch.save(training, tr)
torch.save(test, te)
print(tr)

print('done')
