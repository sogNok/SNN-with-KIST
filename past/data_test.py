import torch
import random

training = 'training_t1mc6.pt'
test = 'test_t1mc6.pt'

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

training = (data[:10000], target[:10000])
test = (data[10000:], target[10000:])

torch.save(training, 'training_t1rc6.pt')
torch.save(test, 'test_t1rc6.pt')
