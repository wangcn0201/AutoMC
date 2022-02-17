
import torch
import math
import numpy as np
from utils import *
import models
from train import *
import os, pickle, copy


def print_sum(x):
    print(np.sum(np.array(x.cpu().detach().numpy())))
f1 = open('model1.pl', 'rb')
model1 = pickle.load(f1).cuda()
f1.close()
f2 = open('model2.pl', 'rb')
model2 = pickle.load(f2).cuda()
f2.close()

x1 = torch.zeros((1, 16, 2, 2)).cuda()
x2 = torch.zeros((1, 16, 2, 2)).cuda()

# res1 = model1.conv1(x1)
# res2 = model2.conv1(x2)
# print_sum(res1)
# print_sum(res2)

# print('*****')
with torch.no_grad():
    res1 = model1.layer1[0].bn1(x1)
    print_sum(res1)
    res2 = model2.layer1[0].bn1(x1)
    print_sum(res2)
print(res1 - res2)

print('*****')

for x in model1.layer1[0].bn1.parameters():
    print(x, end=' ')
print('')
for x in model2.layer1[0].bn1.parameters():
    print(x, end=' ')
print('')

'''
x = self.conv1(x)

x = self.layer1(x)  # 32x32
x = self.layer2(x)  # 16x16
x = self.layer3(x)  # 8x8
x = self.bn(x)
x = self.act(x)

x = self.avgpool(x)
x = x.view(x.size(0), -1)
x = self.fc(x)
'''
# f1 = open('modules1.pl', 'rb')
# modules1 = pickle.load(f1)
# f1.close()
# f2 = open('modules2.pl', 'rb')
# modules2 = pickle.load(f2)
# f2.close()
# for module1, module2 in zip(modules1, modules2):
#     for item, new_item in zip(module1[0].parameters(), module2[0].parameters()):
#         print(item.shape)
#         str1 = '{:.4f}'.format(np.sum(np.array(module1[0].weight.data.cpu())))
#         str2 = '{:.4f}'.format(np.sum(np.array(module2[0].weight.data.cpu())))
#         print(str1 == str2, end=' ')
#         print(str1, end=' ')
#         print(str2, end=' ')
#         print(module1[2])

'''
a1 = np.array([[2, 3, 4]])
a2 = np.array([
    [1, 2, 0, 4], 
    [2, 3, 0, 5],
    [7, 1, 0, 9]
])
a3 = np.array([
    [2, 7, 0], 
    [7, 3, 0], 
    [0, 0, 0], 
    [3, 0, 0],
])
a4 = np.array([
    [2], 
    [7], 
    [0]
])
print(np.dot(np.dot(np.dot(a1, a2), a3), a4))

a1 = np.array([[2, 3, 4]])
a2 = np.array([
    [1, 2, 4], 
    [2, 3, 5],
    [7, 1, 9]
])
a3 = np.array([
    [2, 7], 
    [7, 3], 
    [3, 0],
])
a4 = np.array([
    [2], 
    [7]
])
print(np.dot(np.dot(np.dot(a1, a2), a3), a4))
'''
