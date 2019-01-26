
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import torch.optim as optim
import os
import math
from torch.autograd import Variable

class co_train_classifier(nn.Module):
    def __init__(self):
        super(co_train_classifier, self).__init__()
        self.c1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(128)
        self.r1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(128)
        self.r2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.r3 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.m1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d1 = nn.Dropout2d(p=0.5)

        self.c4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b4 = nn.BatchNorm2d(256)
        self.r4 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b5 = nn.BatchNorm2d(256)
        self.r5 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.b6 = nn.BatchNorm2d(256)
        self.r6 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.m2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d2 = nn.Dropout2d(p=0.5)

        self.c7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.b7 = nn.BatchNorm2d(512)
        self.r7 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.b8 = nn.BatchNorm2d(256)
        self.r8 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.b9 = nn.BatchNorm2d(128)
        self.r9 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.fc = nn.Linear(128, 10)
        self.sf = nn.Softmax()

    def forward(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.r1(x)
        x = self.c2(x)
        x = self.b2(x)
        x = self.r2(x)
        x = self.c3(x)
        x = self.b3(x)
        x = self.r3(x)
        x = self.m1(x)
        x = self.d1(x)


        x = self.c4(x)
        x = self.b4(x)
        x = self.r4(x)
        x = self.c5(x)
        x = self.b5(x)
        x = self.r5(x)
        x = self.c6(x)
        x = self.b6(x)
        x = self.r6(x)
        x = self.m2(x)
        x = self.d2(x)

        x = self.c7(x)
        x = self.b7(x)
        x = self.r7(x)
        x = self.c8(x)
        x = self.b8(x)
        x = self.r8(x)
        x = self.c9(x)
        x = self.b9(x)
        x = self.r9(x)
        # print(x.size())
        x = torch.mean(x, dim=[2, 3]) # # Global average pooling
        x_logit = self.fc(x)
        x_softmax = self.sf(x_logit)
        return x_softmax, x_logit


def adjust_learning_rate(optimizer, epoch):
    """cosine scheduling"""
    lr = 0.05*(1.0+math.cos((epoch-1)*math.pi/600))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, -3, 3)
    # Return the perturbed image
    return perturbed_image

def jsd(p,q):
    softmax = nn.Softmax(dim = 1)
    kld = nn.KLDivLoss()
    p = softmax(p)
    q = softmax(q)
    return 0.5*kld(p, 0.5*(p + q)) + 0.5*kld(q, 0.5*(p + q))

def loss_sup():
    # CE, by default, is averaged over each loss element in the batch
    ce = nn.CrossEntropyLoss() 
    loss1 = ce(outputs1, labels1)
    loss2 = ce(outputs2, labels2) 
    return loss1+loss2

def loss_cot():
# the Jensen-Shannon divergence between p1(x) and p2(x)
    return jsd()

def loss_diff():
    ce = nn.CrossEntropyLoss() 
    loss1 = ce(output1, )
    loss2 = ce(outputs2,) 
    return loss1+loss2


batch_size = 100
# label data propotion 4600/50000 for cifar 10 
p_l = 0.09
p_u = 1 - 0.09

#standard data augmentation of cifar10
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])




testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=False, num_workers=2)
S_idx = []
U_idx = []
dataiter = iter(trainloader)
train = 10*[[]]
for i in range(50000):
    inputs, labels = dataiter.next()
    train[labels].append(i)

for i in range(10):
    S_idx = S_idx + train[i][0:400]
    U_idx = U_idx + train[i][400:]

S_sampler = torch.utils.data.SubsetRandomSampler(S_idx)
U_sampler = torch.utils.data.SubsetRandomSampler(U_idx)


S_loader1 = torch.utils.data.DataLoader(
        trainset, batch_size=9, sampler=S_sampler,
        num_workers=2, pin_memory=True)

S_loader2 = torch.utils.data.DataLoader(
        trainset, batch_size=9, sampler=S_sampler,
        num_workers=2, pin_memory=True)

U_loader = torch.utils.data.DataLoader(
        trainset, batch_size=91, sampler=U_sampler,
        num_workers=2, pin_memory=True)



step = len(trainset)/batch_size
# training
net1 = co_train_classifier()
net2 = co_train_classifier()
net1.cuda()
net2.cuda()
params = list(net1.parameters()) + list(net2.parameters())
# stochastic gradient descent with momentum 0.9 and weight decay0.0001 in paper page 7
optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0001)

net1.train()
net2.train()
for epoch in range(600):  # loop over the dataset multiple times
    
    lamda_sup = 1 
    lamda_cot_max = 10
    lamda_diff_max = 1
    if epoch < 80:
        lamda_cot = lamda_cot_max*math.exp(-5*(1-epoch/80)**2)
        lamda_diff = lamda_diff_max*math.exp(-5*(1-epoch/80)**2)
    else: 
        lamda_cot = lamda_cot_max
        lamda_diff = lamda_diff_max

    total1 = 0
    total2 = 0
    train_correct1 = 0
    train_correct2 = 0
    running_loss = 0.0
    ls = 0.0
    lc = 0.0 
    ld = 0.0
    i = 0
    S_iter1 = iter(S_loader1)
    S_iter2 = iter(S_loader1)
    U_iter = iter(U_loader)
    adjust_learning_rate(optimizer, epoch)
    if(epoch % 100 ==0):
        torch.save(net1.state_dict(), 'co_train_classifier_1.pkl')
        torch.save(net2.state_dict(), 'co_train_classifier_2.pkl')
    while(i < step):



        inputs_S1, labels_S1 = S_iter1.next()
        inputs_S2, labels_S2 = S_iter2.next()
        inputs_U, _ = U_iter.next()

        inputs_S1 = inputs_S1.cuda()
        labels_S1 = labels_S1.cuda()
        inputs_S2 = inputs_S2.cuda()
        labels_S2 = labels_S2.cuda()
        inputs_U = inputs_U.cuda()    
        # generate adversarial example 1
        inputs_S1.requires_grad = True         # Set requires_grad attribute of tensor. Important for Attack
        outputs1 = net1(inputs_S1)
        init_pred = outputs1.max(1, keepdim=True)[1]
        loss1 = F.nll_loss(outputs1, labels_S1)
        net1.zero_grad()
        loss1.backward()
        inputs_S_grad1 = inputs_S1.grad.data
        perturbed_data1 = fgsm_attack(inputs_S1, epsilon, inputs_S_grad1)

        # generate adversarial example 2
        inputs_S2.requires_grad = True         # Set requires_grad attribute of tensor. Important for Attack
        outputs2 = net1(inputs_S2)
        init_pred = outputs2.max(1, keepdim=True)[1]
        loss2 = F.nll_loss(outputs2, labels_S2)
        net2.zero_grad()
        loss2.backward()
        inputs_S_grad2 = inputs_S2.grad.data
        perturbed_data2 = fgsm_attack(inputs_S2, epsilon, inputs_S_grad2)

        _, perturbed_logit1 = net1(perturbed_data1)
        _, perturbed_logit2 = net2(perturbed_data2)

        U_p1, U_logit1 = net1(inputs_U)
        U_p2, U_logit2 = net2(inputs_U)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        _, logit1 = net1(inputs1)
        _, logit2 = net2(inputs2)
        
        loss_sup = loss_sup(logit1, logit2, labels_S1, labels_S2)
        loss_cot = loss_cot(U_p1, U_p2)
        loss_diff = loss_diff(logit1, logit2, perturbed_logit1, perturbed_logit2)
        
        total_loss = lamda_sup*loss_sup + lamda_cot*loss_cot + lamda_diff*loss_diff
        total_loss.backward()
        optimizer.step()
        predictions1 = torch.max(outputs1, 1)
        predictions2 = torch.max(outputs2, 1)
        
        train_correct1 += np.sum(predictions1[1].cpu().numpy() == labels1.cpu().numpy())
        total1 += labels1.size(0)

        train_correct2 += np.sum(predictions2[1].cpu().numpy() == labels2.cpu().numpy())
        total2 += labels2.size(0)
        # print statistics
        running_loss += total_loss.item()
        ls += loss_sup.item()
        lc += loss_cot.item()
        ld += loss_diff.item()
        if i % 40 == 39:    # print every 200 mini-batches
            print('[%d, %5d] total loss: %.3f, loss_sup:%.3f, loss_cot:%.3f, loss_diff:%.3f, training acc 1: %.3f, training acc 2: %.3f' %
                  (epoch + 1, i + 1, running_loss / 40, ls, lc, ld, 100. * train_correct1 / total1, 100. * train_correct2 / total2))
            running_loss = 0.0
            ls = 0.0
            lc = 0.0 
            ld = 0.0
            total1 = 0
            total2 = 0
            train_correct1 = 0
            train_correct2 = 0

        i = i + 1
print('Finished Training')

torch.save(net1.state_dict(), 'co_train_classifier_1.pkl')
torch.save(net2.state_dict(), 'co_train_classifier_2.pkl')

print("Saved!")
