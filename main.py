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
from tensorboardX import SummaryWriter 
from random import shuffle
import pickle
import argparse
from tqdm import tqdm

from model import co_train_classifier



parser = argparse.ArgumentParser(description='Deep Co-Training for Semi-Supervised Image Recognition')
parser.add_argument('--sess', default='default', type=str, help='session id')
parser.add_argument('--batchsize', '-b', default=100, type=int)
parser.add_argument('--lambda_cot_max', default=10, type=int)
parser.add_argument('--lambda_diff_max', default=0.5, type=float)
parser.add_argument('--seed', default=1234, type=int)
parser.add_argument('--epochs', default=600, type=int)
parser.add_argument('--warm_up', default=80, type=int)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--decay', default=1e-4, type=float)
parser.add_argument('--epsilon', default=0.02, type=float)
parser.add_argument('--num_class', default=10, type=int)
parser.add_argument('--cifar10_dir', default='./data', type=str)
parser.add_argument('--tensorboard_dir', default='tensorboard/', type=str)
parser.add_argument('--checkpoint_dir', default='checkpoint', type=str)
parser.add_argument('--base_lr', default=0.05, type=float)
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


# main
# for reproducibility
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)


if not os.path.isdir(args.tensorboard_dir):
        os.mkdir(args.tensorboard_dir)

writer = SummaryWriter(args.tensorboard_dir)
start_epoch = 0
end_epoch = args.epochs
class_num = args.num_class 
batch_size = args.batchsize
U_batch_size = int(batch_size * 46./50.) # note that the ratio of labelled/unlabelled data need to be equal to 4000/46000
S_batch_size = batch_size - U_batch_size
lambda_cot_max = args.lambda_cot_max
lambda_diff_max = args.lambda_diff_max
lambda_cot = 0
lambda_diff = 0
best_acc = 0.0  

def adjust_learning_rate(optimizer, epoch):
    """cosine scheduling"""
    epoch = epoch + 1
    lr = args.base_lr*(1.0 + math.cos((epoch-1)*math.pi/args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_lamda(epoch):
    epoch = epoch + 1
    global lambda_cot
    global lambda_diff
    if epoch <= args.warm_up:
        lambda_cot = lambda_cot_max*math.exp(-5*((1-epoch)/args.warm_up)**2)
        lambda_diff = lambda_diff_max*math.exp(-5*((1-epoch)/args.warm_up)**2)
    else: 
        lambda_cot = lambda_cot_max
        lambda_diff = lambda_diff_max    

def jsd(p,q):
    kld = nn.KLDivLoss(reduction='batchmean')
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    a = S(p)
    b = S(q)
    c = LS(0.5*(p + q))
    return ((0.5*kld(c,a) + 0.5*kld(c, b)))

def loss_sup(logit1, logit2, labels_S1, labels_S2):
    # CE, by default, is averaged over each loss element in the batch
    ce = nn.CrossEntropyLoss() 
    loss1 = ce(logit1, labels_S1)
    loss2 = ce(logit2, labels_S2) 
    return loss1+loss2

def loss_cot(logit1, logit2):
# the Jensen-Shannon divergence between p1(x) and p2(x)
    return jsd(logit1, logit2)

def loss_diff(logit1, logit2, perturbed_logit1, perturbed_logit2, U_logit1, U_logit2, perturbed_logit_U1, perturbed_logit_U2):
    S = nn.Softmax(dim = 1)
    LS = nn.LogSoftmax(dim = 1)
    
    a = S(logit2) * LS(perturbed_logit1)
    a = torch.sum(a)

    b = S(logit1) * LS(perturbed_logit2)
    b = torch.sum(b)

    c = S(U_logit2) * LS(perturbed_logit_U1)
    c = torch.sum(c)

    d = S(U_logit1) * LS(perturbed_logit_U2)
    d = torch.sum(d)

    return -(a+b+c+d)/batch_size


def get_adv_example(net, inputs, labels, optimizer):
    net.eval()
    inputs.requires_grad_()
    optimizer.zero_grad()
    net.zero_grad()
    ce = nn.CrossEntropyLoss()
    outputs = net(inputs)
    loss = ce(outputs,labels)
    loss.backward()
    epsilon = args.epsilon
    x_grad = torch.sign(inputs.grad)
    x_adversarial = inputs.detach()+epsilon*x_grad
    net.train()
    return x_adversarial

# labelled data propotion 4000/50000 for cifar 10 
# unlabelled data propotion 46000/50000 for cifar 10 
# standard data augmentation on cifar10
transform_train = transforms.Compose([
    transforms.RandomAffine(0, translate=(1/16,1/16)), # translation at most two pixels
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

testset = torchvision.datasets.CIFAR10(root=args.cifar10_dir, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

trainset = torchvision.datasets.CIFAR10(root=args.cifar10_dir, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)

S_idx = []
U_idx = []
dataiter = iter(trainloader)
train = [[],[],[],[],[],[],[],[],[],[]]

for i in range(len(trainset)):
    inputs, labels = dataiter.next()
    train[labels].append(i)

for i in range(class_num):
    shuffle(train[i])
    S_idx = S_idx + train[i][0:400]
    U_idx = U_idx + train[i][400:]

#save the indexes in case we need the exact ones for comparison
with open("labelled_index.txt","wb") as fp:
    pickle.dump(S_idx,fp)

with open("unlabelled_index.txt","wb") as fp:
    pickle.dump(U_idx,fp)

S_sampler = torch.utils.data.SubsetRandomSampler(S_idx)
U_sampler = torch.utils.data.SubsetRandomSampler(U_idx)



S_loader1 = torch.utils.data.DataLoader(
        trainset, batch_size=S_batch_size, sampler=S_sampler,
        num_workers=2, pin_memory=True)

S_loader2 = torch.utils.data.DataLoader(
        trainset, batch_size=S_batch_size, sampler=S_sampler,
        num_workers=2, pin_memory=True)

U_loader = torch.utils.data.DataLoader(
        trainset, batch_size=U_batch_size, sampler=U_sampler,
        num_workers=2, pin_memory=True)


step = int(len(trainset)/batch_size)

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./'+ args.checkpoint_dir + '/ckpt.t7.' +
                            args.sess + '_' + str(args.seed))
    net1 = checkpoint['net1']
    net2 = checkpoint['net2']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    print('Building model..')
    start_epoch = 0
    net1 = co_train_classifier()
    net2 = co_train_classifier()

net1.cuda()
net2.cuda()
params = list(net1.parameters()) + list(net2.parameters())
optimizer = optim.SGD(params, lr=args.base_lr, momentum=args.momentum, weight_decay=args.decay)
ce = nn.CrossEntropyLoss() 

def checkpoint(epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net1': net1,
        'net2': net2,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    torch.save(state, './'+ args.checkpoint_dir +'/ckpt.t7.' +
               args.sess + '_' + str(args.seed))


def train(epoch):
    net1.train()
    net2.train()

    adjust_learning_rate(optimizer, epoch)
    adjust_lamda(epoch)
    
    total_S1 = 0
    total_S2 = 0
    total_U1 = 0
    total_U2 = 0
    train_correct_S1 = 0
    train_correct_S2 = 0
    train_correct_U1 = 0
    train_correct_U2 = 0
    running_loss = 0.0
    ls = 0.0
    lc = 0.0 
    ld = 0.0
    
    # create iterator for b1, b2, bu
    S_iter1 = iter(S_loader1)
    S_iter2 = iter(S_loader2)
    U_iter = iter(U_loader)
    print('epoch:', epoch+1)
    for i in tqdm(range(step)):
        inputs_S1, labels_S1 = S_iter1.next()
        inputs_S2, labels_S2 = S_iter2.next()
        inputs_U, labels_U = U_iter.next() # note that labels_U will not be used for training. 

        inputs_S1 = inputs_S1.cuda()
        labels_S1 = labels_S1.cuda()
        inputs_S2 = inputs_S2.cuda()
        labels_S2 = labels_S2.cuda()
        inputs_U = inputs_U.cuda()    

        perturbed_data1 = get_adv_example(net1, inputs_S1, labels_S1, optimizer)
        perturbed_data2 = get_adv_example(net2, inputs_S2, labels_S2, optimizer)


        perturbed_logit1 = net1(perturbed_data2)
        perturbed_logit2 = net2(perturbed_data1)
       
        S_logit1 = net1(inputs_S1)
        S_logit2 = net2(inputs_S2)
        U_logit1 = net1(inputs_U)
        U_logit2 = net2(inputs_U)

        predictions_S1 = torch.max(S_logit1, 1)
        predictions_S2 = torch.max(S_logit2, 1)
        predictions_U1 = torch.max(U_logit1, 1)
        predictions_U2 = torch.max(U_logit2, 1)
        
        
        perturbed_data_U1 = get_adv_example(net1, inputs_U, predictions_U1[1], optimizer)
        perturbed_data_U2 = get_adv_example(net2, inputs_U, predictions_U2[1], optimizer)


        perturbed_logit_U1 = net1(perturbed_data_U2)
        perturbed_logit_U2 = net2(perturbed_data_U1)

        # zero the parameter gradients
        optimizer.zero_grad()
        net1.zero_grad()
        net2.zero_grad()

        
        Loss_sup = loss_sup(S_logit1, S_logit2, labels_S1, labels_S2)
        Loss_cot = loss_cot(U_logit1, U_logit2)
        Loss_diff = loss_diff(S_logit1, S_logit2, perturbed_logit1, perturbed_logit2, U_logit1, U_logit2, perturbed_logit_U1, perturbed_logit_U2)
        
        total_loss = Loss_sup + lambda_cot*Loss_cot + lambda_diff*Loss_diff
        total_loss.backward()
        optimizer.step()


        train_correct_S1 += np.sum(predictions_S1[1].cpu().numpy() == labels_S1.cpu().numpy())
        total_S1 += labels_S1.size(0)

        train_correct_U1 += np.sum(predictions_U1[1].cpu().numpy() == labels_U.cpu().numpy())
        total_U1 += labels_U.size(0)

        train_correct_S2 += np.sum(predictions_S2[1].cpu().numpy() == labels_S2.cpu().numpy())
        total_S2 += labels_S2.size(0)

        train_correct_U2 += np.sum(predictions_U2[1].cpu().numpy() == labels_U.cpu().numpy())
        total_U2 += labels_U.size(0)
        
        running_loss += total_loss.item()
        ls += Loss_sup.item()
        lc += Loss_cot.item()
        ld += Loss_diff.item()
        # print statistics
        
        writer.add_scalars('data/loss', {'loss_sup': Loss_sup.item(), 'loss_cot': Loss_cot.item(), 'loss_diff': Loss_diff.item()}, (epoch)*(500)+i)
        writer.add_scalars('data/training_accuracy', {'net1 acc': 100. * (train_correct_S1) / (total_S1), 'net2 acc': 100. * (train_correct_S2) / (total_S2)}, (epoch)*(500)+i)
        if (i+1)%50 == 0:
            tqdm.write('net1 training acc: %.3f%% | net2 training acc: %.3f%% | total loss: %.3f | loss_sup: %.3f | loss_cot: %.3f | loss_diff: %.3f  '
                % (100. * (train_correct_S1+train_correct_U1) / (total_S1+total_U1), 100. * (train_correct_S2+train_correct_U2) / (total_S2+total_U2), running_loss/(i+1), ls/(i+1), lc/(i+1), ld/(i+1)))


def test(epoch):
    global best_acc
    net1.eval()
    net2.eval()
    correct1 = 0
    correct2 = 0
    total1 = 0
    total2 = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.cuda()
            targets = targets.cuda()

            outputs1 = net1(inputs)
            predicted1 = outputs1.max(1)
            total1 += targets.size(0)
            correct1 += predicted1[1].eq(targets).sum().item()

            outputs2 = net2(inputs)
            predicted2 = outputs2.max(1)
            total2 += targets.size(0)
            correct2 += predicted2[1].eq(targets).sum().item()

    print('\nnet1 test acc: %.3f%% (%d/%d) | net2 test acc: %.3f%% (%d/%d)'
        % (100.*correct1/total1, correct1, total1, 100.*correct2/total2, correct2, total2))
    writer.add_scalars('data/testing_accuracy', {'net1 acc': 100.*correct1/total1, 'net2 acc': 100.*correct2/total2}, epoch)

    acc = ((100.*correct1/total1)+(100.*correct2/total2))/2
    if acc > best_acc:
        best_acc = acc
        checkpoint(epoch)

for epoch in range(start_epoch, end_epoch):
    train(epoch)
    test(epoch)

writer.export_scalars_to_json('./'+ args.tensorboard_dir + 'output.json')
writer.close()
