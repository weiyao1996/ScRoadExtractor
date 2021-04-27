import torch

import torch.utils.data as data
import os

from time import time

from networks.dinknet import ResNet34_EdgeNet
from framework import MyFrame
from loss import Regularized_Loss
from data import ImageFolder

SHAPE = (512, 512)

sat_dir = '/data/train/sat/'
lab_dir = '/data/train/mask_proposal/'
hed_dir = '/data/train/rough_edge/'
imagelist = os.listdir(lab_dir)
trainlist = map(lambda x: x[:-9], imagelist)
NAME = 'DBNet_0'
BATCHSIZE_PER_CARD = 2
solver = MyFrame(ResNet34_EdgeNet, Regularized_Loss, 2e-4)
batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD

dataset = ImageFolder(trainlist, sat_dir, lab_dir, hed_dir)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=0)

mylog = open('logs/' + NAME + '.log', 'w')
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.

for epoch in range(1, total_epoch + 1):
    data_loader_iter = iter(data_loader)
    train_epoch_loss = 0
    for img, mask, hed in data_loader_iter:
        solver.set_input(img, mask, hed)
        train_loss = solver.optimize()
        train_epoch_loss += train_loss
    train_epoch_loss /= len(data_loader_iter)
    print('********', file=mylog)
    print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
    print('train_loss:', train_epoch_loss, file=mylog)
    print('SHAPE:', SHAPE, file=mylog)
    print('********')
    print('epoch:', epoch, '    time:', int(time() - tic))
    print('train_loss:', train_epoch_loss)
    print('SHAPE:', SHAPE)

    if train_epoch_loss >= train_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        train_epoch_best_loss = train_epoch_loss
        solver.save('weights/' + NAME + '.th')
    if no_optim > 6:
        print('early stop at %d epoch' % epoch, file=mylog)
        print('early stop at %d epoch' % epoch)
        break
    if no_optim > 3:
        if solver.old_lr < 5e-7:
            break
        solver.load('weights/' + NAME + '.th')
        solver.update_lr(5.0, factor=True, mylog=mylog)
    mylog.flush()

print('Finish!', file=mylog)
print('Finish!')
mylog.close()
