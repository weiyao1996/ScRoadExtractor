import torch
import torch.nn as nn
from torch.autograd import Variable as V

import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, loss, lr=2e-4, evalmode = False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=lr)
        self.loss = loss().cuda()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, img_batch, mask_batch=None, hed_batch=None, img_id=None):
        self.img = img_batch
        self.mask = mask_batch
        self.hed = hed_batch
        self.img_id = img_id

    def forward(self, volatile=False):
        self.img = V(self.img.cuda(), volatile=volatile)
        if self.mask is not None:
            self.mask = V(self.mask.cuda(), volatile=volatile)
        if self.hed is not None:
            self.hed = V(self.hed.cuda(), volatile=volatile)

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        # pred = self.net.forward(self.img)
        pred, edge = self.net.forward(self.img)
        # loss = self.loss(self.mask, pred)
        # loss = self.loss(pred, self.mask, self.img)
        loss = self.loss(pred, self.mask, self.img, edge, self.hed)
        loss = loss.cuda()
        loss = loss.type(torch.cuda.FloatTensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr
