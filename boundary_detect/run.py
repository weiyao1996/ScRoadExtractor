#!/usr/bin/env python

import torch

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:3]).split('+')[0]) >= 41) # requires at least pytorch version 0.4.1

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		self.moduleVggOne = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggTwo = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggThr = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggFou = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleVggFiv = torch.nn.Sequential(
			torch.nn.MaxPool2d(kernel_size=2, stride=2),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False),
			torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			torch.nn.ReLU(inplace=False)
		)

		self.moduleScoreOne = torch.nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreTwo = torch.nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreThr = torch.nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreFou = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)
		self.moduleScoreFiv = torch.nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1, padding=0)

		self.moduleCombine = torch.nn.Sequential(
			torch.nn.Conv2d(in_channels=5, out_channels=1, kernel_size=1, stride=1, padding=0),
			torch.nn.Sigmoid()
		)

		self.load_state_dict(torch.load('./network-bsds500.pytorch'))
	# end

	def forward(self, tensorInput):
		tensorBlue = (tensorInput[:, 0:1, :, :] * 255.0) - 104.00698793
		tensorGreen = (tensorInput[:, 1:2, :, :] * 255.0) - 116.66876762
		tensorRed = (tensorInput[:, 2:3, :, :] * 255.0) - 122.67891434

		tensorInput = torch.cat([ tensorBlue, tensorGreen, tensorRed ], 1)

		tensorVggOne = self.moduleVggOne(tensorInput)
		tensorVggTwo = self.moduleVggTwo(tensorVggOne)
		tensorVggThr = self.moduleVggThr(tensorVggTwo)
		tensorVggFou = self.moduleVggFou(tensorVggThr)
		tensorVggFiv = self.moduleVggFiv(tensorVggFou)

		tensorScoreOne = self.moduleScoreOne(tensorVggOne)
		tensorScoreTwo = self.moduleScoreTwo(tensorVggTwo)
		tensorScoreThr = self.moduleScoreThr(tensorVggThr)
		tensorScoreFou = self.moduleScoreFou(tensorVggFou)
		tensorScoreFiv = self.moduleScoreFiv(tensorVggFiv)

		tensorScoreOne = torch.nn.functional.interpolate(input=tensorScoreOne, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreTwo = torch.nn.functional.interpolate(input=tensorScoreTwo, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreThr = torch.nn.functional.interpolate(input=tensorScoreThr, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreFou = torch.nn.functional.interpolate(input=tensorScoreFou, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)
		tensorScoreFiv = torch.nn.functional.interpolate(input=tensorScoreFiv, size=(tensorInput.size(2), tensorInput.size(3)), mode='bilinear', align_corners=False)

		return self.moduleCombine(torch.cat([ tensorScoreOne, tensorScoreTwo, tensorScoreThr, tensorScoreFou, tensorScoreFiv ], 1))
	# end
# end

moduleNetwork = Network().cuda().eval()

##########################################################

def estimate(tensorInput):
	intWidth = tensorInput.size(2)
	intHeight = tensorInput.size(1)

	assert(intWidth == 512) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	assert(intHeight == 512) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	return moduleNetwork(tensorInput.cuda().view(1, 3, intHeight, intWidth))[0, :, :, :].cpu()
# end

##########################################################

if __name__ == '__main__':

	img_root = '/data/train/sat/'
	out_root = '/data/train/rough_edge/'
	if os.path.isdir(out_root):
		pass
	else:
		os.makedirs(out_root)

	name_list = os.listdir(img_root)
	for name in name_list:
		region_info = name[:-7]
		arguments_strIn = img_root + region_info + 'sat.png'
		arguments_strOut = out_root + region_info + 'hed.png'

		tensorInput = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strIn))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0))
		tensorOutput = estimate(tensorInput)
		PIL.Image.fromarray((tensorOutput.clamp(0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0).astype(numpy.uint8)).save(arguments_strOut)
# end
