import torch
import torch.nn as nn
from DenseCRFLoss import DenseCRFLoss


class partial_BCELoss(nn.Module):
    def __init__(self):
        super(partial_BCELoss, self).__init__()

    def forward(self, y_pred, y_true):
        index_true = (torch.zeros(size=y_true.shape)).cuda()
        index_true[y_true != 128] = 1 # pixels labeled 128 are those unlabeled pixels.
        # print("ckck"+str(torch.sum(index_true)))
        y_true = y_true / 255.0
        smooth = 0.00001
        bce_loss = -(y_true*torch.log(y_pred + smooth) + (1-y_true)*torch.log(1-y_pred + smooth))
        pCEloss = (torch.sum(torch.mul(bce_loss, index_true)) + smooth) / (torch.sum(index_true) + smooth)
        return pCEloss


class Regularized_Loss(nn.Module):
    def __init__(self, batch=True):
        super(Regularized_Loss, self).__init__()
        self.batch = batch
        self.pCEloss = partial_BCELoss()
        self.densecrflosslayer = DenseCRFLoss(weight=2e-9, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
        self.mse_loss = nn.MSELoss()

    def denormalizeimage(self, images, mean=(0., 0., 0.), std=(1., 1., 1.)):
        """Denormalize tensor images with mean and standard deviation.
        Args:
            images (tensor): N*C*H*W
            mean (tuple): means for each channel.
            std (tuple): standard deviations for each channel.
        """
        images = images.cpu().numpy()
        # N*C*H*W to N*H*W*C
        images = images.transpose((0, 2, 3, 1))
        images *= std
        images += mean
        images *= 255.0
        # N*H*W*C to N*C*H*W
        images = images.transpose((0, 3, 1, 2))
        return torch.tensor(images)

    def boundary_loss(self, edge, hed_true):
        loss = self.mse_loss(edge, hed_true)
        return loss

    def __call__(self, y_pred, y_true, image, edge, hed_true):
        denormalized_image = self.denormalizeimage(image, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        croppings = (y_true != 128).float() # pixels labeled 128 are those unlabeled pixels.
        densecrfloss = self.densecrflosslayer(denormalized_image, y_pred, croppings)
        densecrfloss = densecrfloss.cuda()
        pCEloss = self.pCEloss(y_pred, y_true)
        boundaryloss = self.boundary_loss(edge, hed_true)
        loss = pCEloss + 0.5 * densecrfloss + 0.7 * boundaryloss

        # print("pBCE:"+str(pCEloss)+";denseCRF:"+str(densecrfloss)+";boundary:"+str(boundaryloss))

        return loss
