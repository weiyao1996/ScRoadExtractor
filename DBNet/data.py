"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data
from torch.autograd import Variable as V

import cv2
import numpy as np
import os


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask, hed,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))
        hed = cv2.warpPerspective(hed, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask, hed


def randomHorizontalFlip(image, mask, hed, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
        hed = cv2.flip(hed, 1)

    return image, mask, hed


def randomVerticleFlip(image, mask, hed, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
        hed = cv2.flip(hed, 0)

    return image, mask, hed


def randomRotate90(image, mask, hed, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)
        hed = np.rot90(hed)
        # ka

    return image, mask, hed


def default_loader(id, sat_dir, lab_dir):
    img = cv2.imread(os.path.join(sat_dir,'{}_sat.png').format(id))
    mask = cv2.imread(os.path.join(lab_dir+'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    #--------------------------------------------------------
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    #--------------------------------------------------------

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 -1.6
    # mask = np.array(mask).transpose(2,0,1)/255.0
    mask = np.array(mask).transpose(2,0,1)
    # mask[mask>=0.5] = 1
    # mask[mask<=0.5] = 0

    return img, mask


def default_loader111(id, sat_dir, lab_dir, hed_dir):
    img = cv2.imread(os.path.join(sat_dir, '{}_sat.png').format(id))
    mask = cv2.imread(os.path.join(lab_dir+'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    hed = cv2.imread(os.path.join(hed_dir+'{}_hed.png').format(id), cv2.IMREAD_GRAYSCALE)
    #--------------------------------------------------------
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))

    img, mask, hed = randomShiftScaleRotate(img, mask, hed,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask, hed = randomHorizontalFlip(img, mask, hed)
    img, mask, hed = randomVerticleFlip(img, mask, hed)
    img, mask, hed = randomRotate90(img, mask, hed)
    #--------------------------------------------------------

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 -1.6
    # mask = np.array(mask).transpose(2,0,1)/255.0
    mask = np.array(mask).transpose(2,0,1)
    # mask[mask>=0.5] = 1
    # mask[mask<=0.5] = 0
    hed = np.expand_dims(hed, axis=2)
    hed = np.array(hed).transpose(2,0,1)/255.0
    hed[hed > 1] = 1
    hed[hed < 0] = 0

    return img, mask, hed


class ImageFolder(data.Dataset):

    def __init__(self, trainlist, sat_dir, lab_dir, hed_dir):
    # def __init__(self, trainlist, sat_dir, lab_dir):
        self.ids = list(trainlist)
        # self.loader = default_loader
        self.loader111 = default_loader111
        self.sat_dir = sat_dir
        self.lab_dir = lab_dir
        self.hed_dir = hed_dir

    def __getitem__(self, index):
        id = self.ids[index]
        # img, mask = self.loader(id, self.sat_dir, self.lab_dir)
        img, mask, hed = self.loader111(id, self.sat_dir, self.lab_dir, self.hed_dir)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        hed = torch.Tensor(hed)
        # return img, mask
        return img, mask, hed

    def __len__(self):
        return len(self.ids)
