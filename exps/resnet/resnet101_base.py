import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = '/mnt/truenas/scratch/hzh_hb3/datasets/coco/coco' #'/mnt/weka/scratch/datasets/coco'
        self.max_epoch = 200
        self.no_aug_epochs = 10
        #
        self.basic_lr_per_img = 0.01 / 72.0
    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [512, 1024, 2048]
        out_channels = [256, 512, 1024]
        from yolox.models import YOLOX, YOLOPAFPN_ResNet, YOLOXHead
        backbone = YOLOPAFPN_ResNet(in_channels=in_channels, out_channels=out_channels, act=self.act,resnet_depth=101)
        head = YOLOXHead(self.num_classes, self.width, in_channels=out_channels, act=self.act)
        self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model
