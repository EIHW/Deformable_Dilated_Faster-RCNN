import torch
from torchvision.models import vgg16

from .ops.deform_v2 import DCN


class VGG16(object):
	def __init__(self, dilation=[1, 1, 1], fix_block1_2=True):
		"""
		Initializes a vgg16 network and extracts the backbone and classifier.
		:param dilation: Integer. The dilation to use.
		:param fix_block1_2: Bool. Fix the weights of the layers 0-9.
		"""
		vgg = vgg16(pretrained=True)
		# exclude pool5
		self.backbone = vgg.features[:-1]
		# exclude pool4
		del self.backbone[23]
		# fix weights of Conv1 and Conv2
		for layer in self.backbone[:10]:
			for p in layer.parameters():
				# if pretrained=True => requires_grad=False => weights are fixed
				p.requires_grad = not fix_block1_2
		# set dilation
		self.backbone[23].dilation = dilation[0]
		self.backbone[23].padding = dilation[0]
		self.backbone[25].dilation = dilation[1]
		self.backbone[25].padding = dilation[1]
		self.backbone[27].dilation = dilation[2]
		self.backbone[27].padding = dilation[2]
		# out channels
		self.backbone.out_channels = 512

		self.classifier = list(vgg.classifier._modules.values())[:-1]
		self.classifier = torch.nn.Sequential(*self.classifier)
		# Replace fully-connected layers
		self.classifier[0] = torch.nn.Conv2d(512, 512, (3, 3), stride=1, padding=0)  # Conv6
		self.classifier[3] = torch.nn.Conv2d(512, 512, (5, 5), stride=1, padding=0)  # Conv7


class VGG16DeformConv(VGG16):
	def __init__(self, dilation=[1, 1, 1], fix_block1_2=True):
		super(VGG16DeformConv, self).__init__(dilation, fix_block1_2)
		# Replace default Conv with DeformConv
		self.backbone[23] = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=dilation[0], dilation=dilation[0])
		self.backbone[25] = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=dilation[1], dilation=dilation[1])
		self.backbone[27] = DCN(512, 512, kernel_size=(3, 3), stride=1, padding=dilation[2], dilation=dilation[2])
