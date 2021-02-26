from enum import Enum

from torchvision.models.detection.faster_rcnn import AnchorGenerator, FastRCNNPredictor, FasterRCNN
from torchvision.ops import MultiScaleRoIAlign

from src.models.detection.roi_pool import MultiScaleDeformPSRoIAlign
from .vgg_16 import VGG16, VGG16DeformConv


class FasterRCNNType(Enum):
	BASIC = 1
	DEFORMABLE_CONV = 2
	DEFORMABLE_ROI = 3
	DEFORMABLE_CONV_ROI = 4

	@staticmethod
	def get_name(type):
		"""
		Get the name of the requested model type.
		:param type: FasterRCNNType. The type of the model.
		:return: The name of the model type or raise an exception if unknown.
		"""
		if isinstance(type, FasterRCNNType):
			type = type.value
		if FasterRCNNType.BASIC.value == type:
			return 'Basic'
		if FasterRCNNType.DEFORMABLE_CONV.value == type:
			return 'DeformConv'
		if FasterRCNNType.DEFORMABLE_ROI.value == type:
			return 'DeformRoI'
		if FasterRCNNType.DEFORMABLE_CONV_ROI.value == type:
			return 'DeformConvRoI'
		raise Exception('Unknown type "{}"'.format(type))

	@staticmethod
	def get_type(name):
		"""
		Get the type of the requested model name.
		:param name: String. The name of the model.
		:return: The type of the model or raise an exception if unknown.
		"""
		if 'Basic' == name:
			return FasterRCNNType.BASIC.value
		if 'DeformConv' == name:
			return FasterRCNNType.DEFORMABLE_CONV.value
		if 'DeformRoI' == name:
			return FasterRCNNType.DEFORMABLE_ROI.value
		if 'DeformConvRoI' == name:
			return FasterRCNNType.DEFORMABLE_CONV_ROI.value
		raise Exception('Unknown type "{}"'.format(type))


def faster_rcnn_model_builder(type=FasterRCNNType.BASIC, dilation=[1, 1, 1]):
	"""
	The faster r-cnn model builder.
	:param type: FasterRCNNType. The type of model.
	:param dilation: Integer. See https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html for further information.
	:return: The complete model.
	"""
	backbone = VGG16(dilation)
	rpn_anchor_generator = AnchorGenerator(sizes=((16, 24, 32, 48, 96),), aspect_ratios=((0.5, 1.0, 2.0),))
	box_roi_pool = MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
	box_predictor = FastRCNNPredictor(512, num_classes=2)
	if type == FasterRCNNType.DEFORMABLE_CONV.value or type == FasterRCNNType.DEFORMABLE_CONV_ROI.value:
		backbone = VGG16DeformConv(dilation)
	if type == FasterRCNNType.DEFORMABLE_ROI.value or type == FasterRCNNType.DEFORMABLE_CONV_ROI.value:
		box_roi_pool = MultiScaleDeformPSRoIAlign(featmap_names=['0'], output_dim=512, sampling_ratio=2)
	return model(backbone, rpn_anchor_generator, box_roi_pool, box_predictor)


def model(basic_layers, rpn_anchor_generator, box_roi_pool, box_predictor):
	"""
	Creates a faster r-cnn model with all the required modules.
	:param basic_layers: The backbone (e.g. VGG16).
	:param box_roi_pool: The roi pool must be a (sub-)type of MultiScaleRoIAlign.
	:return: The faster r-cnn model.
	"""
	return FasterRCNN(
			backbone=basic_layers.backbone,
			min_size=512, max_size=512,
			rpn_anchor_generator=rpn_anchor_generator,
			rpn_batch_size_per_image=32,
			box_roi_pool=box_roi_pool,
			box_head=basic_layers.classifier,
			box_predictor=box_predictor,
			box_detections_per_img=32
	)
