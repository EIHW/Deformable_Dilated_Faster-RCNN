import os
from collections import OrderedDict

import matplotlib.patches as patch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

from src.dataset import DatasetType, DeepLesionDataset
from src.models import FasterRCNNType, faster_rcnn_model_builder
from src.reporter import extract_checkpoint_attrs


def create_plot(name, images, targets, detections, save_path):
	"""
	Visualize the images given with the targets and detections as annotations.
	:param name: The name of the images.
	:param images: The images to visualize.
	:param targets: The targets to annotate.
	:param detections: The detection results to annotate.
	:param save_path: The path where to save the generated plot.
	"""
	for img_idx, img in zip(range(len(images)), images):
		figure, ax = plt.subplots(1)
		ax.set_axis_off()
		gt_boxes = [] if targets is None else targets['boxes']
		for box in gt_boxes:
			rect = patch.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
			                       edgecolor='y', facecolor='none', linewidth=3, linestyle='--')
			ax.add_patch(rect)
		boxes = [] if detections is None else detections['boxes']
		scores = [] if detections is None else detections['scores']
		for idx, box, score in zip(range(len(boxes)), boxes, scores):
			#if score > 0.5:
			overlaps = box_iou(boxes, gt_boxes) if len(gt_boxes) > 0 else torch.zeros(len(boxes))
			color = 'darkgreen' if overlaps[idx] > 0.5 else 'r'
			rect = patch.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1],
								   edgecolor=color, facecolor='none', linewidth=2, label=score)
			ax.text(box[0], box[1] - 2, round(score.item(), 2), color=color, fontsize=12)
			ax.add_patch(rect)
		ax.imshow(img, cmap=None if detections is None and targets is None else 'gray')
		plt.tight_layout()
		path = os.path.join(save_path, '{}_{}.png'.format(name, img_idx))
		plt.savefig(path, aspect='auto', bbox_inches='tight', pad_inches=0)
		plt.close()


def generate_preview(data_dir, csv_file, checkpoint_dir, output_dir):
	"""
	Generates a report for the given dataset on all checkpoints available.
	:param data_dir: String. The path to the data.
	:param csv_file: String. The csv file which describes the dataset.
	:param checkpoint_dir: String. The path to the checkpoints directory.
	:param output_dir: String. The path to the output files.
	"""
	dataset = DeepLesionDataset(data_dir, csv_file, batch_size=1, type=DatasetType.TEST)
	samples = [dataset[i] for i in np.random.randint(len(dataset), size=10)]
	pth_files = os.listdir(checkpoint_dir)
	for attr in [extract_checkpoint_attrs(checkpoint_dir, f) for f in pth_files if f.endswith('8.pth')]:
		pth_file = attr['file']
		name = attr['name']
		dilation = attr['di']
		type = FasterRCNNType.get_type(name)
		base_path = os.path.join(os.path.join(output_dir, 'preview'), 'Type={}_Dilation={}'.format(type, dilation))
		model = faster_rcnn_model_builder(type, dilation)
		model.load_state_dict(torch.load(pth_file, map_location='cpu')['model'])
		model.eval()
		with torch.no_grad():
			sample_idx = 1
			for sample in tqdm(samples, desc='Generating previews for {} with {} Dilation'.format(name, dilation)):
				sample_path = os.path.join(base_path, '{}-Sample'.format(sample_idx))
				sample_idx += 1
				if not os.path.exists(sample_path):
					os.makedirs(sample_path)

				images, targets = sample
				images, targets = model.transform(images, targets)
				base_img = images.tensors[0][1].numpy()
				features = model.backbone(images.tensors)  # 1x512x46x46
				if isinstance(features, torch.Tensor):
					features = OrderedDict([('0', features)])
				model.rpn.nms_thresh = 0.3
				proposals, proposal_losses = model.rpn(images, features, targets)
				box_features = model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
				detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)

				# Create preview of ground-truth image
				create_plot('GT-Image', images.tensors[0], targets[0], None, sample_path)
				# Create preview of FeatureMaps-image
				feature_selection = features['0'][0]  # 512x46x46
				random_features = np.random.choice(range(len(feature_selection)), size=10)
				feature_selection = feature_selection[random_features]  # 10x46x46
				create_plot('Backbone-Feature-Image', feature_selection, None, None, sample_path)
				# Create preview of RPN-image
				det_tmp = {'boxes': proposals[0], 'scores': np.ones(len(proposals[0]))}
				create_plot('RPN-Image', [base_img], None, det_tmp, sample_path)
				# Create preview of RoI Feature Maps
				feature_selection = box_features[0]
				random_features = np.random.choice(range(len(feature_selection)), size=10)
				feature_selection = feature_selection[random_features]
				create_plot('RoI-Feature-Image', feature_selection, None, None, sample_path)
				# Create preview of Detection-image
				create_plot('RoI-Detections-Image', [base_img], targets[0], detections[0], sample_path)
		# Clean up CUDA cache
		del model
		torch.cuda.empty_cache()
