import torch
import torchvision
from torchvision.models.detection.faster_rcnn import MultiScaleRoIAlign
from torchvision.ops.poolers import _onnx_merge_levels

from src.models.ops.deform_v2 import DCNPooling


class MultiScaleDeformPSRoIAlign(MultiScaleRoIAlign):
	def __init__(self, featmap_names, output_dim, sampling_ratio):
		"""
		Multi-scale deformable PSRoIAlign pooling, which is useful for detection with or without FPN.
		It infers the scale of the pooling via the heuristics present in the FPN paper.
		:param featmap_names: List[str]. the names of the feature maps that will be used for the pooling.
		:param output_dim: int. Output size for the pooled region.
		:param sampling_ratio: Integer. Sampling ratio for ROIAlign.
		"""
		super(MultiScaleDeformPSRoIAlign, self).__init__(featmap_names, output_dim, sampling_ratio)
		self.pool = DCNPooling(spatial_scale=1, pooled_size=7, output_dim=output_dim, no_trans=False)

	def forward(self, x, boxes, image_shapes):
		"""
		See torchvision.models.detection.faster_rcnn.MultiScaleRoIAlign.forward(...) for documentation
		"""
		x_filtered = []
		for k, v in x.items():
			if k in self.featmap_names:
				x_filtered.append(v)
		num_levels = len(x_filtered)
		rois = self.convert_to_roi_format(boxes)
		if self.scales is None:
			self.setup_scales(x_filtered, image_shapes)

		scales = self.scales
		assert scales is not None

		if num_levels == 1:
			self.pool.spatial_scale = scales[0]
			return self.pool(x_filtered[0], rois)

		mapper = self.map_levels
		assert mapper is not None

		levels = mapper(boxes)

		num_rois = len(rois)
		num_channels = x_filtered[0].shape[1]

		dtype, device = x_filtered[0].dtype, x_filtered[0].device
		result = torch.zeros(
				(num_rois, num_channels,) + self.output_size,
				dtype=dtype,
				device=device,
		)

		tracing_results = []
		for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
			idx_in_level = torch.nonzero(levels == level).squeeze(1)
			rois_per_level = rois[idx_in_level]

			self.pool.spatial_scale = scale
			result_idx_in_level = self.pool(per_level_feature, rois_per_level)

			if torchvision._is_tracing():
				tracing_results.append(result_idx_in_level.to(dtype))
			else:
				result[idx_in_level] = result_idx_in_level

		if torchvision._is_tracing():
			result = _onnx_merge_levels(levels, tracing_results)

		return result
