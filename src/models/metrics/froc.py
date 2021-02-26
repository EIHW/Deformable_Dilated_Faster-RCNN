import numpy as np
import torch
from ignite.metrics import Metric
from torchvision.ops import box_iou


class FROC(Metric):
    def __init__(self, avg_fps, iou_threshold):
        """
        Initializes a new FROC curve which will interpolate the values at all the avg_fps provided and uses the
        IoU-threshold for determining whether or not the boxes overlap.
        :param avg_fps: array[Float]. The average false positives to interpolate at.
        :param iou_threshold: Float. The threshold at which the boxes will be determined as overlap or not.
        """
        super(FROC, self).__init__(output_transform=lambda x: x)
        self.avg_fps = avg_fps
        self.iou_th = iou_threshold
        self.llf_list = []
        self.nll_list = []
        self.gt_list = []

    def reset(self):
        """
        Resets the variables for the calculation to restart.
        """
        self.llf_list = []
        self.nll_list = []
        self.gt_list = []

    def update(self, output):
        """
        Updates the algorithm with the next batch of images.
        :param output: (Prediction, Ground-Truth). Provides the prediction output and ground truth data.
        """
        pred, gt = output
        boxes_all = [x['boxes'] for x in pred]  # [NxMx4] sorted highest score to lowest score
        scores_all = [x['scores'] for x in pred]  # [NxMx4] sorted highest score to lowest score
        gts_all = [x['boxes'] for x in gt]  # [Nx1x4]
        patient_all = [x['patient_index'] for x in gt]  # [Nx1]
        batch_size = len(boxes_all)
        for batch_idx in range(batch_size):
            for gt_box in gts_all[batch_idx]:
                overlaps = box_iou(boxes_all[batch_idx], torch.stack([gt_box]))  # [Mx1]
                overlaps = overlaps.squeeze(dim=1)  # [M]
                lesion_localization_mask = overlaps >= self.iou_th  # Mask of all true positives
                lesion_localization_scores = scores_all[batch_idx][lesion_localization_mask]  # True positives - Scores
                lesion_localization = lesion_localization_scores.detach().cpu().numpy().tolist()
                non_lesion_localization_mask = overlaps < self.iou_th
                non_lesion_localization = scores_all[batch_idx][
                    non_lesion_localization_mask]  # False positives - Scores
                non_lesion_localization = non_lesion_localization.detach().cpu().numpy().tolist()
                self.llf_list.append(lesion_localization)
                self.nll_list.append(non_lesion_localization)
            self.gt_list.append(len(gts_all[batch_idx]))

    def compute(self):
        """
        Computes the algorithms output.
        :return: array[Float]. The interpolated output over the given false positives.
        """
        number_of_images = len(self.gt_list)
        total_gt = sum(self.gt_list)
        total_sens = []
        total_fps = []
        for threshold in np.arange(0.0, 1.0, 0.01):
            fps, tps = [], []
            for img_idx in range(number_of_images):
                fps.append((np.asarray(self.nll_list[img_idx]) >= threshold).sum())
                tps.append((np.asarray(self.llf_list[img_idx]) >= threshold).sum())
            fps.append(0)
            tps.append(0)
            fps = np.asarray(fps).sum() / float(number_of_images)
            total_fps.append(fps)
            sensitivity = np.asarray(tps).sum() / float(total_gt)
            total_sens.append(sensitivity)
        total_fps = sorted(total_fps)
        total_sens = sorted(total_sens)
        sens_itp = np.interp(self.avg_fps, total_fps, total_sens)
        return torch.from_numpy(sens_itp)
