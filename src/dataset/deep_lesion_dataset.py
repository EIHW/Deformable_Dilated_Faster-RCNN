import os
from enum import Enum

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image(img_path, key_slice_index):
	"""
	Load the image defined by the path and slice index as well as the image with a slice index greater and lower.
	:param img_path: The path of the image.
	:param key_slice_index: The number of the image slice.
	:return: A 3-dimensional image.
	"""
	# Read 3 16-bit-images to stack them together to a semi 3D image
	key_slices = [key_slice_index - 1, key_slice_index, key_slice_index + 1]
	img_paths = [os.path.join(img_path, '{:03d}.png'.format(idx)) for idx in key_slices]
	img0 = np.array(Image.open(img_paths[0] if os.path.exists(img_paths[0]) else img_paths[1]))
	img1 = np.array(Image.open(img_paths[1]))
	img2 = np.array(Image.open(img_paths[2] if os.path.exists(img_paths[2]) else img_paths[1]))
	img = np.zeros((3, img1.shape[0], img1.shape[1]), dtype=np.float)
	img[0] = img0
	img[1] = img1
	img[2] = img2
	# Obtain the original Hounsfield unit (HU)
	img = img.astype(np.float, copy=False) - 32768
	# Use single windowing (-1024 to 3071 HU) that covers intensity ranges of lung, soft tissue, and bone
	single_windowing = [-1024, 3071]
	img -= np.min(single_windowing)
	img /= np.max(single_windowing) - np.min(single_windowing)
	# Update boundaries to [0, 1]
	img = np.clip(img, 0, 1)
	# Transpose image from WxHxC to CxWxH
	# img = img.transpose((2, 0, 1))
	return img


def splitter(items, separator, convert_type):
	"""
	Split the string of items into a list of items.
	:param items: A string of joined items.
	:param separator: The separator used for separation.
	:param convert_type: The type the item should be converted into after separation.
	:return: A list of items.
	"""
	return [convert_type(x) for x in items.split(separator)]


class CsvHeaders(Enum):
	FILE_NAME = 'File_name'
	PATIENT_INDEX = 'Patient_index'
	STUDY_INDEX = 'Study_index'
	SERIES_ID = 'Series_ID'
	KEY_SLICE_INDEX = 'Key_slice_index'
	MEASUREMENT_COORDINATES = 'Measurement_coordinates'
	BOUNDING_BOXES = 'Bounding_boxes'
	LESION_DIAMETERS_PIXEL = 'Lesion_diameters_Pixel_'
	NORMALIZED_LESION_LOCATION = 'Normalized_lesion_location'
	COARSE_LESION_TYPE = 'Coarse_lesion_type'
	POSSIBLY_NOISY = 'Possibly_noisy'
	SLICE_RANGE = 'Slice_range'
	SPACING_MM_PX = 'Spacing_mm_px_'
	IMAGE_SIZE = 'Image_size'
	DICOM_WINDOWS = 'DICOM_windows'
	PATIENT_GENDER = 'Patient_gender'
	PATIENT_AGE = 'Patient_age'
	TRAIN_VAL_TEST = 'Train_Val_Test'


class DatasetType(Enum):
	TRAIN = 1
	VALIDATION = 2
	TEST = 3


class DeepLesionDataset(Dataset):
	def __init__(self, root, csv_file, batch_size=1, type=DatasetType.TEST):
		"""
		Create a lazy loading DeepLesion dataset.
		:param root: The root path to the images of the dataset.
		:param csv_file: The path to the csv file containing the annotations.
		:param batch_size: The batch size.
		:param type: The type of the dataset split.
		"""
		self.root = root
		self.batch_size = batch_size
		self.df = pd.read_csv(csv_file)
		self.df = self.df[self.df[CsvHeaders.TRAIN_VAL_TEST.value] == type.value]
		self.df.drop_duplicates(keep=False, inplace=True)

	def __len__(self):
		return int(len(self.df.index) / self.batch_size)

	def __getitem__(self, item):
		while item >= len(self):
			item -= len(self)
		item = slice(item * self.batch_size, (item + 1) * self.batch_size)
		result = self.df[item]
		images = []
		target = []
		for idx in range(len(result.index)):
			entry = result.iloc[idx]
			patient_indices = entry[CsvHeaders.PATIENT_INDEX.value]
			study_indices = entry[CsvHeaders.STUDY_INDEX.value]
			series_ids = entry[CsvHeaders.SERIES_ID.value]
			key_slice_indices = entry[CsvHeaders.KEY_SLICE_INDEX.value]
			lesion_types = entry[CsvHeaders.COARSE_LESION_TYPE.value]
			lesion_types = np.asarray([int(lesion) for lesion in splitter(lesion_types, ';', str)])
			lesions = entry[CsvHeaders.BOUNDING_BOXES.value]
			lesions = np.asarray([splitter(box, ',', float) for box in splitter(lesions, ';', str)])
			lesion_area = (lesions[:, 3] - lesions[:, 1]) * (lesions[:, 2] - lesions[:, 0])
			labels = np.ones(len(lesions))
			image = load_image(
					os.path.join(self.root, '{:06d}_{:02d}_{:02d}'.format(patient_indices, study_indices, series_ids)),
					key_slice_indices)
			image = torch.from_numpy(image).float()
			images.append(image)
			target.append({
				'labels': torch.from_numpy(labels).long(),
				'boxes': torch.from_numpy(lesions).float(),
				'area': torch.from_numpy(lesion_area).float(),
				'patient_index': torch.from_numpy(np.asarray(patient_indices)).int(),
				'study_index': torch.from_numpy(np.asarray(study_indices)).int(),
				'series_id': torch.from_numpy(np.asarray(series_ids)).int(),
				'key_slice_index': torch.from_numpy(np.asarray(key_slice_indices)).int(),
				'lesion_type': torch.from_numpy(lesion_types).int(),
			})
		return images, target
