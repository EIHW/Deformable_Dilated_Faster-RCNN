import os
import zipfile

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.deep_lesion_dataset import CsvHeaders


def extract_dataset(data_dir, csv_file, output_dir):
	"""
	Extract the relevant images from the complete DeepLesion dataset for faster processing.
	:param data_dir: The directory of the zip files.
	:param csv_file: The csv file embodying the annotations.
	:param output_dir: The directory to output the extracted subset.
	"""
	# check if output directory exists
	os.makedirs(output_dir, exist_ok=True)
	# Variables
	new_csv_file = os.path.join(output_dir, os.path.basename(csv_file))
	original_df = pd.read_csv(csv_file)
	subset_df = pd.DataFrame()
	# Iterate through all zip_file files
	t = tqdm([file for file in os.listdir(data_dir) if file.endswith('.zip')], desc='Extraction')
	log = {}
	for zip_folder in t:
		log.update({'zip_file': zip_folder})
		t.set_postfix(log)
		zip_path = os.path.join(data_dir, zip_folder)
		try:
			with zipfile.ZipFile(zip_path) as zip_file:
				for folder in [folder for folder in zip_file.namelist() if not folder.endswith('.png')]:
					# Extract information from folder name
					folder_name = folder.split('/')[1]
					folder_name_parts = folder_name.split('_')
					local_df = original_df[original_df[CsvHeaders.PATIENT_INDEX.value] == int(folder_name_parts[0])]
					local_df = local_df[local_df[CsvHeaders.STUDY_INDEX.value] == int(folder_name_parts[1])]
					local_df = local_df[local_df[CsvHeaders.SERIES_ID.value] == int(folder_name_parts[2])]
					assert len(local_df.index) > 0, 'No entry found for {} in {}'.format(folder_name, zip_folder)
					# Extract needed slice images
					for key_slice in np.unique(local_df[CsvHeaders.KEY_SLICE_INDEX.value].values):
						slice_df = local_df[local_df[CsvHeaders.KEY_SLICE_INDEX.value] == key_slice]
						slices = [key_slice - 1, key_slice, key_slice + 1]
						files = [os.path.join(folder, '{:03d}.png'.format(k)) for k in slices]
						files = [f for f in files if f in zip_file.namelist()]
						zip_file.extractall(members=files, path=output_dir)
						# Merge rows to one row
						item_df = pd.DataFrame()
						for header in slice_df.keys():
							unique_value = np.unique(slice_df[header].values)
							if len(unique_value) > 1:
								item_df[header] = ';'.join([str(v) for v in unique_value])
							else:
								item_df[header] = unique_value
						assert len(item_df.index) == 1, 'The item_df length is not allowed to be greater 1'
						subset_df = pd.concat([subset_df, item_df])
					log.update({'extracted': len(subset_df.index)})
		except zipfile.BadZipFile as e:
			# In case the zip_file file is unable to be read
			log.update({'exception': 'Zip={}, Exception={}'.format(zip_folder, e)})
	subset_df.to_csv(new_csv_file)
	print('Saved in csv: {}'.format(new_csv_file))
