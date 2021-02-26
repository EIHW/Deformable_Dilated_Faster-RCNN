import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib as tikzplt
import torch
from ignite.utils import convert_tensor
from pandas import DataFrame, read_csv
from tqdm import tqdm

from src.dataset import DatasetType, DeepLesionDataset
from src.models import FasterRCNNType, faster_rcnn_model_builder
from src.models.metrics import FROC


def generate_report(data_dir, csv_file, checkpoint_dir, output_dir, avg_fps):
	"""
	Generates a report for the given dataset on all checkpoints available.
	:param data_dir: String. The path to the data.
	:param csv_file: String. The csv file which describes the dataset.
	:param checkpoint_dir: String. The path to the checkpoints directory.
	:param output_dir: String. The path to the output files.
	:param avg_fps: List[Float]. The x-value for the false positives to interpolate on.
	"""
	if os.path.exists(checkpoint_dir):
		# Predict on complete dataset for all models
		dataset = DeepLesionDataset(data_dir, csv_file, batch_size=1, type=DatasetType.TEST)
		pth_files = os.listdir(checkpoint_dir)
		result_files = []
		for attr in [extract_checkpoint_attrs(checkpoint_dir, f) for f in pth_files if f.endswith('8.pth')]:
			pth_file = attr['file']
			name = attr['name']
			dilation = attr['di']
			type = FasterRCNNType.get_type(name)
			result_file = predict_on_model(
					type=type,
					dilation=dilation,
					dataset=dataset,
					pth_file=pth_file,
					output_dir=output_dir,
					cuda=True)
			result_files.append(result_file)
		# Calculate FROC curve for all models
		df = calculate_froc_values(result_files, avg_fps, output_dir)
		# Generate model grid for all models grouped by type
		plot_model_grid(output_dir, avg_fps, df)


def predict_on_model(type, dilation, pth_file, dataset, output_dir, cuda):
	"""
	Generate predictions for a model type with dilation.
	:param type: Integer. The type of model.
	:param dilation: List[Integer]. The dilation to use.
	:param pth_file: String. The checkpoint file path of the model to load.
	:param dataset: DeepLesionDataset. The dataset to use.
	:param output_dir: String. The output directory to save the results to.
	:param cuda: Bool. Use cuda or not.
	:return: String. A path of a file containing the prediction results.
	"""
	title = 'model={}-dilation={}'.format(type, sum(dilation))
	print('Predict on', title)
	result = []
	if not has_model_dump(output_dir, title):
		model = faster_rcnn_model_builder(type, dilation)
		model.load_state_dict(torch.load(pth_file)['model'])
		if cuda and torch.cuda.is_available():
			model.cuda()
		model.eval()
		with torch.no_grad():
			for idx in tqdm(range(len(dataset)), desc=title):
				imgs, target = dataset[idx]
				if cuda:
					imgs = [convert_tensor(image, device='cuda', non_blocking=False) for image in imgs]
					target = convert_tensor(target, device='cuda', non_blocking=False)
				y_pred = model(imgs, target)
				result.append((y_pred, target))
		# Clean up CUDA cache
		del model
		torch.cuda.empty_cache()
		return save_json(title, result, output_dir)
	return os.path.join(output_dir, '{}.json'.format(title))


def calculate_froc_values(files, avg_fps, output_dir):
	lesion_type_names = ['bone', 'abdomen', 'mediastinum', 'liver', 'lung', 'kidney', 'soft tissue', 'pelvis']
	columns = ['model', 'dilation'] + lesion_type_names + ['total', 'fps']
	df_total = DataFrame(columns=columns)
	for file_path in files:
		filename = os.path.basename(file_path)
		base_filename = filename[:-5]
		file_attr = base_filename.split('-')
		model_type = int(file_attr[0].split('=')[1])
		dilation = int(file_attr[1].split('=')[1])
		csv_file = os.path.join(os.path.join(output_dir, 'froc'), '{}-froc.csv'.format(base_filename))
		print('Report for', FasterRCNNType.get_name(model_type), 'with', convert_total_dilation(dilation), 'Dilation')

		if os.path.exists(csv_file):
			content = read_csv(csv_file, header=0)
			df_total = df_total.append(content)
			continue

		pred_result = load_json(output_dir, filename)
		result = {'model': model_type, 'dilation': dilation, 'fps': avg_fps}
		x_values = avg_fps
		# Initialize FROCs for all types
		result.update({key: FROC(avg_fps=x_values, iou_threshold=0.5) for key in ['total'] + lesion_type_names})
		# Update FROCs on predicted dataset
		for values in tqdm(pred_result, desc='FROC for model={} with dilation={}'.format(model_type, dilation)):
			pred, target = values
			lesion_type = target[0]['lesion_type']
			if lesion_type.ndim == 0:
				result[lesion_type_names[lesion_type - 1]].update(values)
			else:
				for l_type in lesion_type:
					if l_type > 0:
						result[lesion_type_names[l_type - 1]].update(values)
			result['total'].update(values)
		# Calculate FROCs
		result.update({key: result[key].compute() for key in ['total'] + lesion_type_names})
		# save as csv
		df = DataFrame(result)
		df.index.name = 'index'
		if not os.path.exists(os.path.join(output_dir, 'froc')):
			os.makedirs(os.path.join(output_dir, 'froc'))
		df.to_csv(os.path.join(os.path.join(output_dir, 'froc'), '{}-froc.csv'.format(base_filename)))
		df_total.append(df)
	return df_total


def plot_model_grid(directory, avg_fps, df):
	for model_type in FasterRCNNType:
		title = '{}-total'.format(FasterRCNNType.get_name(model_type))
		if not os.path.exists(os.path.join(directory, title)):
			model_type_df = df[df['model'] == model_type.value]
			dilations = np.unique(model_type_df['dilation'].values)
			plt.figure(figsize=(6, 6))
			plt.axis([0, max(avg_fps), 0.0, 1])
			for d in dilations:
				label = 'Dilation-{}'.format(convert_total_dilation(d))
				model_dilation_df = model_type_df[model_type_df['dilation'] == d]
				total_values = model_dilation_df['total'].values
				plt.plot(avg_fps, total_values)
				plt.scatter(avg_fps, total_values, label=label)
			plt.xlabel('Average false positives per image')
			plt.ylabel('Sensitivitiy')
			plt.legend(loc='lower right')
			plt.grid()
			save_tikz(directory, title)
			# plt.show()
			plt.close()


def plot_best_each_fps(avg_fps, df):
	for fps in avg_fps:
		fps_df = df[df['fps'] == fps]
		arg_max = fps_df['total'].idxmax()
		fps_df = fps_df.loc[arg_max]
		print('FPS={} - Model={} and Dilation={} with a total result of: {}'.format(fps, FasterRCNNType.get_name(
				fps_df['model']), fps_df['dilation'], fps_df['total']))


def extract_checkpoint_attrs(dir, checkpoint_file):
	attrs = checkpoint_file[:-4].split('_')
	result = {a.split('-')[0]: a.split('-')[1] for a in attrs[1:5]}
	result['name'] = attrs[0]
	result['file'] = os.path.join(dir, checkpoint_file)
	result['di'] = convert_total_dilation(result['di'])
	return result


def convert_total_dilation(total_dilation):
	total_dilation = int(total_dilation)
	di_0 = total_dilation % 3
	if di_0 > 0:
		di_1 = int(total_dilation / 3)
		di_2 = total_dilation - (di_0 + di_1)
		return [di_0, di_1, di_2]
	else:
		return [int(total_dilation / 3)] * 3


def save_json(title, result, directory, overwrite=False):
	def detach_tensors(dict_list):
		for dict_ in dict_list:
			for key, entry in dict_.items():
				dict_[key] = entry.detach().cpu().numpy().tolist()

	file_name = os.path.join(os.path.join(directory, 'prediction'), '{}.json'.format(title))
	if not os.path.exists(os.path.dirname(file_name)):
		os.makedirs(os.path.dirname(file_name))
	if not os.path.exists(file_name) or overwrite:
		for pred, target in result:
			detach_tensors(pred)
			detach_tensors(target)
		with open(file_name, "w") as write_file:
			json.dump(result, write_file)
	return file_name


def load_json(directory, filename, cuda=False):
	def attach_tensors(dict_list):
		for dict_ in dict_list:
			for key, entry in dict_.items():
				if key == 'boxes' and len(entry) == 0:
					dict_[key] = torch.empty((0, 4), dtype=torch.float64)
				else:
					dict_[key] = torch.from_numpy(np.asarray(entry))
				if cuda:
					dict_[key] = convert_tensor(dict_[key], device='cuda', non_blocking=False)

	with open(os.path.join(os.path.join(directory, 'prediction'), filename), "r") as read_file:
		dump_array = json.load(read_file)
		for pred, target in dump_array:
			attach_tensors(pred)
			attach_tensors(target)
		return dump_array


def has_model_dump(output_dir, title):
	return os.path.exists(os.path.join(os.path.join(output_dir, 'prediction'), '{}.json'.format(title)))


def save_tikz(directory, title):
	tikz_dir = os.path.join(directory, 'tikz')
	if not os.path.exists(tikz_dir):
		os.makedirs(tikz_dir)
	tikzplt.save(os.path.join(tikz_dir, '{}.tex'.format(title)))
