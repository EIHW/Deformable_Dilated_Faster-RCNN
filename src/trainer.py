import itertools
import os

import numpy as np
import torch
from ignite.contrib.handlers.tensorboard_logger import OptimizerParamsHandler, OutputHandler, TensorboardLogger
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.utils import convert_tensor

from src.dataset import DatasetType, DeepLesionDataset
from src.models import FasterRCNNType, faster_rcnn_model_builder
from src.models.metrics import FROC


def get_free_gpu():
	"""
	Scan the system for available GPUs' and return the one with the most memory available.
	NOTE: Only available for linux systems!
	:return: Integer. The index of the GPU.
	"""
	os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
	if os.path.exists('tmp'):
		memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
		os.remove('tmp')
		return np.argmax(memory_available)
	return 0


def get_device(model):
	"""
	Extract the device to run on from the model.
	:param model: The model to train.
	:return: String. The name of the device.
	"""
	if next(model.parameters()).is_cuda:
		return 'cuda:{}'.format(torch.cuda.current_device())
	else:
		return 'cpu'


def prepare_batch(batch, device=None, non_blocking=False):
	"""
	Move the batch to the provided device.
	:param batch: The batch to prepare.
	:param device: The device to move to (e.g. cpu or gpu).
	:param non_blocking: Bool. Whether it should be blocking or not.
	:return: The prepared batch.
	"""
	images, target = batch
	return [convert_tensor(image, device=device, non_blocking=non_blocking) for image in images], \
	       convert_tensor(target, device=device, non_blocking=non_blocking)


def create_name(name, epochs, lr, lr_decay_step, dilation, batch_size):
	"""
	Create a name that includes all the given hyper-parameters.
	:param name: The name of the model.
	:param epochs: The amount of epochs to train.
	:param lr: The learning rate to use for training.
	:param lr_decay_step: The amount of steps before the learning rate gets reduced.
	:param dilation: The dilation.
	:param batch_size: The batch size.
	:return: The name.
	"""
	return '{}_ep-{}_lr-{}_de-{}_di-{}_bs-{}'.format(name, epochs, lr, lr_decay_step, sum(dilation), batch_size)


class Trainer(Engine):
	def __init__(self, name, model, log_dir, lr, lr_decay_step, adam=False):
		"""
		Initialize to train the given model.
		:param name: The name of the model to be trained.
		:param model: The model to be trained.
		:param log_dir: String. The log directory of the tensorboard.
		:param lr: Float. The learning rate.
		:param lr_decay_step: Integer. The amount of steps the learning rate decays.
		:param adam: Bool. Whether to use adam optimizer or not.
		"""
		super(Trainer, self).__init__(self.update_model)
		self.model = model
		# tqdm
		ProgressBar(persist=True).attach(self)
		# Optimizer
		params = [p for p in model.parameters() if p.requires_grad]
		if adam:
			self.optimizer = torch.optim.Adam(params, lr=lr)
		else:
			self.optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9)
		# Scheduler
		if lr_decay_step > 0:
			self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay_step, gamma=0.1)
			self.add_event_handler(Events.EPOCH_COMPLETED, lambda e: e.scheduler.step())
		else:
			self.scheduler = None
		# Terminate if nan values found
		self.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
		# Tensorboard logging
		self.tb_logger = TensorboardLogger(log_dir=os.path.join(log_dir, name))
		self.add_event_handler(Events.COMPLETED, lambda x: self.tb_logger.close())
		self.tb_logger.attach(self,
		                      log_handler=OptimizerParamsHandler(self.optimizer),
		                      event_name=Events.EPOCH_COMPLETED)
		self.tb_logger.attach(self,
		                      log_handler=OutputHandler(tag='training', output_transform=lambda x: {
			                      'rpn_box_loss': round(self.state.output['loss_rpn_box_reg'].item(), 4),
			                      'rpn_cls_loss': round(self.state.output['loss_objectness'].item(), 4),
			                      'roi_box_loss': round(self.state.output['loss_box_reg'].item(), 4),
			                      'roi_cls_loss': round(self.state.output['loss_classifier'].item(), 4)
		                      }),
		                      event_name=Events.EPOCH_COMPLETED)
		# Run on GPU (cuda) if available
		if torch.cuda.is_available():
			torch.cuda.set_device(int(get_free_gpu()))
			model.cuda(torch.cuda.current_device())

	@staticmethod
	def update_model(engine, batch):
		"""
		Runs the model on the given data batch and does the backpropagation.
		:param engine: The Trainer engine.
		:param batch: The batch to train on.
		:return: The loss values.
		"""
		engine.model.train()
		engine.model.rpn.nms_thresh = 0.7
		img, target = prepare_batch(batch, device=get_device(engine.model))
		engine.optimizer.zero_grad()
		loss = engine.model(img, target)
		losses = sum(l for l in loss.values())
		losses.backward()
		engine.optimizer.step()
		return loss


class Evaluator(Engine):
	def __init__(self, model, tb_logger):
		"""
		Initialize to evaluate the given model.
		:param model: The model to be evaluated.
		:param tb_logger: The tensorboard to be logged to.
		"""
		super(Evaluator, self).__init__(self.predict_on_batch)
		self.model = model
		# FROC
		avg_fps = list(range(1, 26))
		avg_fps.append(0.5)
		avg_fps.sort()
		tags = ['froc_{}fp'.format(fp) for fp in avg_fps]
		for avg_fp, tag in zip(avg_fps, tags):
			FROC([avg_fp], iou_threshold=0.5).attach(self, tag)
		# tqdm
		ProgressBar(persist=True).attach(self)
		# Tensorboard logging
		tb_logger.attach(self,
		                 log_handler=OutputHandler(tag='validation',
		                                           metric_names=tags,
		                                           global_step_transform=lambda engine, name: engine.state.epoch),
		                 event_name=Events.EPOCH_COMPLETED)

	@staticmethod
	def predict_on_batch(engine, batch):
		"""
		Runs the model on the given data batch.
		:param engine: The Evaluator engine.
		:param batch: The batch to evaluate on.
		:return: The predicted values and the target values.
		"""
		engine.model.eval()
		engine.model.rpn.nms_thresh = 0.3
		with torch.no_grad():
			imgs, target = prepare_batch(batch, device=get_device(engine.model))
			y_pred = engine.model(imgs)
		return y_pred, target

	def run(self, data, max_epochs=None, epoch_length=None, seed=None):
		# BugFix: After first run, the max_epochs have to be incremented or set to this engines epoch count.
		if not (self.state is None):
			self.state.max_epochs += 1
		# Run evaluation
		super(Evaluator, self).run(data, max_epochs, epoch_length, seed)


def as_array(value):
	"""
	Checks whether or not the given value is a list. If not, the value is wrapped in a list.
	:param value: List or Other. The value to wrap in a list if it isn't already one.
	:return: The value as a lit.
	"""
	if not isinstance(value, list):
		return [value]
	return value


def train(model_type, lr, lr_decay_step, epochs, dilation, validate, batch_size, log_dir, data_dir, csv_file, use_adam,
          checkpoint_dir, resume_checkpoint):
	"""
	Train the model with the given parameters.
	:param model_type: The type of the model to train.
	:param lr: Float or Array[Float]. The learning rate.
	:param lr_decay_step: Integer or Array[Integer]. The amount of steps the learning rate decays.
	:param epochs: Integer or Array[Integer]. The amount of epochs.
	:param dilation: Integer or Array[Integer]. See https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html for further information.
	:param validate: Bool. Use a validation step after each epoch.
	:param batch_size: Integer or Array[Integer]. The batch size.
	:param log_dir: String. The log directory of the tensorboard.
	:param data_dir: String. The path to the data.
	:param csv_file: String. The csv file which describes the dataset.
	:param use_adam: Bool. Whether to use adam optimizer or not.
	:param checkpoint_dir: String. The path to the checkpoints directory.
	:param resume_checkpoint: String. If None start all over, otherwise start from given checkpoint name.
	"""
	# Datasets
	train_dataset = DeepLesionDataset(data_dir, csv_file, batch_size=batch_size, type=DatasetType.TRAIN)
	validation_dataset = DeepLesionDataset(data_dir, csv_file, batch_size=batch_size, type=DatasetType.VALIDATION)
	# Create combinations of hyper-parameters
	train_variations = itertools.product(*[
		as_array(model_type),
		as_array(epochs),
		as_array(lr),
		as_array(lr_decay_step),
		as_array(dilation),
		as_array(batch_size)
	])
	# Train for all combinations
	for h_type, h_epochs, h_lr, h_lr_decay_step, h_dilation, h_batch_size in train_variations:
		name = FasterRCNNType.get_name(h_type)
		title = create_name(name, h_epochs, h_lr, h_lr_decay_step, h_dilation, h_batch_size)
		checkpoint_files = [d[:-len('_checkpoint_8.pth')] for d in os.listdir(checkpoint_dir) if d.endswith('8.pth')] \
			if os.path.exists(checkpoint_dir) else []
		print({
			'ModelType': name,
			'Epochs': h_epochs,
			'Lr': h_lr,
			'Lr-DecayStep': h_lr_decay_step,
			'Dilation': h_dilation,
			'Batch-Size': h_batch_size
		})
		# Check if training run has already been done (and do it again if overwrite=True)
		if not (title in checkpoint_files):
			# Model
			model = faster_rcnn_model_builder(h_type, h_dilation)
			# Trainer and Evaluator as helper classes
			trainer = Trainer(title, model, log_dir=log_dir, lr=h_lr, lr_decay_step=h_lr_decay_step, adam=use_adam)
			if validate:
				evaluator = Evaluator(model, trainer.tb_logger)
				trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda e: evaluator.run(validation_dataset))
			# Persisting checkpoints
			if not (trainer.scheduler is None):
				to_save = {'model': model, 'optimizer': trainer.optimizer,
				           'lr_scheduler': trainer.scheduler, 'trainer': trainer}
			else:
				to_save = {'model': model, 'optimizer': trainer.optimizer, 'trainer': trainer}
			checkpoint_handler = Checkpoint(to_save, DiskSaver(checkpoint_dir, require_empty=False),
			                                filename_prefix=title,
			                                global_step_transform=lambda engine, name: engine.state.epoch)
			trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler)
			if not (resume_checkpoint is None):
				to_load = to_save
				checkpoint = torch.load(resume_checkpoint)
				if not (checkpoint_dir in resume_checkpoint):
					# Loading a different type of model architecture
					model.load_state_dict(checkpoint, strict=False)
					del checkpoint['model']
					del to_load['model']
					del checkpoint['optimizer']
					del to_load['optimizer']
					del to_load['trainer']
				Checkpoint.load_objects(to_load, checkpoint)
			# Run training
			trainer.run(train_dataset, max_epochs=epochs)
			# Clean up CUDA cache
			del model
			torch.cuda.empty_cache()
