import os
import warnings

import fire

from src.dataset import extract_dataset, download_zips
from src.preview import generate_preview
from src.reporter import generate_report
from src.trainer import train

warnings.filterwarnings("ignore")


class EntryPoint:
    @staticmethod
    def data_extraction(data_dir='data'):
        """
        Extract the images from the zip files that are listed in the csv file and write them to the output directory.
        :param data_dir: String. The path to the dataset directory.
        """
        csv_file, zip_directory = download_zips(os.path.join(data_dir, 'raw'))
        extract_dataset(zip_directory, csv_file, data_dir)

    @staticmethod
    def run(data_dir, dl_info_file, model_type=1, epochs=8, lr=0.002, lr_decay_step=7, batch_size=8,
            dilation=[[1, 1, 1]], validate=True, use_adam=False, log_dir='log', checkpoint_dir='checkpoints',
            checkpoint_file=None, avg_fps=[0, 0.125, 0.25, 0.5, 1, 2, 4, 5, 8], output_dir='output'):
        """
        Train one or multiple model(s) and generate a report afterwards.
        :param data_dir: String. The path to the data directory.
        :param dl_info_file: String. The path of the DL_info.csv file.
        :param model_type: Integer or array[Integer]. The type of the model referred to as a number:
                            BASIC = 1
                            DEFORMABLE_CONV = 2
                            DEFORMABLE_ROI = 3
                            DEFORMABLE_CONV_ROI = 4
        :param epochs: Integer or array[Integer]. The number of epochs to train.
        :param lr: Float or array[Float]. The learning rate that should be used.
        :param lr_decay_step: Integer or array[Integer]. The steps the learning rate should be reduced by a factor of 0.1.
        :param batch_size: Integer or array[Integer]. The batch size the training should be done with.
        :param dilation: array[array[Integer]]. See https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html for further information.
        :param validate: Bool. Use a validation step after each epoch.
        :param use_adam: Bool. Use of the adam optimizer.
        :param log_dir: String. The path to the log directory.
        :param checkpoint_dir: String. The path to the checkpoint directory.
        :param checkpoint_file: String. The path to the checkpoint file from which to resume.
        :param avg_fps: array[Float]. The false positives of the x-axis which should be interpolated.
        :param output_dir: String. The path to the files which will contain the results and the report.
        """
        print('##### TRAINING #####')
        train(model_type=model_type,
              lr=lr,
              lr_decay_step=lr_decay_step,
              epochs=epochs,
              batch_size=batch_size,
              dilation=dilation,
              validate=bool(validate),
              log_dir=log_dir,
              data_dir=data_dir,
              csv_file=dl_info_file,
              use_adam=bool(use_adam),
              checkpoint_dir=checkpoint_dir,
              resume_checkpoint=checkpoint_file)
        print('##### REPORT #####')
        generate_report(data_dir=data_dir,
                        csv_file=dl_info_file,
                        checkpoint_dir=checkpoint_dir,
                        output_dir=output_dir,
                        avg_fps=avg_fps)
        print('##### PREVIEW #####')
        generate_preview(data_dir=data_dir,
                         csv_file=dl_info_file,
                         checkpoint_dir=checkpoint_dir,
                         output_dir=output_dir)


if __name__ == '__main__':
    fire.Fire(EntryPoint)
