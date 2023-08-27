import torch
import glob, pydicom
from PIL import Image
import numpy as np
import os
import multiprocessing as mp
import util.util as util
from sklearn.preprocessing import LabelEncoder
from models.nnUNet.nnunetv2.paths import nnUNet_raw, nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join, load_json
from models.nnUNet.nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from models.nnUNet.nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from models.nnUNet.nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints, plan_experiments, preprocess
from models.nnUNet.nnunetv2.run.run_training import run_training
from models.nnUNet.nnunetv2.evaluation.find_best_configuration import find_best_configuration
from models.nnUNet.nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from models.nnUNet.nnunetv2.utilities.overlay_plots import plot_overlay
from models.nnUNet.nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from models.nnUNet.nnunetv2.imageio.natural_image_reager_writer import NaturalImage2DIO

def dicom_to_tiff(sourcepath, destinationpath):

     #Reading all the dcm images from the directory
    dcmFilePath = sourcepath + 'Cleaned_image'
    labelFilePath = sourcepath + 'Cleaned_mask/*'
    dcmFileList = glob.glob(dcmFilePath + "/*")
    labelFileList = glob.glob(labelFilePath)
    #should there be a requirement to get the name of the indi files. 
    file_name = [x.split("/")[-1] for x in dcmFileList]
    dcmData, dcmPixelData = util.data_loader(dcmFilePath)
    
    #reading the metadata from dicom image
    for id, x in enumerate(dcmPixelData):
        x = np.uint8((np.maximum(x, 0)/x.max())*255)
        im = Image.fromarray(x)
        im.save(destinationpath + '/imagesTr/' + file_name[id] + '_0000.png')
        labelFilePath = sourcepath + 'Cleaned_mask/' + file_name[id] + '.tag'
#        if labelFilePath == labelFileList:
        masked_image = util.maskedImageReader(dcmData[id], labelFilePath)
        masked_image = np.array(masked_image)
        masked_image = np.where(masked_image == 14, 0, masked_image)
        labelencoder = LabelEncoder()
        h,w = masked_image.shape
        mask_reshaped = masked_image.flatten()
        mask_reshaped_encoded = labelencoder.fit_transform(mask_reshaped)
        mask_encoded_original_shape = np.array(mask_reshaped_encoded.reshape(h,w), dtype=np.uint8)
        im = Image.fromarray(mask_encoded_original_shape)
        im.save(destinationpath + '/labelsTr/' + file_name[id] + '.png')


def read_pred():
     dcm_test_file_path = join(nnUNet_raw, dataset_name + '/imagesTs')
     pred_label_file_path = join(nnUNet_raw, dataset_name + '/imagesTs_pred/')
     dcm_test_file_list = glob.glob(dcm_test_file_path + "/*")
     pred_label_file_list = glob.glob(pred_label_file_path + '*')
     #should there be a requirement to get the name of the indi files. 
     test_file_name = [x.split("/")[-1] for x in dcm_test_file_list]
     pred_file_name = [x.split("/")[-1] for x in dcm_test_file_list]
     test_dcmData, test_dcmPixelData = util.data_loader(dcm_test_file_path)
     for id, x in enumerate(test_dcmPixelData):
         pred_label_file_path = pred_label_file_path + pred_label_file_list[id]
         pred_masked_image = util.maskedImageReader(test_dcmData[id], pred_label_file_path)
         pred_masked_image = np.array(pred_masked_image)
         pred_masked_image = np.where(pred_masked_image == 2, 255, pred_masked_image)
         pred_masked_image = np.where(pred_masked_image < 200, 0, pred_masked_image)
         pred_im = Image.fromarray(pred_masked_image)
         pred_im.save(pred_label_file_list[id] + '_processed.png')



destination_path = '/Users/niketsingla/Documents/Melbourne Uni/MAST90106 - Data Science Proj 1/Code/MAST90106_group8/nnUNet_raw/Dataset888_ColorectalCancer'
source_path = '/Users/niketsingla/Documents/Melbourne Uni/MAST90106 - Data Science Proj 1/Code/MAST90106_group8/'
#dicom_to_tiff(source_path, destination_path)
dataset_name = 'Dataset888_ColorectalCancer'
#generate_dataset_json(join(nnUNet_raw, dataset_name), {0: "CT"}, {'background': 0, 'SAT': 1, 'SM': 2, 'VAT': 3, 'IMAT': 4},
#                          270, '.png', dataset_name=dataset_name)

example_folder = join(nnUNet_raw, dataset_name)
num_processes = 6
dataset_ids = [888]
if __name__ == "__main__":  
     os.environ['OMP_NUM_THREADS']="1"
     extract_fingerprints(dataset_ids)
     plan_experiments(dataset_ids)
     preprocess(dataset_ids)
     verify_dataset_integrity(example_folder, num_processes)
     if torch.cuda.is_available():
          device = torch.device("cuda")
          torch.set_num_threads(1)
          torch.set_num_interop_threads(1)
     elif torch.backends.mps.is_available():
          device = torch.device("mps")
     else:
          device = torch.device("cpu")
          #torch.set_num_threads(mp.cpu_count())
          torch.set_num_threads(1)
          torch.set_num_interop_threads(1)
     print(device)
     run_training(dataset_name, '2d', 5, trainer_class_name = 'nnUNetTrainer_50epochs', device=device)
     #    find_best_configuration(dataset_name, 
     #                          [{'plans': 'nnUNetPlans', 'configuration': '2d', 'trainer': 'nnUNetTrainer_5epochs'}],
     #                         False, 
     #                          270, 
     #                          True,
     #                          (3, ))
     predictor = nnUNetPredictor(device=device)
     predictor.initialize_from_trained_model_folder(join(nnUNet_results, dataset_name, 'nnUNetTrainer_20epochs__nnUNetPlans__2d'), '5')
     predictor.predict_from_files(join(nnUNet_raw, dataset_name + '/imagesTs'),
                              join(nnUNet_raw, dataset_name + '/imagesTs_pred'),
                              save_probabilities=False, overwrite=False,
                              num_processes_preprocessing=2, num_processes_segmentation_export=2,
                              folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
        
        ## Read one of the prediction images for a specified segmentation mask
     imgio = NaturalImage2DIO()
     base_path = join(nnUNet_raw, dataset_name)
     test_image_path = join(base_path, 'imagesTs')
     pred_image_path = join(base_path, 'imagesTs_pred')
     test_image_file_list = glob.glob(base_path + "/imagesTs/*")
     #should there be a requirement to get the name of the indi files. 
     test_image_file_name = [x.split("/")[-1] for x in  test_image_file_list]
     pred_image_file_list = glob.glob(base_path + "/imagesTs_pred/*")
     for id, x in enumerate(test_image_file_list):
          output_image = join(base_path, ('imagesTs_pred_processed/' + test_image_file_name[id].split('_')[-2] + '_processed.png'))
          plot_overlay(test_image_file_list[id], join(pred_image_path, test_image_file_name[id].split('_')[-2] + '.png'), imgio, output_image)
     #source_path = '/Users/niketsingla/Documents/Melbourne Uni/MAST90106 - Data Science Proj 1/Code/MAST90106_group8/'
     #test_image = join(nnUNet_raw, dataset_name + '/imagesTs/6887 4 16.5.2012 73-605_0000.png')
     #output_image = join(nnUNet_raw, dataset_name + '/imagesTs/6887 4 16.5.2012 73-605_0000_processed.png')
     #pred_image = join(nnUNet_raw, dataset_name + '/imagesTs_pred/6887 4 16.5.2012 73-605.png')
     
     #plot_overlay(test_image, pred_image, imgio, output_image)