#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import zipfile
from time import time

import requests
from batchgenerators.utilities.file_and_folder_operations import join, isfile

from nnunet.paths import network_training_output_dir


def get_available_models():
    available_models = {
        "Task001_BrainTumour": {
            'description': "Brain Tumor Segmentation. \n"
                           "Segmentation targets are edema, enhancing tumor and necrosis, \n"
                           "Input modalities are 0: FLAIR, 1: T1, 2: T1 with contrast agent, 3: T2. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task001_BrainTumour.zip?download=1"
        },
        "Task002_Heart": {
            'description': "Left Atrium Segmentation. \n"
                           "Segmentation target is the left atrium, \n"
                           "Input modalities are 0: MRI. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task002_Heart.zip?download=1"
        },
        "Task003_Liver": {
            'description': "Liver and Liver Tumor Segmentation. \n"
                           "Segmentation targets are liver and tumors, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task003_Liver.zip?download=1"
        },
        "Task004_Hippocampus": {
            'description': "Hippocampus Segmentation. \n"
                           "Segmentation targets posterior and anterior parts of the hippocampus, \n"
                           "Input modalities are 0: MRI. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task004_Hippocampus.zip?download=1"
        },
        "Task005_Prostate": {
            'description': "Prostate Segmentation. \n"
                           "Segmentation targets are peripheral and central zone, \n"
                           "Input modalities are 0: T2, 1: ADC. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4485926/files/Task005_Prostate.zip?download=1"
        },
        "Task006_Lung": {
            'description': "Lung Nodule Segmentation. \n"
                           "Segmentation target are lung nodules, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task006_Lung.zip?download=1"
        },
        "Task007_Pancreas": {
            'description': "Pancreas Segmentation. \n"
                           "Segmentation targets are pancras and pancreas tumor, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task007_Pancreas.zip?download=1"
        },
        "Task008_HepaticVessel": {
            'description': "Hepatic Vessel Segmentation. \n"
                           "Segmentation targets are hepatic vesels and liver tumors, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task008_HepaticVessel.zip?download=1"
        },
        "Task009_Spleen": {
            'description': "Spleen Segmentation. \n"
                           "Segmentation target is the spleen, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task009_Spleen.zip?download=1"
        },
        "Task010_Colon": {
            'description': "Colon Cancer Segmentation. \n"
                           "Segmentation target are colon caner primaries, \n"
                           "Input modalities are 0: CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': "https://zenodo.org/record/4003545/files/Task010_Colon.zip?download=1"
        },
        "Task017_AbdominalOrganSegmentation": {
            'description': "Multi-Atlas Labeling Beyond the Cranial Vault - Abdomen. \n"
                           "Segmentation targets are thirteen different abdominal organs, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see https://www.synapse.org/#!Synapse:syn3193805/wiki/217754",
            'url': "https://zenodo.org/record/4003545/files/Task017_AbdominalOrganSegmentation.zip?download=1"
        },
        "Task024_Promise": {
            'description': "Prostate MR Image Segmentation 2012. \n"
                           "Segmentation target is the prostate, \n"
                           "Input modalities are 0: T2. \n"
                           "Also see https://promise12.grand-challenge.org/",
            'url': "https://zenodo.org/record/4003545/files/Task024_Promise.zip?download=1"
        },
        "Task027_ACDC": {
            'description': "Automatic Cardiac Diagnosis Challenge. \n"
                           "Segmentation targets are right ventricle, left ventricular cavity and left myocardium, \n"
                           "Input modalities are 0: cine MRI. \n"
                           "Also see https://acdc.creatis.insa-lyon.fr/",
            'url': "https://zenodo.org/record/4003545/files/Task027_ACDC.zip?download=1"
        },
        "Task029_LiTS": {
            'description': "Liver and Liver Tumor Segmentation Challenge. \n"
                           "Segmentation targets are liver and liver tumors, \n"
                           "Input modalities are 0: abdominal CT scan. \n"
                           "Also see https://competitions.codalab.org/competitions/17094",
            'url': "https://zenodo.org/record/4003545/files/Task029_LITS.zip?download=1"
        },
        "Task035_ISBILesionSegmentation": {
            'description': "Longitudinal multiple sclerosis lesion segmentation Challenge. \n"
                           "Segmentation target is MS lesions, \n"
                           "input modalities are 0: FLAIR, 1: MPRAGE, 2: proton density, 3: T2. \n"
                           "Also see https://smart-stats-tools.org/lesion-challenge",
            'url': "https://zenodo.org/record/4003545/files/Task035_ISBILesionSegmentation.zip?download=1"
        },
        "Task038_CHAOS_Task_3_5_Variant2": {
            'description': "CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge (Task 3 & 5). \n"
                           "Segmentation targets are left and right kidney, liver, spleen, \n"
                           "Input modalities are 0: T1 in-phase, T1 out-phase, T2 (can be any of those)\n"
                           "Also see https://chaos.grand-challenge.org/",
            'url': "https://zenodo.org/record/4003545/files/Task038_CHAOS_Task_3_5_Variant2.zip?download=1"
        },
        "Task048_KiTS_clean": {
            'description': "Kidney and Kidney Tumor Segmentation Challenge. "
                           "Segmentation targets kidney and kidney tumors, "
                           "Input modalities are 0: abdominal CT scan. "
                           "Also see https://kits19.grand-challenge.org/",
            'url': "https://zenodo.org/record/4003545/files/Task048_KiTS_clean.zip?download=1"
        },
        "Task055_SegTHOR": {
            'description': "SegTHOR: Segmentation of THoracic Organs at Risk in CT images. \n"
                           "Segmentation targets are aorta, esophagus, heart and trachea, \n"
                           "Input modalities are 0: CT scan. \n"
                           "Also see https://competitions.codalab.org/competitions/21145",
            'url': "https://zenodo.org/record/4003545/files/Task055_SegTHOR.zip?download=1"
        },
        "Task061_CREMI": {
            'description': "MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images (Synaptic Cleft segmentation task). \n"
                           "Segmentation target is synaptic clefts, \n"
                           "Input modalities are 0: serial section transmission electron microscopy of neural tissue. \n"
                           "Also see https://cremi.org/",
            'url': "https://zenodo.org/record/4003545/files/Task061_CREMI.zip?download=1"
        },
        "Task075_Fluo_C3DH_A549_ManAndSim": {
            'description': "Fluo-C3DH-A549-SIM and Fluo-C3DH-A549 datasets of the cell tracking challenge. Segmentation target are C3DH cells in fluorescence microscopy images.\n"
                           "Input modalities are 0: fluorescence_microscopy\n"
                           "Also see http://celltrackingchallenge.net/",
            'url': "https://zenodo.org/record/4003545/files/Task075_Fluo_C3DH_A549_ManAndSim.zip?download=1"
        },
        "Task076_Fluo_N3DH_SIM": {
            'description': "Fluo-N3DH-SIM dataset of the cell tracking challenge. Segmentation target are N3DH cells and cell borders in fluorescence microscopy images.\n"
                           "Input modalities are 0: fluorescence_microscopy\n"
                           "Also see http://celltrackingchallenge.net/\n"
                           "Note that the segmentation output of the models are cell center and cell border. These outputs mus tbe converted to an instance segmentation for the challenge. \n"
                           "See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task076_Fluo_N3DH_SIM.py",
            'url': "https://zenodo.org/record/4003545/files/Task076_Fluo_N3DH_SIM.zip?download=1"
        },
        "Task082_BraTS2020": {
            'description': "Brain tumor segmentation challenge 2020 (BraTS)\n"
                           "Segmentation targets are 0: background, 1: edema, 2: enhancing tumor, 3: necrosis\n"
                           "Input modalities are 0: T1, 1: T1ce, 2: T2, 3: FLAIR (MRI images)\n"
                           "Also see https://www.med.upenn.edu/cbica/brats2020/",
            'url': (
                "https://zenodo.org/record/4635763/files/Task082_nnUNetTrainerV2__nnUNetPlansv2.1_5fold.zip?download=1",
                "https://zenodo.org/record/4635763/files/Task082_nnUNetTrainerV2BraTSRegions_DA3_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip?download=1",
                "https://zenodo.org/record/4635763/files/Task082_nnUNetTrainerV2BraTSRegions_DA4_BN__nnUNetPlansv2.1_bs5_15fold.zip?download=1",
                "https://zenodo.org/record/4635763/files/Task082_nnUNetTrainerV2BraTSRegions_DA4_BN_BD__nnUNetPlansv2.1_bs5_5fold.zip?download=1",
            )
        },
        "Task089_Fluo-N2DH-SIM_thickborder_time": {
            'description': "Fluo-N2DH-SIM dataset of the cell tracking challenge. Segmentation target are nuclei of N2DH cells and cell borders in fluorescence microscopy images.\n"
                           "Input modalities are 0: t minus 4, 0: t minus 3, 0: t minus 2, 0: t minus 1, 0: frame of interest\n"
                           "Note that the input channels are different time steps from a time series acquisition\n"
                           "Note that the segmentation output of the models are cell center and cell border. These outputs mus tbe converted to an instance segmentation for the challenge. \n"
                           "See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task089_Fluo-N2DH-SIM.py\n"
                           "Also see http://celltrackingchallenge.net/",
            'url': "https://zenodo.org/record/4003545/files/Task089_Fluo-N2DH-SIM_thickborder_time.zip?download=1"
        },
        "Task114_heart_MNMs": {
            'description':  "Cardiac MRI short axis images from the M&Ms challenge 2020.\n"
                            "Input modalities are 0: MRI \n"
                            "See also https://www.ub.edu/mnms/ \n"
                            "Note: Labels of the M&Ms Challenge are not in the same order as for the ACDC challenge. \n"
                            "See https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/dataset_conversion/Task114_heart_mnms.py",
            'url': "https://zenodo.org/record/4288464/files/Task114_heart_MNMs.zip?download=1"
        },
        "Task115_COVIDSegChallenge": {
            'description': "Covid lesion segmentation in CT images. Data originates from COVID-19-20 challenge.\n"
                           "Predicted labels are 0: background, 1: covid lesion\n"
                           "Input modalities are 0: CT \n"
                           "See also https://covid-segmentation.grand-challenge.org/",
            'url': (
                "",
                "",
                "",
                "",
                "",
            )
        },

    }
    return available_models


def print_available_pretrained_models():
    print('The following pretrained models are available:\n')
    av_models = get_available_models()
    for m in av_models.keys():
        print('')
        print(m)
        print(av_models[m]['description'])


def download_and_install_pretrained_model_by_name(taskname):
    av_models = get_available_models()
    if taskname not in av_models.keys():
        raise RuntimeError("\nThe requested pretrained model ('%s') is not available." % taskname)
    if len(av_models[taskname]['url']) == 0:
        raise RuntimeError("The requested model has not been uploaded yet. Please check back in a few days")
    url = av_models[taskname]['url']
    if isinstance(url, str):
        download_and_install_from_url(url)
    elif isinstance(url, (tuple, list)):
        for u in url:
            download_and_install_from_url(u)
    else:
        raise RuntimeError('URL for download_and_install_from_url must be either str or list/tuple of str')


def download_and_install_from_url(url):
    assert network_training_output_dir is not None, "Cannot install model because network_training_output_dir is not " \
                                                    "set (RESULTS_FOLDER missing as environment variable, see " \
                                                    "Installation instructions)"
    import http.client
    http.client.HTTPConnection._http_vsn = 10
    http.client.HTTPConnection._http_vsn_str = 'HTTP/1.0'

    import os
    home = os.path.expanduser('~')
    random_number = int(time() * 1e7)
    tempfile = join(home, '.nnunetdownload_%s' % str(random_number))

    try:
        with open(tempfile, 'wb') as f:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=8192 * 16):
                    # If you have chunk encoded response uncomment if
                    # and set chunk_size parameter to None.
                    # if chunk:
                    f.write(chunk)

        print("Download finished. Extracting...")
        install_model_from_zip_file(tempfile)
        print("Done")
    except Exception as e:
        raise e
    finally:
        if isfile(tempfile):
            os.remove(tempfile)


def download_file(url, local_filename):
    # borrowed from https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=None):
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk:
                f.write(chunk)
    return local_filename


def install_model_from_zip_file(zip_file: str):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(network_training_output_dir)


def print_license_warning():
    print('')
    print('######################################################')
    print('!!!!!!!!!!!!!!!!!!!!!!!!WARNING!!!!!!!!!!!!!!!!!!!!!!!')
    print('######################################################')
    print("Using the pretrained model weights is subject to the license of the dataset they were trained on. Some "
          "allow commercial use, others don't. It is your responsibility to make sure you use them appropriately! Use "
          "nnUNet_print_pretrained_model_info(task_name) to see a summary of the dataset and where to find its license!")
    print('######################################################')
    print('')


def download_by_name():
    import argparse
    parser = argparse.ArgumentParser(description="Use this to download pretrained models. CAREFUL: This script will "
                                                 "overwrite "
                                                 "existing models (if they share the same trainer class and plans as "
                                                 "the pretrained model")
    parser.add_argument("task_name", type=str, help='Task name of the pretrained model. To see '
                                                                   'available task names, run nnUNet_print_available_'
                                                                   'pretrained_models')
    args = parser.parse_args()
    taskname = args.task_name
    print_license_warning()
    download_and_install_pretrained_model_by_name(taskname)


def download_by_url():
    import argparse
    parser = argparse.ArgumentParser(
        description="Use this to download pretrained models. This script is intended to download models via url only. "
                    "If you want to download one of our pretrained models, please use nnUNet_download_pretrained_model. "
                    "CAREFUL: This script will overwrite "
                    "existing models (if they share the same trainer class and plans as "
                    "the pretrained model.")
    parser.add_argument("url", type=str, help='URL of the pretrained model')
    args = parser.parse_args()
    url = args.url
    download_and_install_from_url(url)


def install_from_zip_entry_point():
    import argparse
    parser = argparse.ArgumentParser(
        description="Use this to install a zip file containing a pretrained model.")
    parser.add_argument("zip", type=str, help='zip file')
    args = parser.parse_args()
    zip = args.zip
    install_model_from_zip_file(zip)


def print_pretrained_model_requirements():
    import argparse
    parser = argparse.ArgumentParser(description="Use this to see the properties of a pretrained model, especially "
                                                 "what input modalities it requires")
    parser.add_argument("task_name", type=str, help='Task name of the pretrained model. To see '
                                                                   'available task names, run nnUNet_print_available_'
                                                                   'pretrained_models')
    args = parser.parse_args()
    taskname = args.task_name
    av = get_available_models()
    if taskname not in av.keys():
        raise RuntimeError("Invalid task name. This pretrained model does not exist. To see available task names, "
                           "run nnUNet_print_available_pretrained_models")
    print(av[taskname]['description'])


if __name__  == '__main__':
    url = 'https://www.dropbox.com/s/ft54q1gi060vm2x/Task004_Hippocampus.zip?dl=1'