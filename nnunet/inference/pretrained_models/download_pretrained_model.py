import tempfile
import urllib
from urllib.request import urlopen
from nnunet.paths import network_training_output_dir
from subprocess import call


def get_available_models():
    available_models = {
        "Task001_BrainTumour": {
            'description': "Brain Tumor Segmentation. \n"
                           "Segmentation targets are edema, enhancing tumor and necrosis, \n"
                           "input modalities are 0: FLAIR, 1: T1, 2: T1 with contrast agent, 3: T2. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task002_Heart": {
            'description': "Left Atrium Segmentation. \n"
                           "Segmentation target is the left atrium, \n"
                           "input modalities are 0: MRI. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task003_Liver": {
            'description': "Liver and Liver Tumor Segmentation. \n"
                           "Segmentation targets are liver and tumors, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task004_Hippocampus": {
            'description': "Hippocampus Segmentation. \n"
                           "Segmentation targets posterior and anterior parts of the hippocampus, \n"
                           "input modalities are 0: MRI. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task005_Prostate": {
            'description': "Prostate Segmentation. \n"
                           "Segmentation targets are peripheral and central zone, \n"
                           "input modalities are 0: T2, 1: ADC. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task006_Lung": {
            'description': "Lung Nodule Segmentation. \n"
                           "Segmentation target are lung nodules, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task007_Pancreas": {
            'description': "Pancreas Segmentation. \n"
                           "Segmentation targets are pancras and pancreas tumor, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task008_HepaticVessel": {
            'description': "Hepatic Vessel Segmentation. \n"
                           "Segmentation targets are hepatic vesels and liver tumors, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task009_Spleen": {
            'description': "Spleen Segmentation. \n"
                           "Segmentation target is the spleen, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task010_Colon": {
            'description': "Colon Cancer Segmentation. \n"
                           "Segmentation target are colon caner primaries, \n"
                           "input modalities are 0: CT scan. \n"
                           "Also see Medical Segmentation Decathlon, http://medicaldecathlon.com/",
            'url': ""
        },
        "Task017_AbdominalOrganSegmentation": {
            'description': "Multi-Atlas Labeling Beyond the Cranial Vault - Abdomen. \n"
                           "Segmentation targets are thirteen different abdominal organs, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see https://www.synapse.org/#!Synapse:syn3193805/wiki/217754",
            'url': ""
        },
        "Task024_Promise": {
            'description': "Prostate MR Image Segmentation 2012. \n"
                           "Segmentation target is the prostate, \n"
                           "input modalities are 0: T2. \n"
                           "Also see https://promise12.grand-challenge.org/",
            'url': ""
        },
        "Task027_ACDC": {
            'description': "Automatic Cardiac Diagnosis Challenge. \n"
                           "Segmentation targets are right ventricle, left ventricular cavity and left myocardium, \n"
                           "input modalities are 0: cine MRI. \n"
                           "Also see https://acdc.creatis.insa-lyon.fr/",
            'url': ""
        },
        "Task029_LiTS": {
            'description': "Liver and Liver Tumor Segmentation Challenge. \n"
                           "Segmentation targets are liver and liver tumors, \n"
                           "input modalities are 0: abdominal CT scan. \n"
                           "Also see https://competitions.codalab.org/competitions/17094",
            'url': ""
        },
        "Task035_ISBILesionSegmentation": {
            'description': "Longitudinal multiple sclerosis lesion segmentation Challenge. \n"
                           "Segmentation target is MS lesions, \n"
                           "input modalities are 0: FLAIR, 1: MPRAGE, 2: proton density, 3: T2. \n"
                           "Also see https://smart-stats-tools.org/lesion-challenge",
            'url': ""
        },
        "Task038_CHAOS_Task_3_5_Variant2": {
            'description': "CHAOS - Combined (CT-MR) Healthy Abdominal Organ Segmentation Challenge (Task 3 & 5). \n"
                           "Segmentation targets are left and right kidney, liver, spleen, \n"
                           "input modalities are 0: T1 in-phase, T1 out-phase, T2 (can be any of those)\n"
                           "Also see https://chaos.grand-challenge.org/",
            'url': ""
        },
        "Task048_KiTS_clean": {
            'description': "Kidney and Kidney Tumor Segmentation Challenge. "
                           "Segmentation targets kidney and kidney tumors, "
                           "input modalities are 0: abdominal CT scan. "
                           "Also see https://kits19.grand-challenge.org/",
            'url': ""
        },
        "Task055_SegTHOR": {
            'description': "SegTHOR: Segmentation of THoracic Organs at Risk in CT images. \n"
                           "Segmentation targets are aorta, esophagus, heart and trachea, \n"
                           "input modalities are 0: CT scan. \n"
                           "Also see https://competitions.codalab.org/competitions/21145",
            'url': ""
        },
        "Task061_CREMI": {
            'description': "MICCAI Challenge on Circuit Reconstruction from Electron Microscopy Images (Synaptic Cleft segmentation task). \n"
                           "Segmentation target is synaptic clefts, \n"
                           "input modalities are 0: serial section transmission electron microscopy of neural tissue. \n"
                           "Also see https://cremi.org/",
            'url': ""
        },

    }
    return available_models


def print_available_pretrained_models():
    print('The following pretrained models are available:\n')
    av_models = get_available_models()
    for m in av_models.keys():
        print(av_models[m]['description'])


def download_and_install_pretrained_model_by_name(taskname):
    av_models = get_available_models()
    if taskname not in av_models.keys():
        raise RuntimeError("\nThe requested pretrained model ('%s') is not available." % taskname)
    if len(av_models[taskname]['url']) == 0:
        raise RuntimeError("The requested model has not been uploaded yet. Please check back in a few days")
    download_and_install_from_url(av_models[taskname]['url'])


def download_and_install_from_url(url):
    assert network_training_output_dir is not None, "Cannot install model because network_training_output_dir is not " \
                                                    "set (RESULTS_FOLDER missing as environment variable, see " \
                                                    "Installation instructions)"
    with tempfile.NamedTemporaryFile() as f:
        fname = f.name
        print("Downloading pretrained model", url)
        data = urlopen(url).read()
        f.write(data)
        # unzip -o zip_file -d output_dir
        print("Download finished. Extracting...")
        call(['unzip', '-o', '-d', network_training_output_dir, fname])
        print("Done")


if __name__  == '__main__':
    url = 'https://www.dropbox.com/s/ft54q1gi060vm2x/Task004_Hippocampus.zip?dl=1'