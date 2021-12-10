import numpy as np
import os
from natsort import natsorted
from shutil import copyfile
from tqdm import tqdm
import json

dataset_json = {
    "name": "RibSeg",
    "description": "A rib segmentation dataset based on the RibFrac dataset",
    "reference": "https://zenodo.org/record/5336592",
    "licence": "CC BY-NC 4.0",
    "release": "08/30/2021",
    "tensorImageSize": "3D",
    "modality": {
        "0": "CT"
    },
    "labels": {
        "0": "background",
        "1": "rib"
    },
    "numTraining": "370",
    "numTest": "120"
}

training = []
test = []

load_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task148_RibSeg/labelsTr/"

filenames = os.listdir(load_dir)
filenames = np.asarray(filenames)
filenames = natsorted(filenames)

for filename in filenames:
    entry = {"image": "./imagesTr/" + filename, "label": "./labelsTr/" + filename}
    training.append(entry)


load_dir = "/dkfz/cluster/gpu/data/OE0441/k539i/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task148_RibSeg/labelsTs/"

filenames = os.listdir(load_dir)
filenames = np.asarray(filenames)
filenames = natsorted(filenames)

for filename in filenames:
    entry = "./imagesTs/" + filename
    test.append(entry)

dataset_json["training"] = training
dataset_json["test"] = test

with open('dataset.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_json, f, ensure_ascii=False, indent=4)