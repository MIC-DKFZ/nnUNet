from medseg import utils
import os
import json

path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task007_Pancreas/"
labels = False


if not labels:
    filenames = utils.load_filenames(path + "imagesTr/")
else:
    filenames = utils.load_filenames(path + "labelsTr/")

imagesTr = []
counter = 1
for filename in filenames:
    if not labels:
        name = os.path.dirname(filename) + "/" + str(counter).zfill(4) + filename[-12:]
    else:
        name = os.path.dirname(filename) + "/" + str(counter).zfill(4) + filename[-7:]
    json_name = str(counter).zfill(4) + ".nii.gz"
    print(name)
    # os.rename(filename, name)
    counter += 1
    imagesTr.append({"image": "./imagesTr/" + json_name, "label": "./labelsTr/" + json_name})

if not labels:
    print("")
    print(imagesTr[:100])
    print("")
    print(imagesTr[100:])
    with open(path + 'imagesTr.json', 'w') as outfile:
        json.dump(imagesTr[:100], outfile)
    with open(path + 'imagesTs.json', 'w') as outfile:
        json.dump(imagesTr[100:], outfile)