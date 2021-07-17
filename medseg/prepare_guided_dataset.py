from pathlib import Path
from medseg import utils
from shutil import copyfile
import os
from tqdm import tqdm

path = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task002_BrainTumour_guided/"
cases_split = 50
modalities = 4



data_split = cases_split * modalities
Path(path + "predictionsTs/").mkdir(parents=True, exist_ok=True)

split = ["refinement_test", "refinement_val"]

for s in split:
    Path(path + s + "/basic_predictions/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/GridSearchResults/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/images/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/labels/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/recommended_masks/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/refined_predictions/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/refinement_inference_tmp/part0/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/refinement_inference_tmp/part1/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/refinement_inference_tmp/part2/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/refinement_inference_tmp/part3/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/ensemble/bhattacharyya_coefficient/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/ensemble/predictive_entropy/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/ensemble/predictive_variance/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/mcdo/bhattacharyya_coefficient/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/mcdo/predictive_entropy/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/mcdo/predictive_variance/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/tta/bhattacharyya_coefficient/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/tta/predictive_entropy/").mkdir(parents=True, exist_ok=True)
    Path(path + s + "/uncertainties/tta/predictive_variance/").mkdir(parents=True, exist_ok=True)


images = filenames = utils.load_filenames(path + "imagesTs/")
images_val = images[:data_split]
images_test = images[data_split:]

for image in tqdm(images_val):
    copyfile(image, path + "refinement_val/images/" + os.path.basename(image))

for image in tqdm(images_test):
    copyfile(image, path + "refinement_test/images/" + os.path.basename(image))

labels = filenames = utils.load_filenames(path + "labelsTs/")
labels_val = labels[:cases_split]
labels_test = labels[cases_split:]

for label in tqdm(labels_val):
    copyfile(label, path + "refinement_val/labels/" + os.path.basename(label))

for label in tqdm(labels_test):
    copyfile(label, path + "refinement_test/labels/" + os.path.basename(label))



quarter = int(len(images_val) / (4 * modalities))
filenames0 = images_val[:quarter*modalities]
filenames1 = images_val[quarter*modalities:quarter*modalities*2]
filenames2 = images_val[quarter*modalities*2:quarter*modalities*3]
filenames3 = images_val[quarter*modalities*3:]

for filename in tqdm(filenames0):
    copyfile(filename, path + "refinement_val/refinement_inference_tmp/part0/" + os.path.basename(filename))
for filename in tqdm(filenames1):
    copyfile(filename, path + "refinement_val/refinement_inference_tmp/part1/" + os.path.basename(filename))
for filename in tqdm(filenames2):
    copyfile(filename, path + "refinement_val/refinement_inference_tmp/part2/" + os.path.basename(filename))
for filename in tqdm(filenames3):
    copyfile(filename, path + "refinement_val/refinement_inference_tmp/part3/" + os.path.basename(filename))


quarter = int(len(images_test) / (4 * modalities))
filenames0 = images_test[:quarter*modalities]
filenames1 = images_test[quarter*modalities:quarter*modalities*2]
filenames2 = images_test[quarter*modalities*2:quarter*modalities*3]
filenames3 = images_test[quarter*modalities*3:]

for filename in tqdm(filenames0):
    copyfile(filename, path + "refinement_test/refinement_inference_tmp/part0/" + os.path.basename(filename))
for filename in tqdm(filenames1):
    copyfile(filename, path + "refinement_test/refinement_inference_tmp/part1/" + os.path.basename(filename))
for filename in tqdm(filenames2):
    copyfile(filename, path + "refinement_test/refinement_inference_tmp/part2/" + os.path.basename(filename))
for filename in tqdm(filenames3):
    copyfile(filename, path + "refinement_test/refinement_inference_tmp/part3/" + os.path.basename(filename))