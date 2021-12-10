from i3Deep import utils
import numpy as np
from tqdm import tqdm

load_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/labelsTs/"

filenames = utils.load_filenames(load_dir)

shapes = []
for filename in tqdm(filenames):
    gt = utils.load_nifty(filename)[0]
    shapes.append(gt.shape)
    print(gt.shape)

shapes = np.mean(shapes, axis=0)
print(shapes)