from medseg import utils
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pickle
import seaborn as sns

gt_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/labelsTs/"
save_dir = "/gris/gris-f/homelv/kgotkows/datasets/nnUnet_datasets/nnUNet_raw_data/nnUNet_raw_data/Task070_guided_all_public_ggo/"
# gt_dir = utils.fix_path(gt_dir)
# gt_filenames = utils.load_filenames(gt_dir)
#
# data = []
# for filename in tqdm(gt_filenames):
#     gt = utils.load_nifty(filename)[0]
#     gt2size_ratio = np.sum(gt) / np.prod(gt.shape)
#     data.append(gt2size_ratio)
#     print("{}: {}".format(os.path.basename(filename), gt2size_ratio))
#
# with open(save_dir + "gt2size_ratio.pkl", 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(save_dir + "gt2size_ratio.pkl", 'rb') as handle:
    data = pickle.load(handle)

sns.set_theme(style="whitegrid")
sns.swarmplot(data, color="k", alpha=0.8)

# plt.scatter(data)
plt.savefig(save_dir + "gt2size_ratio.png")