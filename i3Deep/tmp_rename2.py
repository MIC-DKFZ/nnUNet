from i3Deep import utils
import os
from tqdm import tqdm

path = ""

filenames = utils.load_filenames(path)
for filename in tqdm(filenames):
    new_filename = path + "0" + os.path.basename(filename)[6:]
    os.rename(filename, new_filename)