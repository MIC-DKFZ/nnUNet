# BraTS2017/2018 example


This folder contains a complete example of how to process BraTS2017/2018 data with batchgenerators. You need to 
adapt the scripts to match your system and download location. The adaptation should be straightforward. All you need to 
do is to change the paths and the number of threads in config.py, then execute `brats2017_preprocessing.py` for 
preprocessing the data.

Once preprocessed, have a look at `brats2017_dataloader_2D.py` and `brats2017_dataloader_3D.py` for how to implement 
data loader for 2D and 3D network training, respectively. Naturally these files contain everything you need, including
data augmentation and multiprocessing. They are not designed to be just executed because there is no network training 
in there. The idea is that you look at them and execute what code they have in a controlled manner so that you can
get a feel for how batchgenerators work. Questions? -> f.isensee@dkfz.de

Why are these not IPython Notebooks? I don't like IPython Notebooks. Simple =)

**IMPORTANT** these DataLoaders are not suited for test set prediction! You need to iterate over preprocessed test 
data by yourself. 
