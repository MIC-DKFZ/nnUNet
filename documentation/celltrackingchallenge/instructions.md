# Fluo-C3DH-A549 and Fluo-C3DH-A549-SIM
These datasets are usable by nnU-Net as is, so all we need to do is convert them into nifti format. 

When participating in the competition we just merged the two datasets into one nnU-Net task and trained for them 
together. This could have been a good idea or a horrible one. This was not evaluated (time was short and we were young).

## Dataset conversion
Open [Task075_Fluo_C3DH_A549_ManAndSim.py](../../nnunet/dataset_conversion/Task075_Fluo_C3DH_A549_ManAndSim.py), 
adapt the paths in it and execute it with python.

Note that this script will create a custom cross-validation split so that we stratify properly. Again we're not 
sure whether this is necessary. Just roll with it.

Now run planning and preprocessing with 
`nnUNet_plan_and_preprocess -t 75 -pl2d None` (`-pl2d None` disables the 2d configuration which is not needed here!)

## Training
You can now execute nnU-Net training:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 75 0
nnUNet_train 3d_fullres nnUNetTrainerV2 75 1
nnUNet_train 3d_fullres nnUNetTrainerV2 75 2
nnUNet_train 3d_fullres nnUNetTrainerV2 75 3
```

## Inference

Use these 4 models as an ensemble for test set prediction (`nnUNet_predict [...] -f 0 1 2 3`). You can use the 
imagesTs folder that was created as part of the dataset converion. After that you need to 
convert the predicted nifti images back to tiff. 

Best would probably be to just use our [inference code](http://celltrackingchallenge.net/participants/DKFZ-GE/) 
(with the pretrained weights). Or at least use it as an inspiration.

# Fluo-N2DH-SIM
This dataset is an instance segmentation problem. But nnU-Net can only do semantic segmentation. Dang. That's it. Bye.

...

Nah just kidding. We use an age old trick (which we didnt invent) and convert the instance segmentation task into a 
semantic segmentation task. Semantic classes are 'cell center' and 'cell border'. This will allow us to convert back
to instance segmentation when we are done.

Ah and this dataset is also a time series and because it may be hard to distinguish cell instances in 2D without time 
information we hack some of that good stuff in there. Essentially we stack the previous images of the frame of interest 
in the color channel. So the input to the model is [t-4, t-3, t-2, t-1, frame_of_interest]. Stacking in color channels
is probably not very effective but remember that we are just applying nnU-Net here and cannot change the method 
(it's supposed to be out-of-the-box ;-) ).

## Dataset conversion

Open the file [Task089_Fluo-N2DH-SIM.py](../../nnunet/dataset_conversion/Task089_Fluo-N2DH-SIM.py) and 
modify the paths. Then execute it with python. This will convert the raw dataset.

Then run `nnUNet_plan_and_preprocess -t 89 -pl3d None` (`-pl3d None` because this is a 2D dataset and we don't need 
the 3d configurations).

## Training
You can now execute nnU-Net training:
```bash
nnUNet_train 2d nnUNetTrainerV2 89 all
```

This just trains a single model on all available training cases. No ensembling here (maybe we should have!?).

## Inference
You can use the images in `imagesTs` but honestly just use the code we provide for inference 
[here](http://celltrackingchallenge.net/participants/DKFZ-GE/). If you choose to run inference all by yourself, 
remember to specify the 'all' fold correctly (`nnUNet_predict [...] -f all`).

# Fluo-N3DH-SIM+
Instance segmentation just like Fluo-N2DH-SIM so we convert the instances into a two-call semantic segmentation 
problem (cell border & center) so that we can solve it with nnU-Net. No messing with time information because 3D is 
[noice](https://thumbs.gfycat.com/CoordinatedEnergeticChipmunk-size_restricted.gif)! 

## Dataset conversion
Open the file [Task076_Fluo_N3DH_SIM.py](../../nnunet/dataset_conversion/Task076_Fluo_N3DH_SIM.py) and 
modify the paths. Then execute it with python. This will convert the raw dataset.

Then run `nnUNet_plan_and_preprocess -t 76 -pl2d None` (`-pl2d None` because this is a 3D dataset and we don't need the 
2d configurations).

## Training
You can now execute nnU-Net training:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2 76 all
```

## Inference
You can use the images in `imagesTs` but honestly just use the code we provide for inference 
[here](http://celltrackingchallenge.net/participants/DKFZ-GE/). If you choose to run inference all by yourself, 
remember to specify the 'all' fold correctly (`nnUNet_predict [...] -f all`).
