# Fluo-C3DH-A549 and Fluo-C3DH-A549-SIM

## Dataset conversion
These datasets are usable by nnU-Net as is, so all we need to do is convert them into nifti format. 
Open [Task075_Fluo_C3DH_A549_ManAndSim.py](../../nnunet/dataset_conversion/Task075_Fluo_C3DH_A549_ManAndSim.py), 
adapt the paths in it and execute it with python.

When participating in the competition we just merged the two datasets into one nnU-Net task and trained for them 
together. This could have been a good idea or a horrible one. This was not evaluated (time was short and we were young).

Note that this script will create a custom cross-validation split so that we stratify properly. Againg it's not 
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
to a 