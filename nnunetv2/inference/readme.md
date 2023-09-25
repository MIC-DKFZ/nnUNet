The nnU-Net inference is now much more dynamic than before, allowing you to more seamlessly integrate nnU-Net into 
your existing workflows.
This readme will give you a quick rundown of your options. This is not a complete guide. Look into the code to learn 
all the details!

# Preface
In terms of speed, the most efficient inference strategy is the one done by the nnU-Net defaults! Images are read on 
the fly and preprocessed in background workers. The main process takes the preprocessed images, predicts them and 
sends the prediction off to another set of background workers which will resize the resulting logits, convert 
them to a segmentation and export the segmentation.

The reason the default setup is the best option is because 

1) loading and preprocessing as well as segmentation export are interlaced with the prediction. The main process can 
focus on communicating with the compute device (i.e. your GPU) and does not have to do any other processing. 
This uses your resources as well as possible!
2) only the images and segmentation that are currently being needed are stored in RAM! Imaging predicting many images 
and having to store all of them + the results in your system memory

# nnUNetPredictor
The new nnUNetPredictor class encapsulates the inferencing code and makes it simple to switch between modes. Your 
code can hold a nnUNetPredictor instance and perform prediction on the fly. Previously this was not possible and each 
new prediction request resulted in reloading the parameters and reinstantiating the network architecture. Not ideal.

The nnUNetPredictor must be ininitialized manually! You will want to use the 
`predictor.initialize_from_trained_model_folder` function for 99% of use cases!

New feature: If you do not specify an output folder / output files then the predicted segmentations will be 
returned 


## Recommended nnU-Net default: predict from source files

tldr:
- loads images on the fly
- performs preprocessing in background workers
- main process focuses only on making predictions
- results are again given to background workers for resampling and (optional) export

pros:
- best suited for predicting a large number of images
- nicer to your RAM

cons:
- not ideal when single images are to be predicted 
- requires images to be present as files

Example:
```python
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    
    # instantiate the nnUNetPredictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_gpu=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset003_Liver/nnUNetTrainer__nnUNetPlans__3d_lowres'),
        use_folds=(0,),
        checkpoint_name='checkpoint_final.pth',
    )
    # variant 1: give input and output folders
    predictor.predict_from_files(join(nnUNet_raw, 'Dataset003_Liver/imagesTs'),
                                 join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres'),
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
```

Instead if giving input and output folders you can also give concrete files. If you give concrete files, there is no 
need for the _0000 suffix anymore! This can be useful in situations where you have no control over the filenames!
Remember that the files must be given as 'list of lists' where each entry in the outer list is a case to be predicted 
and the inner list contains all the files belonging to that case. There is just one file for datasets with just one 
input modality (such as CT) but may be more files for others (such as MRI where there is sometimes T1, T2, Flair etc). 
IMPORTANT: the order in wich the files for each case are given must match the order of the channels as defined in the 
dataset.json!

If you give files as input, you need to give individual output files as output!

```python
    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    indir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs')
    outdir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres')
    predictor.predict_from_files([[join(indir, 'liver_152_0000.nii.gz')], 
                                  [join(indir, 'liver_142_0000.nii.gz')]],
                                 [join(outdir, 'liver_152.nii.gz'),
                                  join(outdir, 'liver_142.nii.gz')],
                                 save_probabilities=False, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
```

Did you know? If you do not specify output files, the predicted segmentations will be returned:
```python
    # variant 2.5, returns segmentations
    indir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs')
    predicted_segmentations = predictor.predict_from_files([[join(indir, 'liver_152_0000.nii.gz')],
                                  [join(indir, 'liver_142_0000.nii.gz')]],
                                 None,
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)
```

## Prediction from npy arrays
tldr:
- you give images as a list of npy arrays
- performs preprocessing in background workers
- main process focuses only on making predictions
- results are again given to background workers for resampling and (optional) export

pros:
- the correct variant for when you have images in RAM already
- well suited for predicting multiple images

cons:
- uses more ram than the default
- unsuited for large number of images as all images must be held in RAM

```python
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
    img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_146_0000.nii.gz')])
    img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_145_0000.nii.gz')])
    img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_144_0000.nii.gz')])
    # we do not set output files so that the segmentations will be returned. You can of course also specify output
    # files instead (no return value on that case)
    ret = predictor.predict_from_list_of_npy_arrays([img, img2, img3, img4],
                                                    None,
                                                    [props, props2, props3, props4],
                                                    None, 2, save_probabilities=False,
                                                    num_processes_segmentation_export=2)
```

## Predicting a single npy array

tldr:
- you give one image as npy array
- everything is done in the main process: preprocessing, prediction, resampling, (export)
- no interlacing, slowest variant!
- ONLY USE THIS IF YOU CANNOT GIVE NNUNET MULTIPLE IMAGES AT ONCE FOR SOME REASON

pros:
- no messing with multiprocessing
- no messing with data iterator blabla

cons:
- slows as heck, yo
- never the right choice unless you can only give a single image at a time to nnU-Net

```python
    # predict a single numpy array
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTr/liver_63_0000.nii.gz')])
    ret = predictor.predict_single_npy_array(img, props, None, None, False)
```

## Predicting with a custom data iterator
tldr: 
- highly flexible
- not for newbies

pros:
- you can do everything yourself
- you have all the freedom you want
- really fast if you remember to use multiprocessing in your iterator

cons:
- you need to do everything yourself
- harder than you might think

```python
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
    img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_146_0000.nii.gz')])
    img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_145_0000.nii.gz')])
    img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_144_0000.nii.gz')])
    # each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
    # If 'ofile' is None, the result will be returned instead of written to a file
    # the iterator is responsible for performing the correct preprocessing!
    # note how the iterator here does not use multiprocessing -> preprocessing will be done in the main thread!
    # take a look at the default iterators for predict_from_files and predict_from_list_of_npy_arrays
    # (they both use predictor.predict_from_data_iterator) for inspiration!
    def my_iterator(list_of_input_arrs, list_of_input_props):
        preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
        for a, p in zip(list_of_input_arrs, list_of_input_props):
            data, seg = preprocessor.run_case_npy(a,
                                                  None,
                                                  p,
                                                  predictor.plans_manager,
                                                  predictor.configuration_manager,
                                                  predictor.dataset_json)
            yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properties': p, 'ofile': None}
    ret = predictor.predict_from_data_iterator(my_iterator([img, img2, img3, img4], [props, props2, props3, props4]),
                                               save_probabilities=False, num_processes_segmentation_export=3)
```