if __name__ == '__main__':
    from nnunetv2.paths import nnUNet_results, nnUNet_raw
    import torch
    from batchgenerators.utilities.file_and_folder_operations import join
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

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

    # variant 2, use list of files as inputs. Note how we use nested lists!!!
    indir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs')
    outdir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs_predlowres')
    predictor.predict_from_files([[join(indir, 'liver_152_0000.nii.gz')],
                                  [join(indir, 'liver_142_0000.nii.gz')]],
                                 [join(outdir, 'liver_152.nii.gz'),
                                  join(outdir, 'liver_142.nii.gz')],
                                 save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)

    # variant 2.5, returns segmentations
    indir = join(nnUNet_raw, 'Dataset003_Liver/imagesTs')
    predicted_segmentations = predictor.predict_from_files([[join(indir, 'liver_152_0000.nii.gz')],
                                                            [join(indir, 'liver_142_0000.nii.gz')]],
                                                           None,
                                                           save_probabilities=False, overwrite=True,
                                                           num_processes_preprocessing=2,
                                                           num_processes_segmentation_export=2,
                                                           folder_with_segs_from_prev_stage=None, num_parts=1,
                                                           part_id=0)

    # predict several npy images
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

    # predict a single numpy array
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
    ret = predictor.predict_single_npy_array(img, props, None, None, False)

    # custom iterator

    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_147_0000.nii.gz')])
    img2, props2 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_146_0000.nii.gz')])
    img3, props3 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_145_0000.nii.gz')])
    img4, props4 = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset003_Liver/imagesTs/liver_144_0000.nii.gz')])


    # each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properites' keys!
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
            yield {'data': torch.from_numpy(data).contiguous().pin_memory(), 'data_properites': p, 'ofile': None}


    ret = predictor.predict_from_data_iterator(my_iterator([img, img2, img3, img4], [props, props2, props3, props4]),
                                               save_probabilities=False, num_processes_segmentation_export=3)
