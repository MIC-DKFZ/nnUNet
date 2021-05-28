from setuptools import setup, find_namespace_packages

setup(name='nnunet_inference_on_cpu_and_gpu',
      packages=find_namespace_packages(include=["nnunet", "nnunet.*"]),
      version='1.6.6',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation. Can do inference on both gpu(if cuda available) and cpu(if cuda not available)',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>=1.6.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.21",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "requests",
            "nibabel", 'tifffile'
      ],
      entry_points={
          'console_scripts': [
              'nnUNet_convert_decathlon_task = nnunet.experiment_planning.nnUNet_convert_decathlon_task:main',
              'nnUNet_plan_and_preprocess = nnunet.experiment_planning.nnUNet_plan_and_preprocess:main',
              'nnUNet_train = nnunet.run.run_training:main',
              'nnUNet_train_DP = nnunet.run.run_training_DP:main',
              'nnUNet_train_DDP = nnunet.run.run_training_DDP:main',
              'nnUNet_predict = nnunet.inference.predict_simple:main',
              'nnUNet_ensemble = nnunet.inference.ensemble_predictions:main',
              'nnUNet_find_best_configuration = nnunet.evaluation.model_selection.figure_out_what_to_submit:main',
              'nnUNet_print_available_pretrained_models = nnunet.inference.pretrained_models.download_pretrained_model:print_available_pretrained_models',
              'nnUNet_print_pretrained_model_info = nnunet.inference.pretrained_models.download_pretrained_model:print_pretrained_model_requirements',
              'nnUNet_download_pretrained_model = nnunet.inference.pretrained_models.download_pretrained_model:download_by_name',
              'nnUNet_download_pretrained_model_by_url = nnunet.inference.pretrained_models.download_pretrained_model:download_by_url',
              'nnUNet_determine_postprocessing = nnunet.postprocessing.consolidate_postprocessing_simple:main',
              'nnUNet_export_model_to_zip = nnunet.inference.pretrained_models.collect_pretrained_models:export_entry_point',
              'nnUNet_install_pretrained_model_from_zip = nnunet.inference.pretrained_models.download_pretrained_model:install_from_zip_entry_point',
              'nnUNet_change_trainer_class = nnunet.inference.change_trainer:main',
              'nnUNet_evaluate_folder = nnunet.evaluation.evaluator:nnunet_evaluate_folder'
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet']
      )
