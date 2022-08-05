from setuptools import setup, find_namespace_packages

setup(name='nnunetv2',
      packages=find_namespace_packages(include=["nnunetv2", "nnunetv2.*"]),
      version='2',
      description='nnU-Net. Framework for out-of-the box biomedical image segmentation.',
      url='https://github.com/MIC-DKFZ/nnUNet',
      author='Helmholtz Imaging Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center',
      author_email='f.isensee@dkfz-heidelberg.de',
      license='Apache License Version 2.0, January 2004',
      install_requires=[
            "torch>=1.8.0a",
            "tqdm",
            "dicom2nifti",
            "scikit-image>=0.14",
            "medpy",
            "scipy",
            "batchgenerators>=0.22",
            "numpy",
            "sklearn",
            "SimpleITK",
            "pandas",
            "graphviz",
            'tifffile',
            'requests',
            "nibabel",
            "pytorch_lightning",
      ],
      entry_points={
          'console_scripts': [
              'nnUNetv2_plan_and_preprocess = nnunetv2.experiment_planning.plan_and_preprocess:plan_and_preprocess',
              'nnUNetv2_extract_fingerprint = nnunetv2.experiment_planning.plan_and_preprocess:extract_fingerprint',
              'nnUNetv2_plan_experiment = nnunetv2.experiment_planning.plan_and_preprocess:plan_experiment',
              'nnUNetv2_preprocess = nnunetv2.experiment_planning.plan_and_preprocess:preprocess',
              'nnUNetv2_train = nnunetv2.run.train_nolightning:nnUNet_train_from_args',
              'nnUNetv2_predict_from_modelfolder = nnunetv2.inference.predict_from_raw_data:predict_entry_point_modelfolder',
              'nnUNetv2_predict = nnunetv2.inference.predict_from_raw_data:predict_entry_point',
              'nnUNetv2_convert_old_nnUNet_dataset = nnunetv2.dataset_conversion.convert_raw_dataset_from_old_nnunet_format:convert_entry_point'
          ],
      },
      keywords=['deep learning', 'image segmentation', 'medical image analysis',
                'medical image segmentation', 'nnU-Net', 'nnunet']
      )
