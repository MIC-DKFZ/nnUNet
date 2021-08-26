The modality entry in dataset.json only determines the normlaization scheme. So if you want to use something different 
then you can just
- create a new subclass of ImageNormalization
- map your custom modality identifier to that subclass in modality_to_normalization_mapping
- run plan and preprocess again with your custom normlaization scheme