What do experiment planners need to do (these are notes for myself while rewriting nnU-Net, they are provided as is 
without further explanations. These notes also include new features):
- (done) preprocessor name should be configurable via cli
- (done) gpu memory target should be configurable via cli
- (done) plans name should be configurable via cli
- (done) data name should be specified in plans (plans specify the data they want to use, this will allow us to manually 
  edit plans files without having to copy the data folders)
- plans must contain:
    - (done) transpose forward/backward
    - (done) preprocessor name (can differ for each config)
    - (done) spacing
    - (done) normalization scheme
    - (done) target spacing
    - (done) conv and pool op kernel sizes
    - (done) base num features for architecture
    - (done) data identifier
    - num conv per stage?
    - (done) use mask for norm
    - [NO. Handled by LabelManager & dataset.json] num segmentation outputs
    - [NO. Handled by LabelManager & dataset.json] ignore class
    - [NO. Handled by LabelManager & dataset.json] list of regions or classes
    - [NO. Handled by LabelManager & dataset.json] regions class order, if applicable
    - (done) resampling function to be used
    - (done) the image reader writer class that should be used


dataset.json
mandatory:
- numTraining
- labels (value 'ignore' has special meaning. Cannot have more than one ignore_label)
- modalities
- file_ending

optional
- overwrite_image_reader_writer (if absent, auto)
- regions
- region_class_order
- 