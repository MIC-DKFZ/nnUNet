What do experiment planners need to do (these are notes for myself while rewriting nnU-Net, thay are provided as is 
without further explanations. These notes also include new features):
- preprocessor name should be configurable via cli
- gpu memory target should be configurable via cli
- plans name should be configurable via cli
- data name should be tied to plans name (plans still specify the data they want to use, this will allow us to manually 
  edit plans files without having to copy the data folders)
- plans must contain:
    - transpose forward/backward
    - preprocessor name
    - spacing
    - normalization scheme
    - target spacing
    - conv and pool op kernel sizes
    - base num features for architecture
    - data identifier
    - num conv per stage?
    - use mask for norm
    - num segmentation outputs
    - ignore class
    - list of regions or classes
    - regions class order, if applicable
    - resampling function to be used
    - the image reader writer class that should be used
