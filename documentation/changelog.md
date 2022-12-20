[v2]

Changes
- supports more input/output data formats through ImageIO classes
- no longer requires cropped data -> less storage and easier dataset updating
- everything is more modular and easier to change (resampling, normalization, ...)
- preprocessed data and seg are stored in different files (allows for different dtypes, less I/O)
- nnUNet_preprocessed subdir is named after configuration->data_identifier which defaults to the name of the 
configuration itself (multiple configurations can point to the same data (useful for experimenting with patch 
and batch sizes), it is the users responsibility not to overwrite existing data with something else). 
Any number of configurations can be stored in plans file and selected by name
