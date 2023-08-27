- Derive your adapter from `BaseReaderWriter`. 
- Reimplement all abstractmethods. 
- make sure to support 2d and 3d input images (or raise some error).
- place it in this folder or nnU-Net won't find it!
- add it to LIST_OF_IO_CLASSES in `reader_writer_registry.py`

Bam, you're done!