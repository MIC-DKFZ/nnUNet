Changes
- supports more input/output data formats through ImageIO classes
- no longer requires cropped data -> less storage and easier dataset updating
- extracting of training data is now multiprocess-safe (ToDo)
- everything is more modular and easier to change (resampling, normalization, ...)