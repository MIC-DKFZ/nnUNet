# Common Issues and their Solutions

- [Common Issues and their Solutions](#common-issues-and-their-solutions)
  * [RuntimeError: Expected scalar type half but found float](#runtimeerror--expected-scalar-type-half-but-found-float)
  * [nnU-Net gets 'stuck' during preprocessing, training or inference](#nnu-net-gets--stuck--during-preprocessing--training-or-inference)
  * [nnU-Net training: RuntimeError: CUDA out of memory](#nnu-net-training--runtimeerror--cuda-out-of-memory)
  * [nnU-Net training in Docker container: RuntimeError: unable to write to file </torch_781_2606105346>](#nnu-net-training-in-docker-container--runtimeerror--unable-to-write-to-file---torch-781-2606105346-)
  * [Downloading pretrained models: unzip: cannot find zipfile directory in one of /home/isensee/.nnunetdownload_16031094034174126](#downloading-pretrained-models--unzip--cannot-find-zipfile-directory-in-one-of--home-isensee-nnunetdownload-16031094034174126)
  * [Downloading pre-trained models: `unzip: 'unzip' is not recognized as an internal or external command` OR `Command 'unzip' not found`](#downloading-pre-trained-models---unzip---unzip--is-not-recognized-as-an-internal-or-external-command--or--command--unzip--not-found-)
  * [nnU-Net training (2D U-Net): High (and increasing) system RAM usage, OOM](#nnu-net-training--2d-u-net---high--and-increasing--system-ram-usage--oom)
  * [nnU-Net training of cascade: Error `seg from prev stage missing`](#nnu-net-training-of-cascade--error--seg-from-prev-stage-missing-)
  * [nnU-Net training: `RuntimeError: CUDA error: device-side assert triggered`](#nnu-net-training---runtimeerror--cuda-error--device-side-assert-triggered-)
  * [nnU-Net training: Error: mmap length is greater than file size and EOFError](#nnu-net-training--error--mmap-length-is-greater-than-file-size-and-eoferror)
  * [running nnU-Net on Azure instances](#running-nnu-net-on-azure-instances)

## RuntimeError: Expected scalar type half but found float

This can happen when running inference (or training) with mixed precision enabled on older GPU hardware. It points 
to some operations not being implemented in half precision for the type of GPU you are using. There are flags to enforce
 the use of fp32 for both nnUNet_predict and nnUNet_train. If you run into this error, using these flags will probably 
 solve it. See `nnUNet_predict -h` and `nnUNet_train -h` for what the flags are.

## nnU-Net gets 'stuck' during preprocessing, training or inference
nnU-Net uses python multiprocessing to leverage multiple CPU cores during preprocessing, background workers for data 
augmentation in training, preprocessing of cases during inference as well as resampling and exporting the final 
predictions during validation and inference. Unfortunately, python (or maybe it is just me as a programmer) is not 
very good at communicating errors that happen in background workers, causing the main process to indefinitely wait for 
them to return indefinitely.

Whenever nnU-Net appears to be stuck, this is what you should do:

1) There is almost always an error message which will give you an indication of what the problem is. This error message 
is often not at the bottom of the text output, but further up. If you run nnU-Net on a GPU cluster (like we do) the 
error message may be WAYYYY off in the log file, sometimes at the very start of the training/inference. Locate the 
error message (if necessary copy the stdout to a text editor and search for 'error')

2) If there is no error message, this could mean that your OS silently killed a background worker because it was about 
to go out of memory. In this case, please rerun whatever command you have been running and closely monitor your system 
RAM (not GPU memory!) usage. If your RAM is full or close to full, you need to take action:
   - reduce the number of background workers: use `-tl` and `-tf` in `nnUNet_plan_and_preprocess` (you may have to 
   go as low as 1!). Reduce the number of workers used by `nnUNet_predict` by reducing `--num_threads_preprocessing` and 
   `--num_threads_nifti_save`.
   - If even `-tf 1` during preprocessing is not low enough, consider adding a swap partition located on an SSD.
   - upgrade your RAM! (32 GB should get the job done)


## nnU-Net training: RuntimeError: CUDA out of memory

This section is dealing with error messages such as this:

```
RuntimeError: CUDA out of memory. Tried to allocate 4.16 GiB (GPU 0; 10.76 GiB total capacity; 2.82 GiB already allocated; 4.18 GiB free; 4.33 GiB reserved in total by PyTorch)
```

This message appears when the GPU memory is insufficient. For most datasets, nnU-Net uses about 8GB of video memory. 
To ensure that you can run all trainings, we recommend to use a GPU with at least 11GB (this will have some headroom).
If you are running other programs on the GPU you intend to train on (for example the GUI of your operating system), 
the amount of VRAM available to nnU-Net is less than whatever is on your GPU. Please close all unnecessary programs or 
invest in a second GPU. We for example like to use a low cost GPU (GTX 1050 or slower) for the display outputs while 
having the 2080ti (or equivalant) handle the training.

At the start of each training, cuDNN will run some benchmarks in order to figure out the fastest convolution algorithm 
for the current network architecture (we use `torch.backends.cudnn.benchmark=True`). VRAM consumption will jump all over
the place while these benchmarks run and can briefly exceed the 8GB nnU-Net typically requires. If you keep running into
 `RuntimeError: CUDA out of memory` problems you may want to consider disabling that. You can do so by setting the 
 `--deterministic` flag when using `nnUNet_train`. Setting this flag can slow down your training, so it is recommended 
 to only use it if necessary.
 
## nnU-Net training in Docker container: RuntimeError: unable to write to file </torch_781_2606105346>

Nvidia NGC (https://ngc.nvidia.com/catalog/containers/nvidia:pytorch) is a great place to find Docker containers with 
the most recent software (pytorch, cuDNN, etc.) in them. When starting Docker containers with command provided on the 
Nvidia website, the docker will crash with errors like this when running nnU-Net: `RuntimeError: unable to write to 
file </torch_781_2606105346>`. Please start the docker with the `--ipc=host` flag to solve this.

## Downloading pretrained models: unzip: cannot find zipfile directory in one of /home/isensee/.nnunetdownload_16031094034174126

Sometimes downloading the large zip files containing our pretrained models can fail and cause the error above. Please 
make sure to use the most recent nnU-Net version (we constantly try to improve the downloading). If that does not fix it
you can always download the zip file from our zenodo (https://zenodo.org/record/4003545) and use the 
`nnUNet_install_pretrained_model_from_zip` command to install the model.

## Downloading pre-trained models: `unzip: 'unzip' is not recognized as an internal or external command` OR `Command 'unzip' not found`

On Windows systems and on a bare WSL2 system, the `unzip` command may not be present.
Either install it, unzip the pre-trained model from zenodo download, or update to a newer version of nnUNet that uses the Python build in
(https://docs.python.org/3/library/zipfile.html)

## nnU-Net training (2D U-Net): High (and increasing) system RAM usage, OOM

There was a issue with mixed precision causing a system RAM memory leak. This is fixed when using cuDNN 8.0.2 or newer, 
but the current pytorch master comes with cuDNN 7.6.5. If you encounter this problem, please consider using Nvidias NGC 
pytorch container for training (the pytorch it comes with has a recent cuDNN version). You can also install the new 
cuDNN version on your system and compile pytorch yourself (instructions on the pytorch website!). This is what we do at DKFZ.


## nnU-Net training of cascade: Error `seg from prev stage missing` 
You need to run all five folds of `3d_lowres`. Segmentations of the previous stage can only be generated from the 
validation set, otherwise we would overfit.

## nnU-Net training: `RuntimeError: CUDA error: device-side assert triggered`
This error often goes along with something like `void THCudaTensor_scatterFillKernel(TensorInfo<Real, IndexType>, 
TensorInfo<long, IndexType>, Real, int, IndexType) [with IndexType = unsigned int, Real = float, Dims = -1]: 
block: [4770,0,0], thread: [374,0,0] Assertion indexValue >= 0 && indexValue < tensor.sizes[dim] failed.`.

This means that your dataset contains unexpected values in the segmentations. nnU-Net expects all labels to be 
consecutive integers. So if your dataset has 4 classes (background and three foreground labels), then the labels 
must be 0, 1, 2, 3 (where 0 must be background!). There cannot be any other values in the ground truth segmentations.

If you run `nnUNet_plan_and_preprocess` with the `--verify_dataset_integrity` option, this should never happen because 
it will check for wrong values in the label images.

## nnU-Net training: Error: mmap length is greater than file size and EOFError
Please delete all .npy files in the nnUNet_preprocessed folder of the test you were trying to train. Then try again.

## nnU-Net training: Error: MultiThreadedAugmenter.abort_event was set, something went wrong. Maybe one of your workers crashed. This is not the actual error message! Look further up your stdout to see what caused the error. Please also check whether your RAM was full
Please delete all .npy files in the nnUNet_preprocessed folder of the test you were trying to train. Then try again.

## running nnU-Net on Azure instances
see https://github.com/MIC-DKFZ/nnUNet/issues/437, thank you @Alaska47
