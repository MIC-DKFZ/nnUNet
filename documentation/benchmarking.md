# nnU-Netv2 benchmarks

Does your system run like it should? Is your epoch time longer than expected? What epoch times should you expect?

Look no further for we have the solution here!

## What does the nnU-netv2 benchmark do?

nnU-Net's benchmark trains models for 5 epochs. At the end, the fastest epoch will 
be noted down, along with the GPU name, torch version and cudnn version. You can find the benchmark output in the 
corresponding nnUNet_results subfolder (see example below). Don't worry, we also provide scripts to collect your 
results. Or you just start a benchmark and look at the console output. Everything is possible. Nothing is forbidden.

The benchmark implementation revolves around two trainers:
- `nnUNetTrainerBenchmark_5epochs` runs a regular training for 5 epochs. When completed, writes a .json file with the fastest 
epoch time as well as the GPU used and the torch and cudnn versions. Useful for speed testing the entire pipeline 
(data loading, augmentation, GPU training)
- `nnUNetTrainerBenchmark_5epochs_noDataLoading` is the same, but it doesn't do any data loading or augmentation. It 
just presents dummy arrays to the GPU. Useful for checking pure GPU speed.

## How to run the nnU-Netv2 benchmark?
It's quite simple, actually. It looks just like a regular nnU-Net training.

We provide reference numbers for some of the Medical Segmentation Decathlon datasets because they are easily 
accessible: [download here](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2). If it needs to be 
quick and dirty, focus on Tasks 2 and 4. Download and extract the data and convert them to the nnU-Net format with 
`nnUNetv2_convert_MSD_dataset`. 
Run `nnUNetv2_plan_and_preprocess` for them.

Then, for each dataset, run the following commands (only one per GPU! Or one after the other):

```bash
nnUNetv2_train DATSET_ID 2d 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train DATSET_ID 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs
nnUNetv2_train DATSET_ID 2d 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading
nnUNetv2_train DATSET_ID 3d_fullres 0 -tr nnUNetTrainerBenchmark_5epochs_noDataLoading
```

If you want to inspect the outcome manually, check (for example!) your 
`nnUNet_results/DATASET_NAME/nnUNetTrainerBenchmark_5epochs__nnUNetPlans__3d_fullres/fold_0/` folder for the `benchmark_result.json` file.

Note that there can be multiple entries in this file if the benchmark was run on different GPU types, torch versions or cudnn versions!

If you want to summarize your results like we did in our [results](#results), check the 
[summary script](../nnunetv2/batch_running/benchmarking/summarize_benchmark_results.py). Here you need to change the 
torch version, cudnn version and dataset you want to summarize, then execute the script. You can find the exact 
values you need to put there in one of your `benchmark_result.json` files.

## Results
We have tested a variety of GPUs and summarized the results in a 
[spreadsheet](https://docs.google.com/spreadsheets/d/12Cvt_gr8XU2qWaE0XJk5jJlxMEESPxyqW0CWbQhTNNY/edit?usp=sharing). 
Note that you can select the torch and cudnn versions at the bottom! There may be comments in this spreadsheet. Read them!

## Result interpretation

Results are shown as epoch time in seconds. Lower is better (duh). Epoch times can fluctuate between runs, so as 
long as you are within like 5-10% of the numbers we report, everything should be dandy. 

If not, here is how you can try to find the culprit!

The first thing to do is to compare the performance between the `nnUNetTrainerBenchmark_5epochs_noDataLoading` and 
`nnUNetTrainerBenchmark_5epochs` trainers. If the difference is about the same as we report in our spreadsheet, but 
both your numbers are worse, the problem is with your GPU:

- Are you certain you compare the correct GPU? (duh)
- If yes, then you might want to install PyTorch in a different way. Never `pip install torch`! Go to the
[PyTorch installation](https://pytorch.org/get-started/locally/) page, select the most recent cuda version your 
system supports and only then copy and execute the correct command! Either pip or conda should work
- If the problem is still not fixed, we recommend you try 
[compiling pytorch from source](https://github.com/pytorch/pytorch#from-source). It's more difficult but that's 
how we roll here at the DKFZ (at least the cool kids here).
- Another thing to consider is to try exactly the same torch + cudnn version as we did in our spreadsheet. 
Sometimes newer versions can actually degrade performance and there might be bugs from time to time. Older versions 
are also often a lot slower!
- Finally, some very basic things that could impact your GPU performance: 
  - Is the GPU cooled adequately? Check the temperature with `nvidia-smi`. Hot GPUs throttle performance in order to not self-destruct
  - Is your OS using the GPU for displaying your desktop at the same time? If so then you can expect a performance 
  penalty (I dunno like 10% !?). That's expected and OK.
  - Are other users using the GPU as well?


If you see a large performance difference between `nnUNetTrainerBenchmark_5epochs_noDataLoading` (fast) and 
`nnUNetTrainerBenchmark_5epochs` (slow) then the problem might be related to data loading and augmentation. As a 
reminder, nnU-net does not use pre-augmented images (offline augmentation) but instead generates augmented training 
samples on the fly during training (no, you cannot switch it to offline). This requires that your system can do partial 
reads of the image files fast enough (SSD storage required!) and that your CPU is powerful enough to run the augmentations.

Check the following:

- [CPU bottleneck] How many CPU threads are running during the training? nnU-Net uses 12 processes for data augmentation by default. 
If you see those 12 running constantly during training, consider increasing the number of processes used for data 
augmentation (provided there is headroom on your CPU!). Increase the number until you see less active workers than 
you configured (or just set the number to 32 and forget about it). You can do so by exporting the `nnUNet_n_proc_DA` environment variable:
  `export nnUNet_n_proc_DA=XX` followed by the training OR `nnUNet_n_proc_DA=XX nnUNetv2_train [...]`. If you want to 
make this permanent, add the environment variable to your .bashrc (Linux!). If your CPU does not support more 
processes (setting more processes than your CPU has threads makes 
no sense!) you are out of luck and in desperate need of a system upgrade!
- [I/O bottleneck] If you don't see 12 (or nnUNet_n_proc_DA if you set it) processes running but your training times 
are still slow then open up `top` (sorry, Windows users) and look at the value left of 'wa' in the row that begins 
with '%Cpu (s)'. If this is >1.0 (arbitrarily set threshold here, essentially look for unusually high 'wa'. In a 
healthy training 'wa' will be almost 0) then your storage cannot keep up with data loading. Make sure to set 
nnUNet_preprocessed to a folder that is located on an SSD. nvme is preferred over SATA. PCIe3 is enough. 3000MB/s 
sequential read recommended.
- [funky stuff] Sometimes there is funky stuff going on, especially when batch sizes are large, files are small and 
patch sizes are small as well. As part of the data loading process, nnU-Net needs to open and close a file for each 
training sample. Now imagine a dataset like Dataset004_Hippocampus where for the 2d config we have a batch size of 
366 and we run 250 iterations in <10s on an A100. That's a lotta files per second (366 * 250 / 10 = 9150 files per second). 
Oof. If the files are on some network drive (even if it's nvme) then (probably) good night. The good news: nnU-Net
has got you covered: add `export nnUNet_keep_files_open=True` to your .bashrc and the problem goes away. The neat 
part: it causes new problems if you are not allowed to have enough open files. You may have to increase the number 
of allowed open files. `ulimit -n` gives your current limit (Linux only). It should not be something like 1024. 
Increasing that to 65535 works well for me. See here for how to change these limits: 
[Link](https://kupczynski.info/posts/ubuntu-18-10-ulimits/) 
(works for Ubuntu 18, google for your OS!).

