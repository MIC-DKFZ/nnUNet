# How to generate custom splits in nnU-Net

Sometimes, the default 5-fold cross-validation split by nnU-Net does not fit a project. Maybe you want to run 3-fold 
cross-validation instead? Or maybe your training cases cannot be split randomly and require careful stratification. 
Fear not, for nnU-Net has got you covered (it really can do anything <3).

The splits nnU-Net uses are generated in the `do_split` function of nnUNetTrainer. This function will first look for 
existing splits, stored as a file, and if no split exists it will create one. So if you wish to influence the split, 
manually creating a split file that will then be recognized and used is the way to go!

The split file is located in the `nnUNet_preprocessed/DATASETXXX_NAME` folder. So it is best practice to first 
populate this folder by running `nnUNetv2_plan_and_preproccess`.

Splits are stored as a .json file. They are a simple python list. The length of that list is the number of splits it 
contains (so it's 5 in the default nnU-Net). Each list entry is a dictionary with keys 'train' and 'val'. Values are 
again simply lists with the train identifiers in each set. To illustrate this, I am just messing with the Dataset002 
file as an example:

```commandline
In [1]: from batchgenerators.utilities.file_and_folder_operations import load_json

In [2]: splits = load_json('splits_final.json')

In [3]: len(splits)
Out[3]: 5

In [4]: splits[0].keys()
Out[4]: dict_keys(['train', 'val'])

In [5]: len(splits[0]['train'])
Out[5]: 16

In [6]: len(splits[0]['val'])
Out[6]: 4

In [7]: print(splits[0])
{'train': ['la_003', 'la_004', 'la_005', 'la_009', 'la_010', 'la_011', 'la_014', 'la_017', 'la_018', 'la_019', 'la_020', 'la_022', 'la_023', 'la_026', 'la_029', 'la_030'],
'val': ['la_007', 'la_016', 'la_021', 'la_024']}
```

If you are still not sure what splits are supposed to look like, simply download some reference dataset from the
[Medical Decathlon](http://medicaldecathlon.com/), start some training (to generate the splits) and manually inspect 
the .json file with your text editor of choice!

In order to generate your custom splits, all you need to do is reproduce the data structure explained above and save it as 
`splits_final.json` in the `nnUNet_preprocessed/DATASETXXX_NAME` folder. Then use `nnUNetv2_train` etc. as usual.