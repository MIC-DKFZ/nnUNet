# Pretraining with nnU-Net

## Intro

So far nnU-Net only supports supervised pre-training, meaning that you train a regular nnU-Net on some pretraining dataset 
and then use the final network weights as initialization for your target dataset. 

As a reminder, many training hyperparameters such as patch size and network topology differ between datasets as a 
result of the automated dataset analysis and experiment planning nnU-Net is known for. So, out of the box, it is not 
possible to simply take the network weights from some dataset and then reuse them for another.

Consequently, the plans need to be aligned between the two tasks. In this README we show how this can be achieved and 
how the resulting weights can then be used for initialization.

### Terminology

Throughout this README we use the following terminology:

- `pretraining dataset` is the dataset you intend to run the pretraining on
- `finetuning dataset` is the dataset you are interested in; the one you wish to fine tune on


## Training on the pretraining dataset

In order to obtain matching network topologies we need to transfer the plans from one dataset to another. Since we are 
only interested in the finetuning dataset, we first need to run experiment planning (and preprocessing) for it:

```bash
nnUNetv2_plan_and_preprocess -d FINETUNING_DATASET
```

Then we need to extract the dataset fingerprint of the pretraining dataset, if not yet available:

```bash
nnUNetv2_extract_fingerprint -d PRETRAINING_DATASET
```

Now we can take the plans from the finetuning dataset and transfer it to the pretraining dataset:

```bash
nnUNetv2_move_plans_between_datasets -s FINETUNING_DATASET -t PRETRAINING_DATASET -sp FINETUNING_PLANS_IDENTIFIER -tp PRETRAINING_PLANS_IDENTIFIER
```

`FINETUNING_PLANS_IDENTIFIER` is hereby probably nnUNetPlans unless you changed the experiment planner in 
nnUNetv2_plan_and_preprocess. For `PRETRAINING_PLANS_IDENTIFIER` we recommend you set something custom in order to not 
overwrite default plans.

Note that EVERYTHING is transferred between the datasets. Not just the network topology, batch size and patch size but 
also the normalization scheme! Therefore, a transfer between datasets that use different normalization schemes may not 
work well (but it could, depending on the schemes!).

Note on CT normalization: Yes, also the clip values, mean and std are transferred!

Now you can run the preprocessing on the pretraining dataset:

```bash
nnUNetv2_preprocess -d PRETRAINING_DATASET -plans_name PRETRAINING_PLANS_IDENTIFIER
```

And run the training as usual:

```bash
nnUNetv2_train PRETRAINING_DATASET CONFIG all -p PRETRAINING_PLANS_IDENTIFIER
```

Note how we use the 'all' fold to train on all available data. For pretraining it does not make sense to split the data.

## Using pretrained weights

Once pretraining is completed (or you obtain compatible weights by other means) you can use them to initialize your model:

```bash
nnUNetv2_train FINETUNING_DATASET CONFIG FOLD -pretrained_weights PATH_TO_CHECKPOINT
```

Specify the checkpoint in PATH_TO_CHECKPOINT.

When loading pretrained weights, all layers except the segmentation layers will be used! 

So far there are no specific nnU-Net trainers for fine tuning, so the current recommendation is to just use 
nnUNetTrainer. You can however easily write your own trainers with learning rate ramp up, fine-tuning of segmentation 
heads or shorter training time.
