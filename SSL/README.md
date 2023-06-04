# DeSD-code
This is the official pytorch implementation of our MICCAI 2022 paper "DeSD: Self-Supervised Learning with Deep Self-Distillation for 3D Medical Image Segmentation". In this paper, we reformulate SSL in a Deep Self-Distillation (DeSD) manner to improve the representation quality of both shallow and deep layers. 

This paper is available [here](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_52).


<div align="center">
  <img width="80%" alt="DeSD illustration" src="github/Overview.png">
</div>

## Requirements
CUDA 10.1<br />
Python 3.6<br /> 
Pytorch 1.7.1<br /> 
Torchvision 0.8.2<br />


## Usage

### Installation
* Clone this repo.
```
git clone https://github.com/yeerwen/DeSD.git
cd DeSD
```

### Data Preparation
* Download [DeepLesion dataset](https://nihcc.app.box.com/v/DeepLesion).

### Pre-processing
* Run `DL_save_nifti.py` (from downloaded files) to transfer the PNG image to the nii.gz form.
* Run `re_spacing_ITK.py` to resample CT volumes.
* Run `splitting_to_patches.py` to extract about 125k sub-volumes, and the pre-processed dataset will be saved in `DL_patches_v2/`.

### Training 
* Run `sh run_ssl.sh` for self-supervised pre-training.

### Pre-trained Model
* Pre-trained model is available in [DeSD_Res50](https://drive.google.com/file/d/1NoTPb3n276ZKMlTX0Cpch4E4PeDDW7OH/view?usp=sharing).


### Fine-tune DeSD on your own target task

As for the target segmentation tasks, the 3D model can be initialized with the pre-trained encoder using the following example:
```python
import torch
from torch import nn
# build a 3D segmentation model based on resnet50
class ResNet50_Decoder(nn.Module):
    def __init__(self, Resnet50_encoder, skip_connection, n_class=1, pre_training=True, load_path=None):
        super(ResNet50_Decoder, self).__init__()

        self.encoder = Resnet50_encoder
        self.decoder = Decoder(skip_connection)
        self.seg_head = nn.Conv3d(n_class, kernel_size=1)
        
        if pre_training:
            print('loading from checkpoint ssl: {}'.format(load_path))
            w_before = self.encoder.state_dict()['conv1.weight'].mean()
            pre_dict = torch.load(load_path, map_location='cpu')['teacher']
            pre_dict = {k.replace("module.backbone.", ""): v for k, v in pre_dict.items()}
            # print(pre_dict)
            model_dict = self.encoder.state_dict()
            pre_dict_update = {k:v for k, v in pre_dict.items() if k in model_dict}
            print("[pre_%d/mod_%d]: %d shared layers" % (len(pre_dict), len(model_dict), len(pre_dict_update)))
            model_dict.update(pre_dict_update)
            self.encoder.load_state_dict(model_dict)
            w_after = self.encoder.state_dict()['conv1.weight'].mean()
            print("one-layer before/after: [%.8f, %.8f]" % (w_before, w_after))
        else:
            print("TFS!")

    def forward(self, input):
        outs = self.encoder(input)
        decoder_out = self.deocder(outs)
        out = self.seg_head(decoder_out)
        return out
```


### Citation
If this code is helpful for your study, please cite:

```
@article{DeSD,
  title={DeSD: Self-Supervised Learning with Deep Self-Distillation for 3D Medical Image Segmentation},
  author={Yiwen Ye, Jianpeng Zhang, Ziyang Chen, and Yong Xia},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022},
  pages={545--555},
  year={2022}
}
```

### Acknowledgements
Part of codes is reused from the [DINO](https://github.com/facebookresearch/dino). Thanks to Caron et al. for the codes of DINO.

### Contact
Yiwen Ye (ywye@mail.nwpu.edu.cn)




# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Misc functions.

Mostly copy-paste from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""