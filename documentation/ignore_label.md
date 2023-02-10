# Ignore Label

The _ignore label_ can be used to mark regions that should be ignored by nnU-Net. No gradients are propagated from 
pixels marked with ignore label and these pixels will also be excluded from model evaluation. The two most common 
use-cases for the ignore label are sparse annotations (labeling just a subset of slices, scribble annotations etc) or 
areas within the image that should be ignored for other reasons.

The most useful application of the ignore label certainly (according to us) is sparse data annotation. To drive home 
this point, we conducted an experiment on the [AMOS2022 dataset](https://amos22.grand-challenge.org/)

<img src="assets/sparse_annotation_amos.png" width="768px" />