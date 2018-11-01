# Triplet Loss SBIR
This repo contains code for the CVIU 2016 paper "[Compact Descriptors for Sketch-based Image Retrieval using a Triplet loss Convolutional Neural Network](https://doi.org/10.1016/j.cviu.2017.06.007)" 

## Pre-trained model
The pretrained model and datasets can be downloaded on our [project page](http://www.cvssp.org/data/Flickr25K/CVIU16.html).

## Feature extraction
We provide three scripts for extracting features from image/sketch:

[extract_feat_single.py](extract_feat_single.py) is an example for extracting features of a single image or sketch. Note that in the original paper, gPb is used as the edge extraction method for photographic images, here for simplicity we use Canny edge instead. Canny edge is fast but has lower performance (score 18% mAP on Flickr15K as opposed to 24.5% mAP as reported in the paper with gPb.)

[extract_feat_batch.py](extract_feat_batch.py) is essentially [extract_feat_single.py](extract_feat_single.py) but optimised for batch processing. Useful for extracting features from a large dataset.

[extract_feat_lmdb.py](extract_feat_lmdb.py) is similar to [extract_feat_batch.py](extract_feat_batch.py) but accept preprocessed sketches or image edgemaps in lmdb format. The preprocessed test lmdb (Flickr15K, downloadable on our project page) can be used with this script to reproduce the results (24.5% mAP) of the paper. 

## Reference
```
@article{bui2017compact,
title = {Compact descriptors for sketch-based image retrieval using a triplet loss convolutional neural network},
author = {Tu Bui and Leonardo Ribeiro and Moacir Ponti and John Collomosse},
journal = {Computer Vision and Image Understanding},
year = {2017},
volume={164},
pages={27--37},
issn = {1077-3142}
}
```
