# PhySU-Net
 This is the official code repository of our ICPR 2024 paper entitled "PhySU-Net: Long Temporal Context Transformer for rPPG with Self-Supervised Pre-training". PhySU-Net, is a novel remote photoplethysmography (rPPG) method designed to address the limitations of current approaches. We propose the first long temporal context rPPG transformer network, along with a self-supervised pre-training strategy that leverages unlabeled data. PhySU-Net achieves state-of-the-art performance on public benchmark datasets (OBF, VIPL-HR, and MMSE-HR), demonstrating improved robustness and accuracy in contactless cardiac activity measurement from facial videos. Our self-supervised pre-training, utilizing traditional methods and image masking to generate pseudo-labels, further enhances the model's performance by effectively utilizing unlabeled data.

 You can find our paper at https://link.springer.com/chapter/10.1007/978-3-031-78341-8_15

![METHODVIS](physu-net_method.jpg)
Method Overview: The input video is processed into a stacked MSTmap. For the supervised downstream task, the decoder reconstructs an image with similar temporal and frequency properties as the BVPmap label, and the HR head regresses the global HR value with the HR ground truth as label. For the self-supervised pretext task, only the input and the labels change. The input is a masked version of the MSTmap that the decoder attempts to reconstruct into a full MSTmap. For the HR regression, a CHROM calculated pseudo-label is used.
 
## Dataset Preprocessing

The original videos are firstly preprocessed by extracting the MSTmaps following https://github.com/nxsEdson/CVD-Physiological-Measurement. Both the MSTmaps and groundtruth bvp are resampled to 30 fps. In the example code we assume the data used in pre-processed from OBF, MMSE or VIPL-HR datasets, but can't provide the actual data or preprocessed files. You can find code for pre-preprocessing the data at https://github.com/marukosan93/RS-rPPG or https://github.com/marukosan93/ORPDAD/.

## Training
**CODE IS COMING SOON** <br>
You can SSL pre-train the network to predict MSTmaps from Masked maps and to regress pseudo-hr with pretrain_PU.py. To train using supervised learnig (or transfer if you load the previously pretrained model) use train_PU.py. eval_PU.py can evalute the output signals and calculate metrics. 

## Citation

@inproceedings{savic2024physu,
  title={PhySU-net: Long temporal context transformer for rPPG with self-supervised pre-training},
  author={Savic, Marko and Zhao, Guoying},
  booktitle={International Conference on Pattern Recognition},
  pages={228--243},
  year={2024},
  organization={Springer}
}
