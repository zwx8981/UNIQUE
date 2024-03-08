# New! IQA-PyTorch implementation
[**IQA-PyTorch**](https://github.com/chaofengc/IQA-PyTorch) supports UNIQUE now! Can be easily used as follows:

```bash
import pyiqa
model = pyiqa.create_metric('unique', as_loss=False)
score = model(img_path)
```

# UNIQUE
The codebase for  
[Uncertainty-aware blind image quality assessment in the laboratory and wild](https://arxiv.org/pdf/2005.13983.pdf) (TIP2021) 
and  
[Learning to blindly assess image quality in the laboratory and wild](https://arxiv.org/pdf/1907.00516.pdf) (ICIP2020)  

![image](https://github.com/zwx8981/UNIQUE/blob/master/UNIQUE_framework.png)

# Prerequisite:
Python 3+  
PyTorch 1.4+  
Matlab  
Successfully tested on Ubuntu18.04, other OS (i.e., other Linux distributions, Windows) should also be ok.

# Usage
## Sampling image pairs from multiple databases
data_all.m  
## Combining the sampled pairs to form the training set
combine_train.m  
## Training on multiple databases for 10 sessions
```
python Main.py --train True --network basecnn --representation BCNN --ranking True --fidelity True --std_modeling True --std_loss True --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --max_epochs2 12 
```
(As for ICIP version, set std_loss to False and sample pairs from TID2013 instead of KADID-10K.)
(For training with binary labels, set fidelity and std_modeling to False.)
## Output predicted quality scores and stds
```
python Main.py --train False --get_scores True
```
## Result analysis
Compute SRCC/PLCC after nonlinear mapping: result_analysis.m  
Compute fidelity loss: eval_fidelity.m

# Pre-trained weights
Google Drive: https://drive.google.com/file/d/18oPH4lALm8mSdZh3fWK97MVq9w3BbEua/view?usp=sharing

Baidu: https://pan.baidu.com/s/1KKncQIoQcbxj7fQlSKUBIQ   code：yyev 

# A basic demo that predict the quality of single images.
```
python demo.py  
```
## Very important ! Make sure that the model has been appropriately set to eval mode !

# Link to download the BID dataset
The BID dataset may be difficult to find online, we provide links here:

Google Drive: https://drive.google.com/drive/folders/1Qmtp-Fo1iiQiyf-9uRUpO-YAAM0mcIey?usp=sharing

Baidu: https://pan.baidu.com/s/1TTyb0FJzUdP6muLSbVN3hQ  code: ptg0

# Training/Testing Data
In addition to the source MATLAB code to generate training/testing data, you may also find the generated files here (If you do not want to generate them yourselves or if you do not have MATLAB):

Google Drive: https://drive.google.com/file/d/1u-6xmedUB0PNA5xM787OY-YfiJg195xA/view

Baidu: https://pan.baidu.com/s/12nb6OTUxnz_rxssg2rthIQ code: 82k3

# Citation
```BibTeX
@article{zhang2021uncertainty,  
  title   = {Uncertainty-aware blind image quality assessment in the laboratory and wild},  
  author  = {Zhang, Weixia and Ma, Kede and Zhai, Guangtao and Yang, Xiaokang},  
  journal = {IEEE Transactions on Image Processing},    
  volume  = {30},  
  pages   = {3474--3486},  
  month   = {Mar.},  
  year    = {2021}
}
```
```BibTeX
@inproceedings{zhang2020learning,  
  title     = {Learning to blindly assess image quality in the laboratory and wild},  
  author    = {Zhang, Weixia and Ma, Kede and Zhai, Guangtao and Yang, Xiaokang},  
  booktitle = {IEEE International Conference on Image Processing},  
  pages     = {111--115},  
  year      = {2020}
}
```
