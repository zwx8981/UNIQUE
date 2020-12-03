# UNIQUE
The codebase for  
[Uncertainty-aware blind image quality assessment in the laboratory and wild](https://arxiv.org/pdf/2005.13983.pdf)  
and  
[Learning to blindly assess image quality in the laboratory and wild](https://arxiv.org/pdf/1907.00516.pdf) (ICIP2020)  

![image](https://github.com/zwx8981/UNIQUE/blob/master/UNIQUE_framework.png)

# Prequisite:
Python 3+  
PyTorch 1.4+  
Matlab  
Successfully tested on Ubuntu18.04, other OS (i.e., other Linux distributions, Windows)should also be ok.

# Usage
## Sampling image pairs from multiple databases
data_all.m  
## Combining the sampled pairs to form the training set
combine_train.m  
## Training on multiple databases for 10 sessions
python Main.py --train True --network basecnn --representation BCNN --ranking True --fidelity True --std_modeling True --std_loss True --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --max_epochs2 12 

(As for ICIP version, set std_loss to False and sample pairs from TID2013 instead of KADID-10K.)
(For training with binary labels, set fideliy and std_modeling to False.)
## Output predicted quality scores and stds
python Main.py --train False --get_scores True
## Result anlysis
Compute SRCC/PLCC after nonlinear mapping: result_analysis.m  
Compute fidelity loss: eval_fidelity.m

# Pre-trained weights
Google: https://drive.google.com/file/d/18oPH4lALm8mSdZh3fWK97MVq9w3BbEua/view?usp=sharing

Baidu: https://pan.baidu.com/s/1KKncQIoQcbxj7fQlSKUBIQ   code：yyev 

A basic demo: python demo.py

# Citation
@article{zhang2020uncertainty,  
  title={Uncertainty-aware blind image quality assessment in the laboratory and wild},  
  author={Zhang, Weixia and Ma, Kede and Zhai, Guangtao and Yang, Xiaokang},  
  journal={CoRR},  
  volume    = {abs/2005.13983},  
  year={2020}
}  
__________________________________________________________________________________________
@inproceedings{zhang2020learning,  
  title={Learning to blindly assess image quality in the laboratory and wild},  
  author={Zhang, Weixia and Ma, Kede and Zhai, Guangtao and Yang, Xiaokang},  
  booktitle={IEEE International Conference on Image Processing},  
  year={2020}
}
