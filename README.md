# UNIQUE
The codebase for  
[Uncertainty-aware blind image quality assessment in the laboratory and wild](https://arxiv.org/pdf/2005.13983.pdf)  
and  
[Learning to blindly assess image quality in the laboratory and wild](https://arxiv.org/pdf/1907.00516.pdf) (To appear in ICIP2020)  

# Requirement:
Python 3+  
PyTorch 1.4+  
Successfully tested on Ubuntu18.04, other OS (i.e., other Linux version, Windows)should also be ok.

# Usage
## Training on multiple databases for 10 sessions
python Main.py --train True --network basecnn --representation BCNN --ranking True --fidelity True --std_modeling True --std_loss True --margin 0.025 --batch_size 128 --batch_size2 32 --image_size 384 --max_epochs 3 --lr 1e-4 --decay_interval 3 --decay_ratio 0.1 --max_epochs2 12

## Find the top-performing checkpoints according to weighted SRCC
python find_best.py
