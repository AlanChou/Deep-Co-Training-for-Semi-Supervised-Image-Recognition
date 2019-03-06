# Deep Co-Training for Semi-Supervised Image Recognition

This is the unofficial PyTorch implementation of the paper "Deep Co-Training for Semi-Supervised Image Recognition" in ECCV 2018. 
http://openaccess.thecvf.com/content_ECCV_2018/papers/Siyuan_Qiao_Deep_Co-Training_for_ECCV_2018_paper.pdf

It only contains the code for 2-view co-training on CIFAR-10. 
2-view co-training on SVHN is not completed yet.
The repository is maintained by Hsin-Ping Chou and Jie-Young Chen.


## Overview

All hyperparameters are set the same as they are mentioned in the paper. There are two things that are not mentioned in the paper. One is the epsilon of the FGSM attack. The other one is how they attack those unlabelled data. We've asked the authors of the paper and got their response. The epsilon was set to 0.02 and they used the pseudo labels of those unlabelled data to attack. Note that for FGSM attack. The authors of the paper also mentioned that they didn't use torch.clamp which is usually used for FGSM while in our implementation we did clamp the value to (0,1). We've tried with/without clamp and it didn't effect the results.

It can achieve 89.1% accuracy (net1 test acc: 89.14% | net2 test acc: 89.06%) with a single 2080ti GPU in 600 epochs (each epoch takes around 2 minutes), which is still far from the reported baseline in the paper (90.93%). Please feel free to correct our implementation if there's anything wrong.

![image](https://github.com/AlanChou/Deep-Co-Training-for-Semi-Supervised-Image-Recognition/blob/master/Test.PNG) 
## Dependencies
This code is based on Python 3.5, with the main dependencies being PyTorch==0.4.1. Additional dependencies for running experiments are: numpy, Pickle, tqdm, math, argparse, os, random, tensorboardX, advertorch.

Advertorch can be installed from https://github.com/BorealisAI/advertorch  
Run the code with command $ CUDA_VISIBLE_DEVICES=0 python3 main.py 
Multi-GPU is not supported with this code.

