# DRL_MPP
Created by Hang Zhang from Northwestern Polytechnical University. 

## Introduction
DRL_MPP is a novel deep reinforcement learning (DRL) framework for machining process planning. The framework takes a part represented by attributed adjacency graphs (AAGs) as input and its machining process scheme as output. 

In light of confidentiality constraints, the original source code and dataset cannot be disclosed at present. However, the modified code and several data have been made available for public access. While these materials may not fully replicate the experimental procedures outlined in the paper, they suffice to realize the fundamental processes delineated therein.

## Setup
(1)	cuda 11.6.112     
(2)	python 3.8.13  
(3)	pytorch 1.12.0   
(4)	tianshou 0.4.11  
(5)   dgl 0.9.1  
(6)   gym 0.25.2  
(7)   tensorboard 2.10.0   

The code is tested on Intel Core i9-10980XE CPU, 128GB memory, and NVIDIA GeForce RTX 3090 GPU. 

## Train
我们提供了两种训练方法：（1）直接运行Train_PPO_mask_net.py.（2）考虑到粗加工通常先于其他加工操作，因此为了进一步提高性能，我们首先运行Train_PPO_only_rough.py以获得能够对粗加工操作进行规划的智能体，而后我们运行Train_PPO_without_rough.py以获得能够对其他加工操作进行工艺规划的智能体。

## Evaluation
The folder "all_eval_data" contains all public evaluation part graphs.  
(1)	Get the source code by cloning the repository: https://github.com/HARRIXJANG/DRLFS_master.git.   
(2)   Copy a txt file that you want to predict from the "all_eval_data" folder to the "eval" folder (only one file can be placed in the "eval" folder at a time).   
(3)	Run `Evaluation.py` to predict. The result is a txt file, in which each row is the handle numbers of all faces that can contruct an isotlated machining feature (the handle number of a face is unique and constant in NX 12.0). According to the result, readers can manually view the final display result in the corresponding CAD model from NX 12.0.    

## Citation
If you use this code please cite:  
```
@inproceedings{  
      title={A novel method for intersecting machining feature segmentation via deep reinforcement learning},  
      author={Hang Zhang, Wenhu Wang, Shusheng Zhang, Yajun Zhang, Jingtao Zhou, Zhen Wang, Bo Huang, and Rui Huang},  
      booktitle={Advanced Engineering Informatics},  
      year={2023}  
    }
``` 
If you have any questions about the code, please feel free to contact me (zhnwpu714@163.com).
