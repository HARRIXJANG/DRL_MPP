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

## Data
Since we do not have permission to provide the original dataset, only the modified parts AAGs are provided in .gml format. The nodes within these graphs store feasible solution matrices comprising normalized feature machining time information. Furthermore, we supply part AAG files (designated with the "all_time" suffix) containing the original feature machining time information.

## Train
We provide two training methods: (1) Directly run Train_PPO_mask_net.py. (2) Considering that roughing typically precedes other machining operations, in order to further improve performance, we first run Train_PPO_only_rough.py to obtain an agent capable of planning roughing operations, followed by running Train_PPO_without_rough.py to obtain an agent capable of planning non-roughing machining operations (e.g. finishing bottom and semi-finishing bottom).

## Machining process planning
We offer two approaches for machining process planning: (1) Executing Predict.py directly to utilize the trained policy network. (2) Employing MonteCarloTreeSearch.py to leverage the trained policy network, value network, and Monte Carlo Tree Search (MCTS) for machining process planning.

## Citation
If you use this code please cite:  
```
@inproceedings{  
      title={Employing Deep Reinforcement Learning for Machining Process Planning: An Improved Framework},  
      author=Hang Zhang, Wenhu Wang, Yue Wang, Yajun Zhang, Jingtao Zhou, Bo Huang, and Shusheng Zhang},  
      booktitle={Journal of Manufacturing Systems},  
      year={2025}  
    }
``` 
If you have any questions about the code, please feel free to contact me (zhnwpu714@163.com).
