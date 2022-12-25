# Riemannian Representation Learning (RRL)

This repository contains a paper with supplementary material for the deep domain adaptation approach DNA, and a pytorch implementation of the DNA approach.

In a nutshell, DNA solves the joint distribution mismatch problem in deep domain adaptation for large scale image recognition. To this end, it exploits a Convolutional Neural Network (CNN) to match the source and target joint distributions in the network representation space under the Relative Chi-Square (RCS) divergence. The following figure illustrates this deep joint distribution matching idea.   


This repository provides the Pytorch code for the work "Domain Generalization by Joint-Product Distribution Alignment" published in Pattern Recognition, 2022. In this work, we study the non-identically distributed supervised learning problem, where the training data are sampled from multiple different (probability）distributions, while the test data are governed by another different yet related distribution. We design a Joint-Product Distribution Alignment (JPDA) approach that aligns a joint distribution and a product distribution to tackle the distribution difference (see the illustration below), with (1) the loss function being the Relative Chi-Square (RCS) divergence, (2) the hypothesis space being the neural network transformation, and (3) the learning algorithm being the minibatch Stochastic Gradient Descent (minibatch SGD).


<img src="Problem.jpg" width="50%">

<img src="Manifold.jpg" width="50%">

##### Tested on
* Python 3.8
* PyTorch 1.11.0
* CUDA 11.4

#### Dataset folder
The folder structure required (e.g OfficeHome)
- data
  - OfficeHome
    - list
      - Art.txt
      - Clipart.txt
      - Prduct.txt
      - Real.txt
    - Art
    - Clipart
    - Product
    - Real


##### How to run

```bash
python  demo.py --dataset officehome --source Product   --target Clipart   --phase pretrain --gpu 0 --start_update_step 2000 --update_interval 1000 --steps 70000 --message "DNA" --alpha_div 0.5 --beta_div 0 --lambda_div 0.1 --patience 10
python demo.py --dataset officehome --source Product   --target Clipart   --phase train --gpu 0 --start_update_step 2000 --update_interval 1000 --steps 70000 --message "DNA" --alpha_div 0.5 --beta_div 0 --lambda_div 0.1 
```


For more details of this multi-source domain adaptation approach,  please refer to the following PR work: 

@article{Chen2023Riemannian,
  author = {Sentao Chen and Lin Zheng and Hanrui Wu},
  journal = {Pattern Recognition},
  title = {Riemannian Representation Learning for Multi-Source Domain Adaptation},
  year = {2023},
  pages = {109271},
  doi = {https://doi.org/10.1016/j.patcog.2022.109271}
}
  
The Pytorch code is currently maintained by Lisheng Wen. If you have any questions regarding the codes, please contact Lisheng Wen via the email wenlisheng992021@163.com.
