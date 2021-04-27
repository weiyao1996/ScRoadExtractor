# ScRoadExtractor
This repository is the official implementation of Scribble-based Weakly Supervised Deep Learning for Road Surface Extraction from Remote Sensing Images (TGRS 2021)
[arxiv](https://arxiv.org/abs/2010.13106);[paper](https://ieeexplore.ieee.org/document/9372390)
![image](https://github.com/weiyao1996/weiyao1996.github.io/blob/master/img/xxxx.png)  

## Dependencies  
· Python == 3.6.7
· Numpy == 1.16.3
· opencv == 4.1.0
· PyMaxflow == 1.2.12
· scipy  == 1.2.1
· Pytorch == 1.1.0  
## Structure  
  
## Usage  
1) Download dataset and prepare for the code
The scribbles can be obtained from OpenStreetMap centerlines, GPS traces, or manually annotation through ArcGIS or other software. Also, you can generate skeletonized road lines by thinning road segmentation maps (skimage.morphology.thin).
With respect to the implementation of HED Boundary detector, you can refer to the folder `boundary_detect`. To generate HED masks, download the pre-trained model [network-bsds500.pytorch] by `download.bash` and run `run.py`. We also provide the pre-trained model [network-bsds500.pytorch] using the link below.
https://pan.baidu.com/s/1AMNnmo7YAk1X3_m8Ky1arw (pwd:0HED)
2) Road label propagation
Run `road_label_propagation.py` to derive proposal masks.
3) DBNet
Run `train.py` for training and run `test.py` for testing. 

## Feedback  
For questions and comments, feel free to contact Yao WEI（email: weiyao@whu.edu.cn)
## Citation  
If you find our work useful in your research, please cite:  
`Yao Wei, and Shunping Ji. Scribble-based Weakly Supervised Deep Learning for Road Surface Extraction from Remote Sensing Images. IEEE Transactions on Geoscience and Remote Sensing 2021.`  
