# mulit-conv-re
code of cvpr HOI4D workshop action segmentation track (2nd prize)

modified from https://github.com/hoi4d/HOI4D_ActionSeg

pdf: assets/Multi_scale_convolution_for_spatio_temporal_modeling_4d_point_cloud.pdf

Pretrained model is provided in [link]()
## Train
python train_pptr.py --output-dir ./output
## Test
python test.py --output-dir ./output

## Install
These packages are needed:
```
torch
numpy
torchvision
```
This code is also based on the environment of pointnet++, so you should install it using following command:
```
cd ./modules
pip install .
```

## Citation
```
@InProceedings{Liu_2022_CVPR,
    author    = {Liu, Yunze and Liu, Yun and Jiang, Che and Lyu, Kangbo and Wan, Weikang and Shen, Hao and Liang, Boqiang and Fu, Zhoujie and Wang, He and Yi, Li},
    title     = {HOI4D: A 4D Egocentric Dataset for Category-Level Human-Object Interaction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21013-21022}
}
```
```
@inproceedings{wen2022point,
  title={Point Primitive Transformer for Long-Term 4D Point Cloud Video Understanding},
  author={Wen, Hao and Liu, Yunze and Huang, Jingwei and Duan, Bo and Yi, Li},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXIX},
  pages={19--35},
  year={2022},
  organization={Springer}
}
```
```
@inproceedings{fan2021point,
  title={Point 4d transformer networks for spatio-temporal modeling in point cloud videos},
  author={Fan, Hehe and Yang, Yi and Kankanhalli, Mohan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14204--14213},
  year={2021}
}
