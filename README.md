# WaferMap Dataset: MixedWM38

[MixedWM38](https://ieeexplore.ieee.org/document/9184890/) Dataset(WaferMap) has more than 38000 wafer maps, including 1 normal pattern, 8 single defect patterns, and 29 mixed defect patterns, a total of 38 defect patterns.

## Requirements:

* Python 3+

* numpy scipy matplotlib tensorflow keras scikit-learn

## Motivation：
Defect pattern recognition (DPR) of wafer maps, especially the mixed-type defect, is critical for determining the root cause of production defects.
We collected a large amount of wafer map data in a wafer manufacturing plant. These wafer maps are obtained by testing the electrical performance of each die on the wafer through test probes. However, there are big differences between the quantity distribution of various types of wafer maps actually collected. In order to maintain the balance between the various types of data, we used the confrontation generation network to generate some wafer maps to maintain the balance of the number of samples among the types. Finally, about 38,000 mixed-mode wafer map defect data sets are formed, which are used to identify mixed-mode wafer map defects and assist the research on the causes of defects in the wafer manufacturing process. In order to facilitate researchers, students, and enthusiasts in related fields to better understand the causes of defects in the wafer manufacturing process, we have published a data set of mixed-mode wafer map defects for your reference and research.

## Description:

* [Dataset Overview](Dataset%20Figure/Wafer%20Map.png)
* [‘arr_0’]: Defect data of mixed-type wafer map, 0 means blank spot, 1 represents normal die that passed the electrical test, and 2 represents broken die that failed the electrical test. The data shape is 52×52.
* [‘arr_1’]: Mixed-type wafer map defect label, using one-hot encoding, a total of 8 dimensions, corresponding to the 8 basic types of wafer map defects (C2-C9).

* Sources:\
[Kaggle](https://www.kaggle.com/co1d7era/mixedtype-wafer-defect-datasets)\
[Google](https://drive.google.com/file/d/1bUow-p9LwkRI4yP74j7Um5s30-WHcxbA/view?usp=sharing)
[Baidu](https://pan.baidu.com/s/19aEazgpLMBPBzjePSCTgHw) Extraction code: mrpm

* [Model Training](train_mutil_label.py)

* More:[MixedWM38](https://ieeexplore.ieee.org/document/9184890/)

## Introduction in other languages
* Chinese：[混合模式晶圆图缺陷数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=77328)

## Citation:
[J. Wang, C. Xu, Z. Yang, J. Zhang and X. Li, "Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition," in IEEE Transactions on Semiconductor Manufacturing, doi: 10.1109/TSM.2020.3020985.](https://ieeexplore.ieee.org/document/9184890/)

## Acknowledgement

Thanks to Dr. Uzma Batool from the University of Technology Malaysia for correcting the label errors in the original dataset! The C7 and C9 labels in the dataset have been corrected!

