# WaferMap

Wafer Bin Map DataSet(WaferMap) has more than 36000 wafer maps, including 1 normal pattern, 8 single defect pattern, and 29 mixed defect pattern, a total of 38 defect pattern.

## Requirements:

* Python 3+

* numpy scipy matplotlib tensorflow keras scikit-learn

## Instructions：
* Download and process the data in [data_mutil_label.npz](https://pan.baidu.com/s/19aEazgpLMBPBzjePSCTgHw)\
Extraction code: mrpm

* In the data set, "data['arr 0']" is the wafer image data, and "data['arr1']" is the label of the wafer image. The wafer map label has eight dimensions, corresponding to the defect mode of C2-C9 [in Wafer Map.png](Wafer%20Map.png)

* To train a model, run [train_mutil_label.py](train_mutil_label.py)

## WaferMap pattern Show:
* The picture introduces 38 wafer pattern and the corresponding feature map after using Deformable Convolution

![](https://github.com/Junliangwangdhu/WaferMap/blob/master/Wafer%20Map.png)


## Introduction in other languages
* Chinese：[混合模式晶圆图缺陷数据集](https://tianchi.aliyun.com/dataset/dataDetail?dataId=77328)

## Cite This:
J. Wang, C. Xu, Z. Yang, J. Zhang and X. Li, "Deformable Convolutional Networks for Efficient Mixed-type Wafer Defect Pattern Recognition," in IEEE Transactions on Semiconductor Manufacturing, doi: 10.1109/TSM.2020.3020985.

