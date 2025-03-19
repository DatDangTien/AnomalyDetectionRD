## Anomaly Detection via Reverse Distillation

This project is modified from the official code [RD4AD](https://github.com/hq-deng/RD4AD) by [hq-deng](https://github.com/hq-deng), based on the paper:  

**Anomaly Detection via Reverse Distillation from One-Class Embedding**  
Hanqiu Deng and Xingyu Li  
CVPR 2022  
[https://arxiv.org/abs/2201.10703](https://arxiv.org/abs/2201.10703)  

## What's Changed?
- Added new preprocessing method
- Added loss visualization
- Modified test, visualize methods
- Tested with custom dataset **GFC** from [Attention-based deep learning for chip-surface-defect detection](https://doi.org/10.1007/s00170-022-09425-4)

## Library Install
	> pytorch == 1.91
	
	> torchvision == 0.10.1
	
	> numpy == 1.20.3
	
	> scipy == 1.7.1
	
	> sklearn == 1.0
	
	> PIL == 8.3.2
 
 ## Dataset
- **MVTec:** [Download](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and unpack to folder *./dataset/mvtec/*
- **GFC:** [Download](https://pan.baidu.com/s/1DsZyyO4ITtsLWqFyGS2KEA) and unpack to folder *./dataset/gfc/*

## Train and Test the Model
To train the model with mvtec dataset
```
python main.py mvtec
python main.py gfc
```
To test -> *./result/{dataset}/benchmark.txt*
```
python test.py test mvtec
python test.py test gfc
```
To visualize image -> *./result/{dataset}/images/*
```
python test.py visualize mvtec
python test.py visualize gfc
```

