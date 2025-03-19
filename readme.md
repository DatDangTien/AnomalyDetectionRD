## Anomaly Detection via Reverse Distillation

This project is modified from the official code of [RD4AD](https://github.com/hq-deng/RD4AD) by [hq-deng](https://github.com/hq-deng), based on the paper:  

**Anomaly Detection via Reverse Distillation from One-Class Embedding**  
Hanqiu Deng and Xingyu Li  
CVPR 2022  
[https://arxiv.org/abs/2201.10703](https://arxiv.org/abs/2201.10703)  

## What's Changed?
- Added new preprocessing method
- Added loss visualization
- Tested with custom dataset from [Attention-based deep learning for chip-surface-defect detection](https://doi.org/10.1007/s00170-022-09425-4)

## Library Install
	> pytorch == 1.91
	
	> torchvision == 0.10.1
	
	> numpy == 1.20.3
	
	> scipy == 1.7.1
	
	> sklearn == 1.0
	
	> PIL == 8.3.2
 
 ## Dataset
Downnload dataset [MVTec](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and unpack to folder *mvtec/*
## Train and Test the Model
To train the model with mvtec dataset
```
python main.py mvtec
```
To test 
```
python test.py test mvtec
```
To visualize image
```
python test.py visualize mvtec
```

