## Anomaly Detection via Reverse Distillation

This repo is developed based on the official code [RD4AD](https://github.com/hq-deng/RD4AD) by [hq-deng](https://github.com/hq-deng), of the paper:  

**Anomaly Detection via Reverse Distillation from One-Class Embedding**  
Hanqiu Deng and Xingyu Li  
CVPR 2022  
[https://arxiv.org/abs/2201.10703](https://arxiv.org/abs/2201.10703)  

## What's Changed?
- Apply for custom dataset **GFC** from [Attention-based deep learning for chip-surface-defect detection](https://doi.org/10.1007/s00170-022-09425-4)
- Explore different backbones, including **ConvNeXt** and **MambaVision**
- Propose Adaptive Stages Fusion, introduce trainable weights for different stages of the backbone
- Expand Metrics, including **AP**, **Overkill** and **Underkill**

 ## Dataset
- **MVTec:** [Download](https://www.mvtec.com/company/research/datasets/mvtec-ad/) and unpack to folder ./dataset/mvtec/
- **GFC:** [Download](https://pan.baidu.com/s/1DsZyyO4ITtsLWqFyGS2KEA) and unpack to folder ./dataset/gfc/, then run:
```commandline
python gfc.py
```

## Library Install
```commandline
pip install -r requirements.txt
pip install -r mamba_requirements.txt --no-build-isolation
``` 

## Usage
### Train the model:
```commandline
python train.py 
    -d <dataset>
    -c <dclass>  
    -be <backbone>
    -is <image_size>
    -w <use_stage_weights>
    -wa <layer_weight_alpha>
    -ig <inverse_gap_feature>
    -bs <batch_size> 
    -s <seed>
    -e <num_epochs>
    -fe <num_fusion_epochs>
    -lr <learning_rate>
    -er <weight_entropy>
    -pa <patience>
```
### Run the test: 
```commandline
python test.py 
    -f test
    -d <dataset>
    -c <dclass>  
    -be <backbone>
    -is <image_size>
    -w <use_stage_weights>
    -wa <layer_weight_alpha>
    -ig <inverse_gap_feature>
```
### Visualize results: 
```commandline
    -f visualize
    -d <dataset>
    -c <dclass>  
    -be <backbone>
    -is <image_size>
    -w <use_stage_weights>
    -wa <layer_weight_alpha>
    -ig <inverse_gap_feature>
```
Visit the folder for saved image: /result/&lt;dataset&gt;/images/

### Example
```commandline
python train.py -d mvtec -be mambavision-b -w 1 -wa 2.0 -er 0.1 -bs 16 -fe 200
python test.py -f test -d -d mvtec -be mambavision-b -w 1 -wa 2.0
python test.py -f visualize -d mvtec -be mambavision-b -w 1 -wa 2.0

```