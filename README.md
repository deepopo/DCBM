# DCBM README


## Prerequisites
We use the same environment setting as the one used in "Concept Bottleneck Model" (CBM). Please run `pip install -r requirements.txt` to achieve the environment. Parts of codes are revised based on CBM.

## Datasets
We use CUB and Derm7pt in this repository. Please refer to:
#### CUB
[Original Dataset](http://www.vision.caltech.edu/datasets/cub_200_2011/), [Processed Dataset](https://worksheets.codalab.org/worksheets/0x362911581fcd4e048ddfd84f47203fd2).
#### Derm7pt
[Original Dataset](http://derm.cs.sfu.ca).

## Usage
We provide several tools in this repository, including `main.py`, `inference.py`, `mine.py`, and `rec.py`. Take `CUB` as example, we have:

### Concept and Label Prediction
#### Train
```
python main.py -dataset CUB -exp DCBM -log_dir ./logs/CUB/ -e 1000 -optimizer sgd -cuda_device 0 -seed 1 -ckpt 1 -use_attr -use_embs -attr_loss_weight 0.1 -embs_weight 1.0 -pretrained -use_aux -early_stop -weighted_loss multiple -data_dir CUB_processed/class_attr_data_10 -concept_percent 100 -n_attributes 112 -num_classes 200 -normalize_loss -b 64 -weight_decay 0.00004 -lr 0.01 -scheduler_step 20 -end2end
```

#### Inference
```
python utils/inference.py -dataset CUB -model_dirs ./logs/CUB/DCBM0.1_1.0_100Model_Seed1/best_model_1.pth -eval_data test -use_attr -concept_percent 100 -n_attributes 112 -data_dir CUB_processed/class_attr_data_10
```

### Forward Intervention and Backward Rectification

#### Train Decoupling Neural Network based on Mutual Information
```
python mine.py CUB -seed 1 -log_dir ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/ -e 10000 -cuda_device 0 -use_attr -data_dir CUB_processed/class_attr_data_10 -model_dir ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/best_model_1.pth -concept_percent 100 -n_attributes 112 -batch_size 64 -lr1 0.001 -lr2 0.0005
```

#### Forward Intervention
```
python mine.py CUB -log_dir ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/ -log_dir2 ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/ -e 1000 -inference -cuda_device 0 -use_attr -data_dir CUB_processed/class_attr_data_10 -model_dir2 ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/best_model_1.pth -concept_percent 100 -n_attributes 112 -batch_size 64 -lr1 0.001 -lr2 0.0005
```

#### Backward Rectification
```
python rec.py CUB -seed 1 -log_dir ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/ -e 2000 -cuda_device 0 -use_attr -data_dir CUB_processed/class_attr_data_10 -model_dir ./logs/CUB/DCBM0.1_0.01_100Model_Seed1/best_model_1.pth -concept_percent 100 -n_attributes 112 -batch_size 64
```