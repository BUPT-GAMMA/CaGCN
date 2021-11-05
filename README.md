# CaGCN

This repo is for source code of NeurIPS 2021 paper "Be Confident! Towards Trustworthy Graph Neural
Networks via Confidence Calibration".

Paper Link: https://arxiv.org/abs/2109.14285

## Environment

- python == 3.8.8
- pytorch == 1.8.1
- dgl -cuda11.1 == 0.6.1
- networkx == 2.5
- numpy == 1.20.2

GPU: GeForce RTX 2080 Ti

CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz



## Confidence Calibration

### CaGCN

```python
python CaGCN.py --model GCN --hidden 64 --dataset dataset --labelrate labelrate --stage 1 --lr_for_cal 0.01 --l2_for_cal 5e-3
python CaGCN.py --model GAT --hidden 8 --dataset dataset --labelrate labelrate --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --l2_for_cal 5e-3
```

- **dataset:**  including [Cora, Citeseer, Pubmed], required.
- **labelrate:** including [20, 40, 60], required.

e.g.,

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 20 --stage 1 --lr_for_cal 0.01 --l2_for_cal 5e-3
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --l2_for_cal 5e-3
```

For CoraFull,

```python
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate labelrate --stage 1 --lr_for_cal 0.01 --l2_for_cal 0.03
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate labelrate --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --l2_for_cal 0.03
```

- **labelrate:** including [20, 40, 60], required.

### Uncalibrated model

```python
python train_others.py --model GCN --hidden 64 --dataset dataset --labelrate labelrate --stage 1 
python train_others.py --model GAT --hidden 8 --dataset dataset --labelrate labelrate --stage 1 --dropout 0.6 --lr 0.005
```

- **dataset:**  including [Cora, Citeseer, Pubmed, CoraFull], required.
- **labelrate:** including [20, 40, 60], required.

e.g.,

```
python train_others.py --model GCN --hidden 64 --dataset Cora --labelrate 20 --stage 1
python train_others.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --stage 1 --dropout 0.6 --lr 0.005
```



### Temperature scaling & Matring Scaling

```python
python train_others.py --model GCN --scaling_method method --hidden 64 --dataset dataset --labelrate labelrate --stage 1 --lr_for_cal 0.01 --max_iter 50
python train_others.py --model GAT --scaling_method method --hidden 8 --dataset dataset --labelrate labelrate --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --max_iter 50
```

- **method:** including [TS, MS], required.
- **dataset:**  including [Cora, Citeseer, Pubmed, CoraFull], required.
- **labelrate:** including [20, 40, 60], required.

e.g.,

```
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Cora --labelrate 20 --stage 1 --lr_for_cal 0.01 --max_iter 50
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --stage 1 --lr_for_cal 0.01 --max_iter 50
```

## Self-Training

### GCN L/C=20

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9
python CaGCN.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.85
```

### GCN L/C=40

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Citeseer --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.85
python CaGCN.py --model GCN --hidden 64 --dataset Pubmed --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.99
```

### GCN L/C=60

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Citeseer --labelrate 60 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Pubmed --labelrate 60 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.6
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate 60 --stage 5 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.9
```

### GAT L/C=20

```python
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Citeseer --labelrate 20 --dropout 0.6 --lr 0.005 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.7
python CaGCN.py --model GAT --hidden 8 --dataset Pubmed --labelrate 20 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate 20 --dropout 0.6 --lr 0.005 --stage 5 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.95
```

### GAT L/C=40

```python
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 40 --dropout 0.6 --lr 0.005 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.9
python CaGCN.py --model GAT --hidden 8 --dataset Citeseer --labelrate 40 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Pubmed --labelrate 40 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate 40 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.95
```

### GAT L/C=60

```PYTHON
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 60 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Citeseer --labelrate 60 --dropout 0.6 --lr 0.005 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Pubmed --labelrate 60 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.85 
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate 60 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.95
```

## More Parameters

For more parameters of baselines, please refer to the Parameter.md

## Contact

If you have any questions, please feel free to contact me with liuhongrui@bupt.edu.cn
