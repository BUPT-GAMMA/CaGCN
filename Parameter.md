# Environment

- python == 3.8.8
- pytorch == 1.8.1
- dgl -cuda11.1 == 0.6.1
- networkx == 2.5
- numpy == 1.20.2

# Usage

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

### CaGCN-st

#### GCN L/C=20

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.9
python CaGCN.py --model GCN --hidden 64 --dataset Pubmed --labelrate 20 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate 20 --stage 4 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.85
```

#### GCN L/C=40

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Citeseer --labelrate 40 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.85
python CaGCN.py --model GCN --hidden 64 --dataset Pubmed --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate 40 --stage 4 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.99
```

#### GCN L/C=60

```python
python CaGCN.py --model GCN --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Citeseer --labelrate 60 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
python CaGCN.py --model GCN --hidden 64 --dataset Pubmed --labelrate 60 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.6
python CaGCN.py --model GCN --hidden 64 --dataset CoraFull --labelrate 60 --stage 5 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.9
```

#### GAT L/C=20

```python
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Citeseer --labelrate 20 --dropout 0.6 --lr 0.005 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.7
python CaGCN.py --model GAT --hidden 8 --dataset Pubmed --labelrate 20 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate 20 --dropout 0.6 --lr 0.005 --stage 5 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.95
```

#### GAT L/C=40

```python
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 40 --dropout 0.6 --lr 0.005 --stage 4 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.9
python CaGCN.py --model GAT --hidden 8 --dataset Citeseer --labelrate 40 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Pubmed --labelrate 40 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.8 
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate 40 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.95
```

#### GAT L/C=60

```PYTHON
python CaGCN.py --model GAT --hidden 8 --dataset Cora --labelrate 60 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 200 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Citeseer --labelrate 60 --dropout 0.6 --lr 0.005 --stage 6 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 150 --threshold 0.8
python CaGCN.py --model GAT --hidden 8 --dataset Pubmed --labelrate 60 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 3 --lr_for_cal 0.001 --l2_for_cal 5e-3 --epoch_for_st 100 --threshold 0.85 
python CaGCN.py --model GAT --hidden 8 --dataset CoraFull --labelrate 60 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --l2_for_cal 0.03 --epoch_for_st 500 --threshold 0.95
```

### TS-st

#### GCN

```python
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Cora --labelrate 20 --stage 3 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Cora --labelrate 40 --stage 6 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Cora --labelrate 60 --stage 4 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Citeseer --labelrate 20 --stage 5 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Citeseer --labelrate 40 --stage 3 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Citeseer --labelrate 60 --stage 2 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Pubmed --labelrate 20 --stage 2 --lr_for_cal 0.01 --max_iter 50 --threshold 0.85
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Pubmed --labelrate 40 --stage 2 --lr_for_cal 0.01 --max_iter 50 --threshold 0.85
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset Pubmed --labelrate 60 --stage 3 --lr_for_cal 0.01 --max_iter 50 --threshold 0.85
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset CoraFull --labelrate 20 --stage 3 --lr_for_cal 0.001 --max_iter 50 --threshold 0.95
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset CoraFull --labelrate 40 --stage 4 --lr_for_cal 0.001 --max_iter 25 --threshold 0.99
python train_others.py --model GCN --scaling_method TS --hidden 64 --dataset CoraFull --labelrate 60 --stage 2 --lr_for_cal 0.001 --max_iter 25 --threshold 0.95
```

#### GAT

```python
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --stage 3 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Cora --labelrate 40 --dropout 0.6 --lr 0.005 --stage 6 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Cora --labelrate 60 --dropout 0.6 --lr 0.005 --stage 4 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Citeseer --labelrate 20 --dropout 0.6 --lr 0.005 --stage 5 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Citeseer --labelrate 40 --dropout 0.6 --lr 0.005 --stage 3 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Citeseer --labelrate 60 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.01 --max_iter 50 --threshold 0.8
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Pubmed --labelrate 20 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 2 --lr_for_cal 0.01 --max_iter 50 --threshold 0.85
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Pubmed --labelrate 40 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 2 --lr_for_cal 0.01 --max_iter 50 --threshold 0.85
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset Pubmed --labelrate 60 --dropout 0.6 --lr 0.005 --weight_decay 1e-3 --stage 3 --lr_for_cal 0.01 --max_iter 50 --threshold 0.85
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset CoraFull --labelrate 20 --dropout 0.6 --lr 0.005 --stage 3 --lr_for_cal 0.001 --max_iter 50 --threshold 0.95
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset CoraFull --labelrate 40 --dropout 0.6 --lr 0.005 --stage 4 --lr_for_cal 0.001 --max_iter 25 --threshold 0.99
python train_others.py --model GAT --scaling_method TS --hidden 8 --dataset CoraFull --labelrate 60 --dropout 0.6 --lr 0.005 --stage 2 --lr_for_cal 0.001 --max_iter 25 --threshold 0.95
```

### self-training

All the labelrate (20, 40, 60) shares the same parameters.

```python
python self_training.py --model GCN --dataset Cora --k 500 --labelrate 20
python self_training.py --model GCN --dataset Citeseer --k 500 --labelrate 20
python self_training.py --model GCN --dataset Pubmed --k 500 --labelrate 20
python self_training.py --model GCN --dataset CoraFull --k 3 --labelrate 20
```

### co-training

#### GCN

```python
python co_training.py --model GCN --dataset Cora --labelrate 20 --k 3
python co_training.py --model GCN --dataset Cora --labelrate 40 --k 3
python co_training.py --model GCN --dataset Cora --labelrate 60 --k 3
python co_training.py --model GCN --dataset Citeseer --labelrate 20 --k 10 --weight_decay 1e-3
python co_training.py --model GCN --dataset Citeseer --labelrate 40 --k 0.1
python co_training.py --model GCN --dataset Citeseer --labelrate 60 --k 0.1
python co_training.py --model GCN --dataset Pubmed --labelrate 20 --k 0.1
python co_training.py --model GCN --dataset CoraFull --labelrate 20 --k 0.1
```

#### GAT

```python
python co_training.py --model GAT --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005  --hidden 8 --k 3
python co_training.py --model GAT --dataset Cora --labelrate 40 --dropout 0.6 --lr 0.005  --hidden 8 --k 1 --weight_decay 1e-3
python co_training.py --model GAT --dataset Cora --labelrate 60 --dropout 0.6 --lr 0.005  --hidden 8 --k 1
python co_training.py --model GAT --dataset Citeseer --labelrate 20 --dropout 0.6 --lr 0.005  --hidden 8 --k 3
python co_training.py --model GAT --dataset Citeseer --labelrate 40 --dropout 0.6 --lr 0.005  --hidden 8 --k 1 --weight_decay 1e-3
python co_training.py --model GAT --dataset Citeseer --labelrate 60 --dropout 0.6 --lr 0.005  --hidden 8 --k 1 --weight_decay 1e-3
python co_training.py --model GAT --dataset Pubmed --labelrate 20 --dropout 0.6 --lr 0.005  --hidden 8 --k 0.1
python co_training.py --model GAT --dataset Pubmed --labelrate 40 --dropout 0.6 --lr 0.005  --hidden 8 --k 0.1
python co_training.py --model GAT --dataset Pubmed --labelrate 60 --dropout 0.6 --lr 0.005  --hidden 8 --k 0.1
python co_training.py --model GAT --dataset CoraFull --labelrate 20 --dropout 0.6 --lr 0.005  --hidden 8 --k 1
python co_training.py --model GAT --dataset CoraFull --labelrate 40 --dropout 0.6 --lr 0.005  --hidden 8 --k 1
python co_training.py --model GAT --dataset CoraFull --labelrate 60 --dropout 0.6 --lr 0.005  --hidden 8 --k 1
```

### Union

#### GCN

```python
python union.py --model GCN --dataset Cora --labelrate 20 --k 3
python union.py --model GCN --dataset Cora --labelrate 40 --k 1
python union.py --model GCN --dataset Cora --labelrate 60 --k 1 --weight_decay 1e-3
python union.py --model GCN --dataset Citeseer --labelrate 20 --k 10 --weight_decay 1e-3
python union.py --model GCN --dataset Citeseer --labelrate 40 --k 15 --weight_decay 1e-3
python union.py --model GCN --dataset Citeseer --labelrate 60 --k 25 --weight_decay 1e-3
python union.py --model GCN --dataset Pubmed --labelrate 20 --k 10 --weight_decay 1e-3
python union.py --model GCN --dataset Pubmed --labelrate 40 --k 25 --weight_decay 1e-3
python union.py --model GCN --dataset Pubmed --labelrate 60 --k 45 --weight_decay 1e-3
python union.py --model GCN --dataset CoraFull --labelrate 20 --k 0.1
python union.py --model GCN --dataset CoraFull --labelrate 40 --k 0.1
python union.py --model GCN --dataset CoraFull --labelrate 60 --k 0.1
```

#### GAT

```python
python union.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --k 45 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Cora --labelrate 40 --dropout 0.6 --lr 0.005 --k 45 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Cora --labelrate 60 --dropout 0.6 --lr 0.005 --k 70 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Citeseer --labelrate 20 --dropout 0.6 --lr 0.005 --k 30 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Citeseer --labelrate 40 --dropout 0.6 --lr 0.005 --k 30 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Citeseer --labelrate 60 --dropout 0.6 --lr 0.005 --k 30 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Pubmed --labelrate 20 --dropout 0.6 --lr 0.005 --k 10 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Pubmed --labelrate 40 --dropout 0.6 --lr 0.005 --k 25 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset Pubmed --labelrate 60 --dropout 0.6 --lr 0.005 --k 45 --weight_decay 1e-3
python union.py --model GAT --hidden 8 --dataset CoraFull --labelrate 20 --dropout 0.6 --lr 0.005 --k 5
python union.py --model GAT --hidden 8 --dataset CoraFull --labelrate 40 --dropout 0.6 --lr 0.005 --k 1
python union.py --model GAT --hidden 8 --dataset CoraFull --labelrate 60 --dropout 0.6 --lr 0.005 --k 5
```

### Intersection

#### GCN

```python
python intersection.py --model GCN --dataset Cora --labelrate 20 --k 15
python intersection.py --model GCN --dataset Cora --labelrate 40 --k 15
python intersection.py --model GCN --dataset Cora --labelrate 60 --k 15
python intersection.py --model GCN --dataset Citeseer --labelrate 20 --k 25 --weight_decay 1e-3
python intersection.py --model GCN --dataset Citeseer --labelrate 40 --k 25 --weight_decay 1e-3
python intersection.py --model GCN --dataset Citeseer --labelrate 60 --k 25 --weight_decay 1e-3
python intersection.py --model GCN --dataset Pubmed --labelrate 20 --k 10 --weight_decay 1e-3
python intersection.py --model GCN --dataset Pubmed --labelrate 40 --k 40 --weight_decay 1e-3
python intersection.py --model GCN --dataset Pubmed --labelrate 60 --k 35 --weight_decay 1e-3
python intersection.py --model GCN --dataset CoraFull --labelrate 20 --k 1
python intersection.py --model GCN --dataset CoraFull --labelrate 40 --k 1
python intersection.py --model GCN --dataset CoraFull --labelrate 60 --k 1
```

#### GAT

```python
python intersection.py --model GAT --hidden 8 --dataset Cora --labelrate 20 --dropout 0.6 --lr 0.005 --k 15
python intersection.py --model GAT --hidden 8 --dataset Cora --labelrate 40 --dropout 0.6 --lr 0.005 --k 15
python intersection.py --model GAT --hidden 8 --dataset Cora --labelrate 60 --dropout 0.6 --lr 0.005 --k 15
python intersection.py --model GAT --hidden 8 --dataset Citeseer --labelrate 20 --dropout 0.6 --lr 0.005 --k 25 --weight_decay 1e-3
python intersection.py --model GAT --hidden 8 --dataset Citeseer --labelrate 40 --dropout 0.6 --lr 0.005 --k 25 --weight_decay 1e-3
python intersection.py --model GAT --hidden 8 --dataset Citeseer --labelrate 60 --dropout 0.6 --lr 0.005 --k 25 --weight_decay 1e-3
python intersection.py --model GAT --hidden 8 --dataset Pubmed --labelrate 20 --dropout 0.6 --lr 0.005 --k 10 --weight_decay 1e-3
python intersection.py --model GAT --hidden 8 --dataset Pubmed --labelrate 40 --dropout 0.6 --lr 0.005 --k 25 --weight_decay 1e-3
python intersection.py --model GAT --hidden 8 --dataset Pubmed --labelrate 60 --dropout 0.6 --lr 0.005 --k 35 --weight_decay 1e-3
python intersection.py --model GAT --hidden 8 --dataset CoraFull --labelrate 20 --dropout 0.6 --lr 0.005 --k 10
python intersection.py --model GAT --hidden 8 --dataset CoraFull --labelrate 40 --dropout 0.6 --lr 0.005 --k 10
python intersection.py --model GAT --hidden 8 --dataset CoraFull --labelrate 60 --dropout 0.6 --lr 0.005 --k 10
```

