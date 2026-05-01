训练参数：LR=1e-4，batch size=32，150个epoch，optimizer是Adam加weight_decay=1e-5，scheduler是ReduceLROnPlateau，factor=0.5，patience=15，loss是MSELoss

用的是QM9数据集，PyG版本，Split用split_42.npz，自己生成的话参考train 104664，val 13083，test 13084，随机种子42

每个epoch记录train MSE、val MSE、val MAE，保存到metrics.csv，写报告需要用