
# 项目使用方法

1. 把data数据集放在cls文件夹下
2. 运行_1_pre_data.py预处理数据  （之后就用不到data数据集了）
3. 运行_2_train_XXX.py开始训练

其中， 

\_2\_train_base.py是最优方案，欲运行_2_train_base.py的最优配置：
 ```python _2_train_base.py --model=models.densenet169 --dropout=0.2 --lr=0.0001```

\_2_train_densenet_attention_q2l.py是“densenet+对标签和特征进行注意力机制+AsymmetricLoss”的次优的复杂方案（在验证集的F1是0.47），欲运行_2_train_densenet_attention_q2l.py：
 ```python _2_train_densenet_attention_q2l.py```
