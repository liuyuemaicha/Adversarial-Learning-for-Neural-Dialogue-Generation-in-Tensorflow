该项目根据论文 Adversarial Learning for Neural Dialogue Generation 实现，论文地址：https://arxiv.org/pdf/1701.06547.pdf

配置说明： 

TensorFlow版本 0.12.0  Python版本 2.7

al_neural_dialogue项目说明：

1.目录名

  data：         存放了分类器的预训练数据；
  
 train_data：    存放了生成器的预训练数据；
 
  disc：         分类器模型的相关代码文件；
  
  gen:           生成器模型的相关代码文件；
  
 utils：         数据操作和配置相关代码文件

2.文件

al_neural_dialogue_train.py  :   对抗学习的训练代码文件

3.运行

进入al_neural_dialogue项目后，运行当前目录下的al_neural_dialogue_train.py文件： python al_neural_dialogue_train.py

文件"al_neural_dialogue_train.py"中main函数说明

def main(_):
    #disc_pre_train()   预训练分类器；
    #gen_pre_train()   预训练生成器；
    al_train()        训练对抗学习模型；	
	
	
模型算法说明：

1、分类器模型使用的是多层LSTM模型（层数可以自己配置）

2、生成器模型使用的是Seq2Seq模型（TensorFlow自带）

3、对抗学习训练中reward值的计算方式使用的Monte Carlo Search（beam search实现）

