# Adversarial-Learning-for-Neural-Dialogue-Generation-in-Tensorflow

the paper: Adversarial Learning for Neural Dialogue Generation    https://arxiv.org/pdf/1701.06547.pdf


the paper translation in Chinese :http://blog.csdn.net/liuyuemaicha/article/details/60581187
## Config:

TensorFlow 0.12.0  Python 2.7

## Introduction to Project: al_neural_dialogue 

### 1.

  
 gen_data：    training data for gen model
 
 disc_data:    training data for disc model
 
  disc：       code about disc model
  
  gen:         code about gen model
  
 utils：       code about data operation and model config

**notice:**

**gen_data** include  chitchat.train.answer, chitchat.train.query, chitchat.dev.answer, chitchat.dev.query (total four files)

**disc_data**  include disc.dev.answer,disc.dev.query, disc.dev.gen 和 disc.train.answer, disc.train.query，disc.tran.gen   (total six files)

**formula of training data**   one sentence one row and splited with space, eg:  i don ' t want to !


### 2.run


python al_neural_dialogue_train.py


**introduction**


def main(_):

'''

    # step_1 training gen model
    
    
    # gen_pre_train()

    # model test
    # gen_test()

    # step_2 gen training data for disc
    # gen_disc()

    # step_3 training disc model
    # disc_pre_train()

    # step_4 training al model
    # al_train()

    # model test
    # gen_test() 
'''

**model introduction**

1、disc model : hierarchical rnn (paper——Building end-to-end dialogue systems using generative hierarchical neural network models)

2、gen model : seq2seq model with attention (GRU cell)

3、method of reward : Monte Carlo Search

4、optimal：Policy Gradient

