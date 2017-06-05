"""
description: this file helps to load raw file and gennerate batch x,y
author:luchi
date:22/11/2016
"""
import numpy as np
import cPickle as pkl

#file path
dataset_path='disc_data/subj0.pkl'

def set_dataset_path(path):
    dataset_path=path




def load_data(debug, max_len,batch_size,n_words=20000,valid_portion=0.1,sort_by_len=True):
    f=open(dataset_path,'rb')
    print ('load disc_data from %s',dataset_path)
    train_set = np.array(pkl.load(f))
    test_set = np.array(pkl.load(f))
    print("train_set: ", np.shape(train_set))
    print("test_set: ", np.shape(test_set))
    f.close()

    train_set_x,train_set_y = train_set

    # w_f = open("corpus", "w")
    # for x, y in zip(train_set_x, train_set_y):
    #     w_f.write(str(x) + "\t" + str(y) + "\n")
    # w_f.close()

    #train_set length
    n_samples= len(train_set_x)
    #shuffle and generate train and valid dataset
    sidx = np.random.permutation(n_samples)
    if debug: print("sidx: ", sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    if debug: print("n_train: ", n_train)

    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    if debug: print("train_set_x[0]: ", train_set_x[0])
    if debug: print("train_set_y[0]: ", train_set_y[0])

    if debug: print("train_set_x[1]: ", train_set_x[1])
    if debug: print("train_set_y[1]: ", train_set_y[1])

    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    if debug: print("train_set shape: ", np.shape(train_set))
    if debug: print("valid_set shape: ", np.shape(valid_set))

    #remove unknow words
    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)



    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]


        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]

    train_set=(train_set_x,train_set_y)
    valid_set=(valid_set_x,valid_set_y)
    test_set=(test_set_x,test_set_y)




    new_train_set_x=np.zeros([len(train_set[0]),max_len])
    if debug: print("new_train_set: ", np.shape(new_train_set_x))
    new_train_set_y=np.zeros(len(train_set[0]))
    if debug: print("new_train_set_y: ", np.shape(new_train_set_y))

    new_valid_set_x=np.zeros([len(valid_set[0]),max_len])
    new_valid_set_y=np.zeros(len(valid_set[0]))

    new_test_set_x=np.zeros([len(test_set[0]),max_len])
    new_test_set_y=np.zeros(len(test_set[0]))

    mask_train_x=np.zeros([max_len,len(train_set[0])])
    mask_test_x=np.zeros([max_len,len(test_set[0])])
    mask_valid_x=np.zeros([max_len,len(valid_set[0])])


    def padding_and_generate_mask(x,y,new_x,new_y,new_mask_x):

        for i,(x,y) in enumerate(zip(x,y)):
            #whether to remove sentences with length larger than maxlen
            if len(x)<=max_len:
                new_x[i,0:len(x)]=x
                new_mask_x[0:len(x),i]=1
                new_y[i]=y
            else:
                new_x[i]=(x[0:max_len])
                new_mask_x[:,i]=1
                new_y[i]=y
        new_set =(new_x,new_y,new_mask_x)
        del new_x,new_y
        return new_set

    train_set=padding_and_generate_mask(train_set[0],train_set[1],new_train_set_x,new_train_set_y,mask_train_x)
    test_set=padding_and_generate_mask(test_set[0],test_set[1],new_test_set_x,new_test_set_y,mask_test_x)
    valid_set=padding_and_generate_mask(valid_set[0],valid_set[1],new_valid_set_x,new_valid_set_y,mask_valid_x)

    return train_set,valid_set,test_set


#return batch dataset
def batch_iter(data,batch_size):

    #get dataset and label
    x,y,mask_x=data
    x=np.array(x)
    y=np.array(y)
    data_size=len(x)
    num_batches_per_epoch=int((data_size-1)/batch_size)
    for batch_index in range(num_batches_per_epoch):
        start_index=batch_index*batch_size
        end_index=min((batch_index+1)*batch_size,data_size)
        return_x = x[start_index:end_index]
        return_y = y[start_index:end_index]
        return_mask_x = mask_x[:,start_index:end_index]
        # if(len(return_x)<batch_size):
        #     print(len(return_x))
        #     print return_x
        #     print return_y
        #     print return_mask_x
        #     import sys
        #     sys.exit(0)
        yield (return_x,return_y,return_mask_x)


