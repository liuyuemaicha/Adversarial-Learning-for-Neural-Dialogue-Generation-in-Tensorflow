import os

import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.discriminator as discs

import utils.conf as conf

gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config

# pre train discriminator
def disc_pre_train():
    discs.train_step(disc_config, evl_config)

# pre train generator
def gen_pre_train():
    gens.train(gen_config)

# prepare data for discriminator and generator
def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs, mc_search=False):
    sample_inputs, sample_labels, responses = gens.gen_sample(sess, gen_config, gen_model, vocab,
                                               source_inputs, source_outputs, mc_search=mc_search)
    print("disc_train_data, mc_search: ", mc_search)
    for input, label in zip(sample_inputs, sample_labels):
        print(str(label) + "\t" + str(input))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    sorted_index = len_argsort(sample_inputs)
    train_set_x = [sample_inputs[i] for i in sorted_index]
    train_set_y = [sample_labels[i] for i in sorted_index]
    train_set=(train_set_x,train_set_y)
    new_train_set_x=np.zeros([len(train_set[0]),disc_config.max_len])
    #print("new_train_set: ", np.shape(new_train_set_x))
    new_train_set_y=np.zeros(len(train_set[0]))
    #print("new_train_set_y: ", np.shape(new_train_set_y))
    mask_train_x=np.zeros([disc_config.max_len,len(train_set[0])])

    def padding_and_generate_mask(x,y,new_x,new_y,new_mask_x):
        for i,(x,y) in enumerate(zip(x,y)):
            #whether to remove sentences with length larger than maxlen
            if len(x)<=disc_config.max_len:
                new_x[i,0:len(x)]=x
                new_mask_x[0:len(x),i]=1
                new_y[i]=y
            else:
                new_x[i]=(x[0:disc_config.max_len])
                new_mask_x[:,i]=1
                new_y[i]=y
        new_set =(new_x,new_y,new_mask_x)
        del new_x,new_y
        return new_set

    train_inputs, train_labels, train_masks =padding_and_generate_mask(train_set[0],train_set[1],
                                                                     new_train_set_x,new_train_set_y,mask_train_x)
    return train_inputs, train_labels, train_masks, responses

# discriminator api
def disc_step(sess, disc_model, train_inputs, train_labels, train_masks):
    feed_dict={}
    feed_dict[disc_model.input_data]=train_inputs
    feed_dict[disc_model.target]=train_labels
    feed_dict[disc_model.mask_x]=train_masks
    disc_model.assign_new_batch_size(sess,len(train_inputs))
    fetches = [disc_model.cost,disc_model.accuracy,disc_model.train_op,disc_model.summary]
    state = sess.run(disc_model._initial_state)
    for i , (c,h) in enumerate(disc_model._initial_state):
        feed_dict[c]=state[i].c
        feed_dict[h]=state[i].h
    cost,accuracy,_,summary = sess.run(fetches,feed_dict)
    print("the train cost is: %f and the train accuracy is %f ."%(cost, accuracy))
    return accuracy

# Adversarial Learning for Neural Dialogue Generation
def al_train():
    gen_config.batch_size = 1
    with tf.Session() as sess:
        disc_model = discs.create_model(sess, disc_config, is_training=True)
        gen_model = gens.create_model(sess, gen_config, forward_only=True)
        vocab, rev_vocab, dev_set, train_set = gens.prepare_data(gen_config)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])

            print("===========================Update Discriminator================================")
            # 1.Sample (X,Y) from real data
            _, _, _, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, 0)
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_inputs, train_labels, train_masks, _ = disc_train_data(sess,gen_model,vocab,
                                                        source_inputs,source_outputs,mc_search=False)
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            disc_step(sess, disc_model, train_inputs, train_labels, train_masks)

            print("===============================Update Generator================================")
            # 1.Sample (X,Y) from real data
            update_gen_data = gen_model.get_batch(train_set, bucket_id, 0)
            encoder, decoder, weights, source_inputs, source_outputs = update_gen_data

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            train_inputs, train_labels, train_masks, responses = disc_train_data(sess,gen_model,vocab,
                                                        source_inputs,source_outputs,mc_search=True)
            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward = disc_step(sess, disc_model, train_inputs, train_labels, train_masks)

            # 4.Update G on (X, ^Y ) using reward r
            dec_gen = responses[0][:gen_config.buckets[bucket_id][1]]
            if len(dec_gen)< gen_config.buckets[bucket_id][1]:
                dec_gen = dec_gen + [0]*(gen_config.buckets[bucket_id][1] - len(dec_gen))
            dec_gen = np.reshape(dec_gen, (-1,1))
            gen_model.step(sess, encoder, dec_gen, weights, bucket_id, forward_only=False,
                   up_reward=True, reward=reward, debug=True)

            # 5.Teacher-Forcing: Update G on (X, Y )
            _, loss, _ = gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False, up_reward=False)
            print("loss: ", loss)

        #add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(disc_config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "disc.model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        pass

def main(_):
    #disc_pre_train()
    #gen_pre_train()
    al_train()

if __name__ == "__main__":
  tf.app.run()
