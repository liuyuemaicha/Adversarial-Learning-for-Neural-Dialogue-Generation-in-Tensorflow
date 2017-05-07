import tensorflow as tf
import numpy as np
import os
import time
import datetime
import random
import utils.data_utils as data_utils
from hier_rnn_model import Hier_rnn_model
from tensorflow.python.platform import gfile
import sys

sys.path.append("../utils")


def evaluate(session, model, config, evl_inputs, evl_labels, evl_masks):
    total_num = len(evl_inputs[0])

    fetches = [model.correct_num, model.prediction, model.logits, model.target]
    feed_dict = {}
    for i in xrange(config.max_len):
        feed_dict[model.input_data[i].name] = evl_inputs[i]
    # feed_dict[model.input_data]=evl_inputs
    feed_dict[model.target.name] = evl_labels
    feed_dict[model.mask_x.name] = evl_masks
    # model.assign_new_batch_size(session,len(evl_inputs))
    # state = session.run(model._initial_state)
    # for i , (c,h) in enumerate(model._initial_state):
    #     feed_dict[c]=state[i].c
    #     feed_dict[h]=state[i].h
    correct_num, prediction, logits, target = session.run(fetches, feed_dict)

    print("total_num: ", total_num)
    print("correct_num: ", correct_num)
    print("prediction: ", prediction)
    # print("logits: ", logits)
    print("target: ", target)

    accuracy = float(correct_num) / total_num
    return accuracy


def hier_read_data(query_path, answer_path, gen_path, max_len=None):
    query_set = []
    answer_set = []
    gen_set = []
    with gfile.GFile(query_path, mode="r") as query_file:
        with gfile.GFile(answer_path, mode="r") as answer_file:
            with gfile.GFile(gen_path, mode="r") as gen_file:
                query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()
                counter = 0
                while query and answer and gen:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  reading disc_data line %d" % counter)
                    query = [int(id) for id in query.strip().split()]
                    query = query[:max_len] + [data_utils.PAD_ID] * (max_len - len(query) if max_len > len(query) else 0)
                    #print("query: ", query)
                    query_set.append(query)
                    answer = [int(id) for id in answer.strip().split()]
                    answer = answer[:max_len] + [data_utils.PAD_ID] * (max_len - len(answer) if max_len > len(answer) else 0)
                    answer_set.append(answer)
                    gen = [int(id) for id in gen.strip().split()]
                    gen = gen[:max_len] + [data_utils.PAD_ID] * (max_len - len(gen) if max_len > len(gen) else 0)
                    gen_set.append(gen)
                    query, answer, gen = query_file.readline(), answer_file.readline(), gen_file.readline()

    return query_set, answer_set, gen_set


def hier_get_batch(config, max_set, query_set, answer_set, gen_set):
    batch_size = config.batch_size
    if batch_size % 2 == 1:
        return IOError("Error")
    train_query = []
    train_answer = []
    label = []
    half_size = batch_size / 2
    for _ in range(half_size):
        index = random.randint(0, max_set)
        train_query.append(query_set[index])
        train_answer.append(answer_set[index])
        label.append(1)
        train_query.append(query_set[index])
        train_answer.append(gen_set[index])
        label.append(0)
    return train_query, train_answer, label


def create_model(sess, config, initializer=None, name="disc_model"):
    with tf.variable_scope(name_or_scope=name, initializer=initializer):
        model = Hier_rnn_model(config=config)
        disc_ckpt_dir = os.path.abspath(os.path.join(config.data_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(disc_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Hier Disc model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Created Hier Disc model with fresh parameters.")
            sess.run(tf.global_variables_initializer())
        return model


def hier_train_step(config_disc, config_evl):
    config = config_disc
    eval_config = config_evl
    eval_config.keep_prob = 1.0

    print("begin training")

    with tf.Graph().as_default(), tf.Session() as session:
        model = create_model(session, config)

        train_path = os.path.join(config.data_dir, "train")
        voc_file_path = [train_path + ".query", train_path + ".answer", train_path + ".gen"]
        vocab_path = os.path.join(config.data_dir, "vocab%d.all" % config.vocabulary_size)
        data_utils.create_vocabulary(vocab_path, voc_file_path, config.vocabulary_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        print("Preparing train disc_data in %s" % config.data_dir)
        train_query_path, train_answer_path, train_gen_path, dev_query_path, dev_answer_path, dev_gen_path = \
            data_utils.hier_prepare_disc_data(config.data_dir, vocab, config.vocabulary_size)

        query_set, answer_set, gen_set = hier_read_data(train_query_path, train_answer_path, train_gen_path, config.max_len)
        #dev_query_set, dev_answer_set, dev_gen_set = hier_read_data(dev_query_path, dev_answer_path, dev_gen_path)
        print("query_set: ", len(query_set))
        current_time = 1
        while True:
            train_query, train_answer, label = hier_get_batch(config, len(query_set)-1, query_set, answer_set,
                                                                  gen_set)
            train_query = np.reshape(train_query, (config.max_len, -1))
            train_answer = np.reshape(train_answer, (config.max_len, -1))

            feed_dict = {}
            for i in xrange(config.max_len):
                feed_dict[model.query[i].name] = train_query[i]
                feed_dict[model.answer[i].name] = train_answer[i]
            feed_dict[model.target.name] = label
            #feed_dict[model.forward_only.name] = False

            fetches = [model.train_op, model.logits, model.loss, model.cost, model.target]
            train_op, logits, loss, cost, target = session.run(fetches, feed_dict)

            if current_time % 200 == 0:
                #print("norm: ", norm)
                print("train_op: ", train_op)
                print("logits: ", logits)
                print("target: ", target)
                print("cost: ", cost)
                disc_ckpt_dir = os.path.abspath(os.path.join(config.data_dir, "checkpoints"))
                if not os.path.exists(disc_ckpt_dir):
                    os.makedirs(disc_ckpt_dir)
                disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                model.saver.save(session, disc_model_path, global_step=model.global_step)
            current_time += 1

