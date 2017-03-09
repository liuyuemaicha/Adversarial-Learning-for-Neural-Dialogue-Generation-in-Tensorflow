import tensorflow as tf
import numpy as np
import os
import time
import datetime
import disc_rnn_model as disc_rnn_model
import utils.data_helper as data_helper
import utils.conf as conf
import sys
sys.path.append('../utils')

def create_model(session, config, is_training):
    """Create translation model and initialize or load parameters in session."""
    model = disc_rnn_model.disc_rnn_model(config=config,is_training=True)

    ckpt = tf.train.get_checkpoint_state(config.train_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created Disc_RNN model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def evaluate(model,session,data, batch_size,global_steps=None,summary_writer=None):

    correct_num=0
    total_num=len(data[0])
    for step, (x,y,mask_x) in enumerate(data_helper.batch_iter(data,batch_size=batch_size)):

         fetches = model.correct_num
         feed_dict={}
         feed_dict[model.input_data]=x
         feed_dict[model.target]=y
         feed_dict[model.mask_x]=mask_x
         model.assign_new_batch_size(session,len(x))
         state = session.run(model._initial_state)
         for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
         count=session.run(fetches,feed_dict)
         correct_num+=count

    accuracy=float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy',accuracy)
    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary,global_steps)
        summary_writer.flush()
    return accuracy

def run_epoch(model,session,data,global_steps,valid_model,valid_data, batch_size, train_summary_writer, valid_summary_writer=None):
    for step, (x,y,mask_x) in enumerate(data_helper.batch_iter(data,batch_size=batch_size)):

        feed_dict={}
        feed_dict[model.input_data]=x
        feed_dict[model.target]=y
        feed_dict[model.mask_x]=mask_x
        model.assign_new_batch_size(session,len(x))
        fetches = [model.cost,model.accuracy,model.train_op,model.summary]
        state = session.run(model._initial_state)
        for i , (c,h) in enumerate(model._initial_state):
            feed_dict[c]=state[i].c
            feed_dict[h]=state[i].h
        cost,accuracy,_,summary = session.run(fetches,feed_dict)
        train_summary_writer.add_summary(summary,global_steps)
        train_summary_writer.flush()
        valid_accuracy=evaluate(valid_model,session,valid_data,global_steps,valid_summary_writer)
        if(global_steps%100==0):
            print("the %i step, train cost is: %f and the train accuracy is %f and the valid accuracy is %f"%(global_steps,cost,accuracy,valid_accuracy))
        global_steps+=1

    return global_steps

def train_step(config_disc, config_evl):

    print("loading the disc train set")
    config = config_disc
    eval_config=config_evl
    eval_config.keep_prob=1.0

    train_data,valid_data,test_data=data_helper.load_data(True, config.max_len,batch_size=config.batch_size)

    print("begin training")

    # gpu_config=tf.ConfigProto()
    # gpu_config.gpu_options.allow_growth=True
    with tf.Graph().as_default(), tf.Session() as session:
        print("model training")
        initializer = tf.random_uniform_initializer(-1*config.init_scale,1*config.init_scale)
        with tf.variable_scope("model",reuse=None,initializer=initializer):
            #model = disc_rnn_model.disc_rnn_model(config=config,is_training=True)
            model = create_model(session, config, is_training=True)

        with tf.variable_scope("model",reuse=True,initializer=initializer):
            #valid_model = disc_rnn_model.disc_rnn_model(config=eval_config,is_training=False)
            #test_model = disc_rnn_model.disc_rnn_model(config=eval_config,is_training=False)
            valid_model = create_model(session, eval_config, is_training=False)
            test_model = create_model(session, eval_config, is_training=False)

        #add summary
        # train_summary_op = tf.merge_summary([model.loss_summary,model.accuracy])
        train_summary_dir = os.path.join(config.out_dir,"summaries","train")
        train_summary_writer =  tf.summary.FileWriter(train_summary_dir,session.graph)

        # dev_summary_op = tf.merge_summary([valid_model.loss_summary,valid_model.accuracy])
        dev_summary_dir = os.path.join(eval_config.out_dir,"summaries","dev")
        dev_summary_writer =  tf.summary.FileWriter(dev_summary_dir,session.graph)

        #add checkpoint
        checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "disc.model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        #saver = tf.train.Saver(tf.all_variables())


        tf.global_variables_initializer().run()
        global_steps=1
        begin_time=int(time.time())

        for i in range(config.num_epoch):
            print("the %d epoch training..."%(i+1))
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)
            model.assign_new_lr(session,config.lr*lr_decay)
            global_steps=run_epoch(model,session,train_data,global_steps,valid_model,
                                   valid_data, config_disc.batch_size, train_summary_writer,dev_summary_writer)

            if i% config.checkpoint_every==0:
                path = model.saver.save(session,checkpoint_prefix,global_steps)
                print("Saved model chechpoint to{}\n".format(path))

        print("the train is finished")
        end_time=int(time.time())
        print("training takes %d seconds already\n"%(end_time-begin_time))
        test_accuracy=evaluate(test_model,session,test_data, config_disc.batch_size)
        print("the test data accuracy is %f"%test_accuracy)
        print("program end!")



def main(_):
    train_step()


if __name__ == "__main__":
    tf.app.run()






