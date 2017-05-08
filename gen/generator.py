from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
from tensorflow.python.platform import gfile
sys.path.append('../utils')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.

def read_data(config, source_path, target_path, max_size=None):
    data_set = [[] for _ in config.buckets]
    with gfile.GFile(source_path, mode="r") as source_file:
        with gfile.GFile(target_path, mode="r") as target_file:
            source, target = source_file.readline(), target_file.readline()
            counter = 0
            while source and target and (not max_size or counter < max_size):
                counter += 1
                if counter % 100000 == 0:
                    print("  reading disc_data line %d" % counter)
                    sys.stdout.flush()
                source_ids = [int(x) for x in source.split()]
                target_ids = [int(x) for x in target.split()]
                target_ids.append(data_utils.EOS_ID)
                for bucket_id, (source_size, target_size) in enumerate(config.buckets): #[bucket_id, (source_size, target_size)]
                    if len(source_ids) < source_size and len(target_ids) < target_size:
                       data_set[bucket_id].append([source_ids, target_ids])
                       break
                source, target = source_file.readline(), target_file.readline()
    return data_set

def create_model(session, gen_config, initializer=None, name="gen_model"):
    """Create translation model and initialize or load parameters in session."""
    with tf.variable_scope(name_or_scope=name, initializer=initializer):
        model = seq2seq_model.Seq2SeqModel(gen_config)
        gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.data_dir, "checkpoints"))
        ckpt = tf.train.get_checkpoint_state(gen_ckpt_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading Gen model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("Created Gen model with fresh parameters.")
            session.run(tf.global_variables_initializer())
        return model

def prepare_data(gen_config):
    train_path = os.path.join(gen_config.data_dir, "chitchat.train")
    voc_file_path = [train_path+".answer", train_path+".query"]
    vocab_path = os.path.join(gen_config.data_dir, "vocab%d.all" % gen_config.vocab_size)
    data_utils.create_vocabulary(vocab_path, voc_file_path, gen_config.vocab_size)
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    print("Preparing Chitchat disc_data in %s" % gen_config.data_dir)
    train_query, train_answer, dev_query, dev_answer = data_utils.prepare_chitchat_data(
        gen_config.data_dir, vocab, gen_config.vocab_size)

    # Read disc_data into buckets and compute their sizes.
    print ("Reading development and training disc_data (limit: %d)."
               % gen_config.max_train_data_size)
    dev_set = read_data(gen_config, dev_query, dev_answer)
    train_set = read_data(gen_config, train_query, train_answer, gen_config.max_train_data_size)

    return vocab, rev_vocab, dev_set, train_set

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob

def train(gen_config):
    vocab, rev_vocab, dev_set, train_set = prepare_data(gen_config)
    for b_set in train_set:
        print("b_set: ", len(b_set))

    with tf.Session() as sess:
    #with tf.device("/gpu:1"):
        # Create model.
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.emb_dim))
        model = create_model(sess, gen_config)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        step_loss_summary = tf.Summary()
        #merge = tf.merge_all_summaries()
        writer = tf.summary.FileWriter("../logs/", sess.graph)

        while True:
            # Choose a bucket according to disc_data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder = model.get_batch(
                train_set, bucket_id, gen_config.batch_size)

            print("bucket_id: ", bucket_id)
            print("encoder_inputs: ", np.shape(encoder_inputs))
            print("decoder_inputs: ", np.shape(decoder_inputs))
            print("target_weights: ", np.shape(target_weights))

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % gen_config.steps_per_checkpoint == 0:

                bucket_value = step_loss_summary.value.add()
                bucket_value.tag = "loss"
                bucket_value.simple_value = float(loss)
                writer.add_summary(step_loss_summary, current_step)

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.data_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                    os.makedirs(gen_ckpt_dir)
                checkpoint_path = os.path.join(gen_ckpt_dir, "chitchat.model")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                # for bucket_id in xrange(len(gen_config.buckets)):
                #   encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                #       dev_set, bucket_id)
                #   _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                #                                target_weights, bucket_id, True)
                #   eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                #   print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                sys.stdout.flush()

def decoder(sess, gen_config, model, vocab, encoder_inputs, decoder_inputs, target_inputs, forward_only=True, mc_search=False):
    pass

def get_predicted_sentence(sess, input_token_ids, vocab, model,
                           beam_size, buckets, mc_search=True,debug=False):
    def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
        #model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        _, _, logits = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, True)
        prob = softmax(logits[dptr][0])
        # print("model_step @ %s" % (datetime.now()))
        return prob

    def greedy_dec(output_logits):
        selected_token_ids = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # if data_utils.EOS_ID in selected_token_ids:
        #   eos = selected_token_ids.index(data_utils.EOS_ID)
        #   selected_token_ids = selected_token_ids[:eos]
        #output_sentence = ' '.join([rev_vocab[t] for t in selected_token_ids])
        return selected_token_ids

    #input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)
    # Which bucket does it belong to?
    bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] > len(input_token_ids)])
    outputs = []
    feed_data = {bucket_id: [(input_token_ids, outputs)]}

    # Get a 1-element batch to feed the sentence to the model.   None,bucket_id, True
    encoder_inputs, decoder_inputs, target_weights, _, _ = model.get_batch(feed_data, bucket_id, 1)
    if debug: print("\n[get_batch]\n", encoder_inputs, decoder_inputs, target_weights)

    ### Original greedy decoding
    if beam_size == 1 or (not mc_search):
        _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
        return [{"dec_inp": greedy_dec(output_logits), 'prob': 1}]

    # Get output logits for the sentence. # initialize beams as (log_prob, empty_string, eos)
    beams, new_beams, results = [(1, {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], []

    for dptr in range(len(decoder_inputs)-1):
      if dptr > 0:
        target_weights[dptr] = [1.]
        beams, new_beams = new_beams[:beam_size], []
      if debug: print("=====[beams]=====", beams)
      heapq.heapify(beams)  # since we will srot and remove something to keep N elements
      for prob, cand in beams:
        if cand['eos']:
          results += [(prob, cand)]
          continue

        all_prob_ts = model_step(encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
        all_prob_t  = [0]*len(all_prob_ts)
        all_prob    = all_prob_ts

        # suppress copy-cat (respond the same as input)
        if dptr < len(input_token_ids):
          all_prob[input_token_ids[dptr]] = all_prob[input_token_ids[dptr]] * 0.01

        # beam search
        for c in np.argsort(all_prob)[::-1][:beam_size]:
          new_cand = {
            'eos'     : (c == data_utils.EOS_ID),
            'dec_inp' : [(np.array([c]) if i == (dptr+1) else k) for i, k in enumerate(cand['dec_inp'])],
            'prob_ts' : cand['prob_ts'] * all_prob_ts[c],
            'prob_t'  : cand['prob_t'] * all_prob_t[c],
            'prob'    : cand['prob'] * all_prob[c],
          }
          new_cand = (new_cand['prob'], new_cand) # for heapq can only sort according to list[0]

          if (len(new_beams) < beam_size):
            heapq.heappush(new_beams, new_cand)
          elif (new_cand[0] > new_beams[0][0]):
            heapq.heapreplace(new_beams, new_cand)

    results += new_beams  # flush last cands

    # post-process results
    res_cands = []
    for prob, cand in sorted(results, reverse=True):
      res_cands.append(cand)
    return res_cands

def gen_sample(sess ,gen_config, model, vocab, source_inputs, source_outputs, mc_search=True):
    sample_inputs = []
    sample_labels =[]
    rep = []

    for source_query, source_answer in zip(source_inputs, source_outputs):
        sample_inputs.append(source_query+source_answer)
        sample_labels.append(1)
        responses = get_predicted_sentence(sess, source_query, vocab,
                                           model, gen_config.beam_size, _buckets, mc_search)

        for resp in responses:
            if gen_config.beam_size == 1 or (not mc_search):
                dec_inp = [dec for dec in resp['dec_inp']]
                rep.append(dec_inp)
                dec_inp = dec_inp[:]
            else:
                dec_inp = [dec.tolist()[0] for dec in resp['dec_inp'][:]]
                rep.append(dec_inp)
                dec_inp = dec_inp[1:]
            print("  (%s) -> %s" % (resp['prob'], dec_inp))
            sample_neg = source_query + dec_inp
            sample_inputs.append(sample_neg)
            sample_labels.append(0)

    return sample_inputs, sample_labels, rep
