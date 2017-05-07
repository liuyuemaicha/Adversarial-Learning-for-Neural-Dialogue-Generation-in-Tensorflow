from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import utils.data_utils as data_utils
import gen.seq2seq as rl_seq2seq
from tensorflow.python.ops import variable_scope

sys.path.append('../utils')

class Seq2SeqModel(object):

  def __init__(self, config, use_lstm=False, num_samples=512, forward=False,
               scope_name='gen_seq2seq', dtype=tf.float32):


    self.scope_name = scope_name
    with tf.variable_scope(self.scope_name):
        self.source_vocab_size = config.vocab_size
        self.target_vocab_size = config.vocab_size
        self.buckets = config.buckets
        self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = config.batch_size
        self.emb_dim = config.emb_dim
        self.num_layers = config.num_layers
        self.max_gradient_norm = config.max_gradient_norm

        #self.up_reward = tf.placeholder(tf.bool, name="up_reward")
        self.mc_search = tf.placeholder(tf.bool, name="mc_search")
        self.forward_only = tf.placeholder(tf.bool, name="forward_only")

        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(self.emb_dim)
        if use_lstm:
          single_cell = tf.nn.rnn_cell.BasicLSTMCell(self.emb_dim)
        cell = single_cell
        if self.num_layers  > 1:
          cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return rl_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols= self.source_vocab_size,
              num_decoder_symbols= self.target_vocab_size,
              embedding_size= self.emb_dim,
              output_projection=output_projection,
              feed_previous=do_decode,
              mc_search=self.mc_search,
              dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(self.buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
        for i in xrange(self.buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None], name="weight{0}".format(i)))
        self.reward = [tf.placeholder(tf.float32, name="reward_%i" % i) for i in range(len(self.buckets))]


        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1] for i in xrange(len(self.decoder_inputs) - 1)]

        self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
            self.encoder_inputs, self.decoder_inputs, targets, self.target_weights,
            self.buckets, self.emb_dim, self.batch_size,
            lambda x, y: seq2seq_f(x, y, tf.select(self.forward_only, True, False)),
            output_projection=output_projection, softmax_loss_function=softmax_loss_function)

        with tf.name_scope("gradient_descent"):
            self.gradient_norms = []
            self.updates = []
            self.gen_params = [p for p in tf.trainable_variables() if self.scope_name in p.name]
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in xrange(len(self.buckets)):
                adjusted_losses = tf.mul(self.losses[b], self.reward[b])
                gradients = tf.gradients(adjusted_losses, self.gen_params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, self.gen_params), global_step=self.global_step))

        # self.saver = tf.train.Saver(tf.all_variables())
        self.gen_variables = [k for k in tf.global_variables() if self.scope_name in k.name]
        self.saver = tf.train.Saver(self.gen_variables)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=True, up_reward=False, reward=None, mc_search=False, debug=True):
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {
        self.forward_only.name: forward_only,
        #self.up_reward.name:  up_reward,
        self.mc_search.name: mc_search
    }
    for l in xrange(len(self.buckets)):
      input_feed[self.reward[l].name] = reward if reward else 1
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only: # normal training
      # output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
      #                self.gradient_norms[bucket_id],  # Gradient norm.
      #                self.losses[bucket_id]]  # Loss for this batch.
        output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else: # testing or reinforcement learning
      output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]  # Loss for this batch.
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], outputs[0]  # Gradient norm, loss, no outputs.
    else:
      return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.

  def get_batch(self, train_data, bucket_id, batch_size,type=0):

    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    #batch_size = self.batch_size
    #print("Batch_Size: %s" %self.batch_size)
    # Get a random batch of encoder and decoder inputs from disc_data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_source_encoder, batch_source_decoder = [], []
    #print("bucket_id: %s" %bucket_id)
    if type == 1:
        batch_size = 1
    for batch_i in xrange(batch_size):
      if type == 1:
        encoder_input, decoder_input = train_data[bucket_id]
      elif type == 2:
        #print("disc_data[bucket_id]: ", disc_data[bucket_id][0])
        encoder_input_a, decoder_input = train_data[bucket_id][0]
        encoder_input = encoder_input_a[batch_i]
      elif type == 0:
          encoder_input, decoder_input = random.choice(train_data[bucket_id])
          print("train en: %s, de: %s" %(encoder_input, decoder_input))

      batch_source_encoder.append(encoder_input)
      batch_source_decoder.append(decoder_input)
      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the disc_data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)

    return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder)
