import os

import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.hier_disc as h_disc
import random
import utils.conf as conf
import utils.data_utils as data_utils

gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config
steps_per_checkpoint = 100
# pre train discriminator
def disc_pre_train():
    #discs.train_step(disc_config, evl_config)
    h_disc.hier_train(disc_config, evl_config)

# pre train generator
def gen_pre_train():
    gens.train(gen_config)

# prepare disc_data for discriminator and generator
def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                    encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False):
    # sample_inputs, sample_labels, responses = gens.gen_sample(sess, gen_config, gen_model, vocab,
    #                                            source_inputs, source_outputs, mc_search=mc_search)
    #outputs = gens.decoder(sess, gen_config, gen_model, vocab, encoder_inputs, decoder_inputs, target_inputs, mc_search)
    #step(self, session, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True, up_reward=False, reward=None, mc_search=False, debug=True):
    train_query, train_answer = [], []
    query_len = gen_config.buckets[bucket_id][0]
    answer_len = gen_config.buckets[bucket_id][1]

    for query, answer in zip(source_inputs, source_outputs):
        query = query[:query_len] + [int(data_utils.PAD_ID)] * (query_len - len(query) if query_len > len(query) else 0)
        train_query.append(query)
        answer = answer[:answer_len] + [int(data_utils.PAD_ID)] * (answer_len - len(answer) if answer_len > len(answer) else 0)
        train_answer.append(answer)
        train_labels = [1 for _ in source_inputs]
    def decoder(num_roll):
        # sample_inputs = np.hstack((source_inputs, source_outputs))
        # sample_labels = [1 for _ in xrange(gen_config.batch_size)]
        #sample_inputs.append(train_data)
        for _ in xrange(num_roll):
            _, _, output_logits = gen_model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id,
                                                 forward_only=True, mc_search=mc_search)

            # output_logits: [seq_len, batch_size, emb_dim]
            # outputs: [seq_len, batch_size] ==(transpose)==> [batch_size, seq_len]
            outputs = [np.argmax(logit, axis=1) for logit in output_logits]
            # for output in outputs:
            #     print ("output len: ", len(output))
            outputs = np.transpose(outputs)

            for i, output in enumerate(outputs):
                train_query.append(train_query[i])
                train_answer.append(output)
                train_labels.append(0)

        return train_query, train_answer, train_labels

    if mc_search:
        train_query, train_answer, train_labels = decoder(gen_config.beam_size)
    else:
        train_query, train_answer, train_labels = decoder(1)

    print("disc_train_data, mc_search: ", mc_search)
    # for query, answer, label in zip(train_query, train_answer, train_labels):
    #     print(str(label) + "\t" + str(query) + ":\t" + str(answer))

    return train_query, train_answer, train_labels

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob

# discriminator api
def disc_step(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=False):
    feed_dict={}
    print("query len : ", len(train_query))
    for i in xrange(len(train_query)):
        #print("train_query ", train_query[i])
        feed_dict[disc_model.query[i].name] = train_query[i]

    print("answer len : ", len(train_answer))
    for i in xrange(len(train_answer)):
        #print("train_answer ", train_answer[i])
        feed_dict[disc_model.answer[i].name] = train_answer[i]

    feed_dict[disc_model.target.name]=train_labels

    if forward_only:
        fetches = [disc_model.b_logits[bucket_id]]
        logits = sess.run(fetches, feed_dict)
        logits = logits[0]
    else:
        fetches = [disc_model.b_train_op[bucket_id], disc_model.b_cost[bucket_id], disc_model.b_logits[bucket_id]]
        train_op, cost, logits = sess.run(fetches,feed_dict)

    print("logits shape: ", np.shape(logits))

    # softmax operation
    logits = np.transpose(softmax(np.transpose(logits)))

    #print("logits: ", logits)
    #print("train_labels: ", train_labels)
    reward = 0.0
    for logit, label in zip(logits, train_labels):
        reward += logit[label]
    reward = reward / len(train_labels)

    print("the train  and the train reward is  " ,reward)
    return reward

# Adversarial Learning for Neural Dialogue Generation
def al_train():
    with tf.Session() as sess:
        current_step = 1
        disc_model = h_disc.create_model(sess, disc_config)
        gen_model = gens.create_model(sess, gen_config)
        vocab, rev_vocab, dev_set, train_set = gens.prepare_data(gen_config)
        for set in train_set:
            print("train len: ", len(set))

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]

        while True:
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])
            disc_config.max_len = gen_config.buckets[bucket_id][0] + gen_config.buckets[bucket_id][1]
            print("===========================Update Discriminator================================")
            # 1.Sample (X,Y) from real disc_data
            print("bucket_id: %d" %bucket_id)

            encoder_inputs, decoder_inputs, target_weights, source_inputs, source_outputs = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size)
            print ("source_inputs: ", len(source_inputs))
            print ("source_outputs: ", len(source_outputs))
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_query, train_answer, train_labels = disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                                                        encoder_inputs, decoder_inputs, target_weights, bucket_id, mc_search=False)
            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            disc_step(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=False)

            print("===============================Update Generator================================")
            # 1.Sample (X,Y) from real disc_data
            update_gen_data = gen_model.get_batch(train_set, bucket_id, gen_config.batch_size)
            encoder, decoder, weights, source_inputs, source_outputs = update_gen_data

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            train_query, train_answer, train_labels = disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs,
                                                                encoder, decoder, weights, bucket_id, mc_search=True)
            train_query = np.transpose(train_query)
            train_answer = np.transpose(train_answer)
            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward = disc_step(sess, bucket_id, disc_model, train_query, train_answer, train_labels, forward_only=True)

            # 4.Update G on (X, ^Y ) using reward r
            _, loss, a =gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False,
                   reward=reward, debug=True)
            print("up_reward: ", a)

            # 5.Teacher-Forcing: Update G on (X, Y )
            _, loss, a = gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False)
            print("loss: ", loss)
            print("normal: ", a)

            if current_step % steps_per_checkpoint == 0:
                print("save disc model")
                disc_ckpt_dir = os.path.abspath(os.path.join(disc_config.data_dir, "checkpoints"))
                if not os.path.exists(disc_ckpt_dir):
                    os.makedirs(disc_ckpt_dir)
                disc_model_path = os.path.join(disc_ckpt_dir, "disc.model")
                disc_model.saver.save(sess, disc_model_path, global_step=disc_model.global_step)

                print("save gen model")
                gen_ckpt_dir = os.path.abspath(os.path.join(gen_config.data_dir, "checkpoints"))
                if not os.path.exists(gen_ckpt_dir):
                    os.makedirs(gen_ckpt_dir)
                gen_model_path = os.path.join(gen_ckpt_dir, "gen.model")
                gen_model.saver.save(sess, gen_model_path, global_step=gen_model.global_step)
            current_step += 1

def main(_):
    #disc_pre_train()
    #gen_pre_train()
    al_train()

if __name__ == "__main__":
    tf.app.run()
