import tensorflow as tf
import numpy as np


class Hier_rnn_model(object):
    def __init__(self, config, name_scope, dtype=tf.float32):
        #with tf.variable_scope(name_or_scope=scope_name):
        emb_dim = config.embed_dim
        num_layers = config.num_layers
        vocab_size = config.vocab_size
        #max_len = config.max_len
        num_class = config.num_class
        buckets = config.buckets
        self.lr = config.lr
        self.global_step = tf.Variable(initial_value=0, trainable=False)

        self.query = []
        self.answer = []
        for i in range(buckets[-1][0]):
            self.query.append(tf.placeholder(dtype=tf.int32, shape=[None], name="query{0}".format(i)))
        for i in xrange(buckets[-1][1]):
            self.answer.append(tf.placeholder(dtype=tf.int32, shape=[None], name="answer{0}".format(i)))

        self.target = tf.placeholder(dtype=tf.int64, shape=[None], name="target")

        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(emb_dim)
        encoder_mutil = tf.nn.rnn_cell.MultiRNNCell([encoder_cell] * num_layers)
        encoder_emb = tf.nn.rnn_cell.EmbeddingWrapper(encoder_mutil, embedding_classes=vocab_size, embedding_size=emb_dim)

        context_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=emb_dim)
        context_multi = tf.nn.rnn_cell.MultiRNNCell([context_cell] * num_layers)

        self.b_query_state = []
        self.b_answer_state = []
        self.b_state = []
        self.b_logits = []
        self.b_loss = []
        #self.b_cost = []
        self.b_train_op = []
        for i, bucket in enumerate(buckets):
            with tf.variable_scope(name_or_scope="Hier_RNN_encoder", reuse=True if i > 0 else None) as var_scope:
                query_output, query_state = tf.nn.rnn(encoder_emb, inputs=self.query[:bucket[0]], dtype=tf.float32)
                # output [max_len, batch_size, emb_dim]   state [num_layer, 2, batch_size, emb_dim]
                var_scope.reuse_variables()
                answer_output, answer_state = tf.nn.rnn(encoder_emb, inputs=self.answer[:bucket[1]], dtype=tf.float32)
                self.b_query_state.append(query_state)
                self.b_answer_state.append(answer_state)
                context_input = [query_state[-1][1], answer_state[-1][1]]

            with tf.variable_scope(name_or_scope="Hier_RNN_context", reuse=True if i > 0 else None):
                output, state = tf.nn.rnn(context_multi, context_input, dtype=tf.float32)
                self.b_state.append(state)
                top_state = state[-1][1]  # [batch_size, emb_dim]

            with tf.variable_scope("Softmax_layer_and_output", reuse=True if i > 0 else None):
                softmax_w = tf.get_variable("softmax_w", [emb_dim, num_class], dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b", [num_class], dtype=tf.float32)
                logits = tf.matmul(top_state, softmax_w) + softmax_b
                self.b_logits.append(logits)

            with tf.name_scope("loss"):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.target)
                mean_loss = tf.reduce_mean(loss)
                self.b_loss.append(mean_loss)

            with tf.name_scope("gradient_descent"):
                disc_params = [var for var in tf.trainable_variables() if name_scope in var.name]
                grads, norm = tf.clip_by_global_norm(tf.gradients(mean_loss, disc_params), config.max_grad_norm)
                #optimizer = tf.train.GradientDescentOptimizer(self.lr)
                optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                train_op = optimizer.apply_gradients(zip(grads, disc_params), global_step=self.global_step)
                self.b_train_op.append(train_op)

        all_variables = [v for v in tf.global_variables() if name_scope in v.name]
        self.saver = tf.train.Saver(all_variables)


class Config(object):
    embed_dim = 12
    lr = 0.1
    num_class = 2
    train_dir = './disc_data/'
    name_model = "disc_model"
    tensorboard_dir = "./tensorboard/disc_log/"
    name_loss = "disc_loss"
    num_layers = 3
    vocab_size = 10
    max_len = 50
    batch_size = 1
    init_scale = 0.1
    buckets = [(5, 10), (10, 15), (20, 25), (40, 50), (50, 50)]
    max_grad_norm = 5


def main(_):
    with tf.Session() as sess:
        query = [[1],[2],[3],[4],[5]]
        answer = [[6],[7],[8],[9],[0],[0],[0],[0],[0],[0]]
        target = [1]
        config = Config
        initializer = tf.random_uniform_initializer(-1 * config.init_scale, 1 * config.init_scale)
        with tf.variable_scope(name_or_scope="rnn_model", initializer=initializer):
            model = Hier_rnn_model(config, name_scope=config.name_model)
            sess.run(tf.global_variables_initializer())
        input_feed = {}
        for i in range(config.buckets[0][0]):
            input_feed[model.query[i].name] = query[i]
        for i in range(config.buckets[0][1]):
            input_feed[model.answer[i].name] = answer[i]
        input_feed[model.target.name] = target

        fetches = [model.b_train_op[0], model.b_query_state[0],  model.b_state[0], model.b_logits[0]]

        train_op, query, state, logits = sess.run(fetches=fetches, feed_dict=input_feed)

        print("query: ", np.shape(query))

    pass

if __name__ == '__main__':
    tf.app.run()



