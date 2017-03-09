import tensorflow as tf
import numpy as np

class disc_rnn_model(object):

    def __init__(self, config, scope_name="disc_rnn", is_training=True):
        self.scope_name = scope_name
        with tf.variable_scope(self.scope_name):
            self.keep_prob=config.keep_prob
            self.batch_size=tf.Variable(0,dtype=tf.int32,trainable=False)

            max_len=config.max_len
            self.input_data=tf.placeholder(tf.int32,[None,max_len])
            self.target = tf.placeholder(tf.int64,[None])
            self.mask_x = tf.placeholder(tf.float32,[max_len,None])

            class_num=config.class_num
            hidden_neural_size=config.hidden_neural_size
            vocabulary_size=config.vocabulary_size
            embed_dim=config.embed_dim
            hidden_layer_num=config.hidden_layer_num
            self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
            self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

            #build LSTM network

            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_neural_size,forget_bias=0.0,state_is_tuple=True)
            if self.keep_prob<1:
                lstm_cell =  tf.nn.rnn_cell.DropoutWrapper(
                    lstm_cell,output_keep_prob=self.keep_prob
                )

            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*hidden_layer_num,state_is_tuple=True)

            self._initial_state = cell.zero_state(self.batch_size,dtype=tf.float32)

            #embedding layer
            with tf.device("/cpu:0"),tf.name_scope("embedding_layer"):
                embedding = tf.get_variable("embedding",[vocabulary_size,embed_dim],dtype=tf.float32)
                inputs=tf.nn.embedding_lookup(embedding,self.input_data) #[batch_size, max_len, embed_dim]

            if self.keep_prob<1:
                inputs = tf.nn.dropout(inputs,self.keep_prob)

            out_put=[]
            state=self._initial_state
            with tf.variable_scope("LSTM_layer"):
                for time_step in range(max_len):
                    if time_step>0: tf.get_variable_scope().reuse_variables()
                    (cell_output,state)=cell(inputs[:,time_step,:],state)
                    out_put.append(cell_output)

            out_put=out_put*self.mask_x[:,:,None]

            with tf.name_scope("mean_pooling_layer"):

                out_put=tf.reduce_sum(out_put,0)/(tf.reduce_sum(self.mask_x,0)[:,None])

            with tf.name_scope("Softmax_layer_and_output"):
                softmax_w = tf.get_variable("softmax_w",[hidden_neural_size,class_num],dtype=tf.float32)
                softmax_b = tf.get_variable("softmax_b",[class_num],dtype=tf.float32)
                self.logits = tf.matmul(out_put,softmax_w)+softmax_b

            with tf.name_scope("loss"):
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits+1e-10,self.target)
                self.cost = tf.reduce_mean(self.loss)

            with tf.name_scope("accuracy"):
                self.prediction = tf.argmax(self.logits,1)
                correct_prediction = tf.equal(self.prediction,self.target)
                self.correct_num=tf.reduce_sum(tf.cast(correct_prediction,tf.float32))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),name="accuracy")

            #add summary
            loss_summary = tf.summary.scalar("loss",self.cost)
            #add summary
            accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

            if not is_training:
                return

            self.globle_step = tf.Variable(0,name="globle_step",trainable=False)
            self.lr = tf.Variable(0.0,trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          config.max_grad_norm)


            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in zip(grads, tvars):
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            self.grad_summaries_merged = tf.summary.merge(grad_summaries)

            self.summary =tf.summary.merge([loss_summary,accuracy_summary,self.grad_summaries_merged])



            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            optimizer.apply_gradients(zip(grads, tvars))
            self.train_op=optimizer.apply_gradients(zip(grads, tvars))

            self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")
            self._lr_update = tf.assign(self.lr,self.new_lr)

            #all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
            all_variables = [k for k in tf.global_variables() if self.scope_name in k.name]
            self.saver = tf.train.Saver(all_variables)

    def assign_new_lr(self,session,lr_value):
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
    def assign_new_batch_size(self,session,batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})



















