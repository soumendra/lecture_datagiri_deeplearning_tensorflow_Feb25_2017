"""We will writing a class based implementation of tensorflow in a less verbose way"""
import tensorflow as tf

class Model:
    def __init__(self, data, target, test_data, test_target, model):
        self.data = data
        self.target = target
        self.x = tf.placeholder(tf.float32, [None,self.data.shape[1]])
        self.y = tf.placeholder(tf.float32, [None, self.target.shape[1]])
        self.model = model(self.x,self.y)
        self.test_data = test_data
        self.target_data = test_target
        self._optimize = None
        self._error = None


    def prediction(self,m):
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 2)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.init(init_op)
            if tf.gfile.Exists(save_path):
                filename = tf.train.latest_checkpoint(save_path)
                print ("[Restoring the last checkpoint with filename]",filename)
                saver.restore(sess,filename)
            prediction = sess.run([tf.nn.softmax(self.model)], feed_dict = {self.x:m})
        return prediction

    def optimize(self, optimizer= "sgd", learning_rate = 0.0001):
        """
        calculate the loss and minimize the loss using an optimizer
        """
        if not self._optimize:
            #self._error = -tf.reduce_sum(self.y*tf.log(tf.nn.softmax(self.model)+ 1e-9))
            #self._error = -tf.reduce_sum(self.y*tf.nn.log_softmax(self.model))
            self._error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.model, labels=self.y))

            # self._error = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(tf.nn.softmax(self.model)+1e-8),
            #                                      reduction_indices=1))
            #out = tf.nn.softmax_cross_entropy_with_logits(logits = self.model, labels=self.y)
            #self._error= tf.reduce_mean(out)
            tf.summary.scalar("cross_entropy", self._error)
            if optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif optimizer == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            elif optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(learning_rate)
            elif optimizer == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(learning_rate)
            elif optimizer == "adagrad":
                optimizer = tf.train.AdagradOptimizer(learning_rate)
            elif optimizer == "ftrl":
                optimizer = tf.train.FtrlOptimizer(learning_rate)
            self._optimize = optimizer.minimize(self._error)


    def train(self, batch_size = 32, epochs = 20, summary_dir = None , save_path = None):
        """
        initialize all the variables, train the network

        inputs:
        batch_size: the number of images used at a time to pass through the network
        epochs: the total number of times each images is passed through the network
        """
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 2)
        merged = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            if tf.gfile.Exists(save_path):
                filename = tf.train.latest_checkpoint(save_path)
                print ("[Restoring the last checkpoint with filename]",filename)
                saver.restore(sess,filename)

            training_writer = tf.summary.FileWriter(summary_dir + '/training',sess.graph)
            total_iter = int(len(self.data)/batch_size)*epochs
            for i in range(total_iter):
                #print ("[Total Iterations]",len(self.data)/batch_size)
                summary, ce, _ = sess.run([merged, self._error,self._optimize], feed_dict={self.x: self.data[i*batch_size:(i+1)*batch_size,], self.y: self.target[i*batch_size:(i+1)*batch_size,]})
                training_writer.add_summary(summary, i)
                if i % 100 == 0:
                    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.model), 1), tf.argmax(self.y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print ("Accuracy:", accuracy.eval({self.x: self.test_data, self.y: self.target_data}), "Error:", ce)
            if save_path:
                if not tf.gfile.IsDirectory(save_path):
                        tf.gfile.MkDir(save_path)
                checkpoint_path = os.path.join(save_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
        print("[The Algorithm is optimized]")
