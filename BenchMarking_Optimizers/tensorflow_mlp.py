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


    # def prediction(self,m):
    #     with tf.Session() as sess:
    #         prediction = tf.nn.softmax(self.model(m))
    #     return prediction

    def optimize(self, optimizer= "sgd"):
        """
        calculate the loss and minimize the loss using an optimizer
        """
        if not self._optimize:
            out = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits = self.model)
            self._error= tf.reduce_mean(out)
            if optimizer == "sgd":
                optimizer = tf.train.GradientDescentOptimizer(0.0001)
            elif optimizer == "adam":
                optimizer = tf.train.AdamOptimizer(0.0001)
            elif optimizer == "rmsprop":
                optimizer = tf.train.RMSPropOptimizer(0.0001)
            elif optimizer == "momentum":
                optimizer = tf.train.MomentumOptimizer(0.0001)
            elif optimizer == "adadelta":
                optimizer = tf.train.AdadeltaOptimizer(0.0001)
            elif optimizer == "adagrad":
                optimizer = tf.train.AdagradOptimizer(0.0001)
            elif optimizer == "ftrl":
                optimizer = tf.train.FtrlOptimizer(0.0001)
            self._optimize = optimizer.minimize(self._error)


    def train(self, batch_size = 32, epochs = 20):
        """
        initialize all the variables, train the network

        inputs:
        batch_size: the number of images used at a time to pass through the network
        epochs: the total number of times each images is passed through the network
        """
        saver = tf.train.Saver(tf.global_variables(), max_to_keep = 20)
        merged = tf.summary.merge_all()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            total_iter = int(len(self.data)/batch_size)*epochs
            for i in range(total_iter):
                #print ("[Total Iterations]",len(self.data)/batch_size)
                ce,_ = sess.run([self._error,self._optimize], feed_dict={self.x: self.data[i*batch_size:(i+1)*batch_size,], self.y: self.target[i*batch_size:(i+1)*batch_size,]})
                if i % 100 == 0:
                    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(self.model), 1), tf.argmax(self.y, 1))
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    print ("Accuracy:", accuracy.eval({self.x: self.test_data[:3000], self.y: self.target_data[:3000]}))
        print("[The Algorithm is optimized]")
