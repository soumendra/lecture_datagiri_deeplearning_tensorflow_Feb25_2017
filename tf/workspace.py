## This is a python file 

#define a Architecture - 2 layer neural network
inputs = ...
with tf.name_scope("input_layer"):
  #define the weights
  w1= tf.Variable(tf.random_normal([784, 256]), name = "weights")
  #define the bias
  b1 = tf.Variable(tf.random_normal([256]), name = "bias")
  #multiply inputs with weights
  matmul = tf.matmul(inputs,w1)
  #Add bias to the matmul operation
  layer_1 = tf.add(matmul, b1)
  # Apply the activation function
  layer_1 = tf.nn.relu(layer_1)
  
# 2nd layer 
with tf.name_scope("hidden_layer"):
  w2 =  tf.Variable(tf.random_normal([256, 128]), name = "weights")
  b2 =  tf.Variable(tf.random_normal([128]), name = "bias")
  layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
  layer_2 = tf.nn.relu(layer_2)

with tf.name_scope("output_layer"):
        w_o = tf.Variable(tf.random_normal([128, 10]), name = "weights")
        b_o = tf.Variable(tf.random_normal([10]), name = "bias")
        out_layer = tf.matmul(layer_2, w_o) + b_o

        
 # Calculate the cost 
with tf.name_scope("cost"):
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out_layer, labels=y_true))
tf.summary.scalar("loss", cost)

# calculate the gradient and back-prop the errors
with tf.name_scope("optimizer"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Calculate the Accuracy:
with tf.name_scope("prediction"):
        prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_true,1))

with tf.name_scope("accuracy"):
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)


#Initialize the variables 
init = tf.global_variable_initializer()

#merge all the summary operations to monitor metrics 
merged_summary_op = tf.summary.merge_all()


# How initialize running ? Launch a session 
with tf.Session() as sess:
  sess.run(init)
  
# How to feed data to the network ?
inputs = tf.placeholder(tf.float32, [None,784]) # None so that we can decide on the batch size later
y_true = tf.placeholder(tf.float32, [None, 10])

# How to feed input placeholders ?
sess.run([optimizer], feed_dict = {inputs: train_features, y_true: train_labels})

# How to feed batches of data ?
for batch in range(n_batches):
  sess.run([optimizer], feed_dict = {inputs: train_features[batch], y_true: train_labels[batch]}


