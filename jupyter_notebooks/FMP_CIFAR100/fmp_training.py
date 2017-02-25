import tensorflow as tf

def is_training(labels, logits):
    labels = tf.squeeze(labels, axis = 1)
    labels = tf.cast(labels,tf.int64)
    with tf.name_scope("cross_entropy"):
        out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits = logits)
        with tf.name_scope("total"):
            cross_entropy = tf.reduce_mean(out)
    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy, train_step
