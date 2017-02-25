""" File reader """

"""
https://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
https://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

"""
import tensorflow as tf
import numpy as np

def read_my_file_format(filename_queue):
    reader = tf.WholeFileReader()
    key, record_string = reader.read(filename_queue)
    example = tf.image.decode_png(record_string,channels = 3)
    example = tf.cast(example, tf.float32)
    example = tf.image.pad_to_bounding_box(example,4,4,36,36)
    label = tf.reshape(key, [1], name=None)
    return example, label

def train_input_pipeline(filenames, batch_size, num_epochs = None):
    with tf.name_scope("train_input"):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs= num_epochs, shuffle = True)
        example, label = read_my_file_format(filename_queue)
        min_after_dequeue = 10000
        capacity = min_after_dequeue+3*batch_size
        example.set_shape([36,36,3])
        label.set_shape([1])
        example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size = batch_size, num_threads = 4, capacity = capacity,
        min_after_dequeue = min_after_dequeue)
    return example_batch, label_batch


def eval_input_pipeline(filenames, batch_size, num_epochs = None):
    with tf.name_scope("eval_input"):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs =num_epochs, shuffle = True)
        example, label = read_my_file_format(filename_queue)
        min_after_dequeue = 64*2
        capacity = min_after_dequeue+3*batch_size
        example.set_shape([36,36,3])
        label.set_shape([1])
        example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size = batch_size, num_threads = 4, capacity = capacity,
        min_after_dequeue = min_after_dequeue)
    return example_batch, label_batch


def array_extract(array, tlab):
    x = array.astype(str)
    x= [tlab.ix[x[i]].values for i in range(len(x))]
    x = np.concatenate(x)
    return x


def model_train(input_placeholder, output_placeholder , train, val, labels_frame, train_step, accuracy, batch_size, num_epochs=1, summary_dir = None):
    iterations = (int(len(train)/batch_size))*num_epochs

    train_images, train_labels = train_input_pipeline(train,batch_size = batch_size, num_epochs = num_epochs)
    eval_images , eval_labels = eval_input_pipeline(val, batch_size = batch_size, num_epochs = num_epochs+1)

    merged = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        train_writer = tf.summary.FileWriter(summary_dir + '/train',sess.graph)
        test_writer = tf.summary.FileWriter(summary_dir + '/test')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        print ("[queue runners started]")

        try:
            print ("[Training has initiated]")
            i, step = 0 , 1
            while not coord.should_stop():
                images, labels = sess.run([train_images, train_labels])
                labels_dummy = np.float32(array_extract(labels, labels_frame))
                summary, _ = sess.run([merged, train_step], feed_dict={input_placeholder:images, output_placeholder:labels_dummy})
                train_writer.add_summary(summary, i)
                i +=1
                if i == step * int(len(train)/batch_size):
                    val_acc = []
                    for j in range(int(len(val)/(batch_size))):
                        images,labels = sess.run([eval_images, eval_labels])
                        labels_dummy =  np.float32(array_extract(labels, labels_frame))
                        summary_val,acc  = sess.run([merged, accuracy], feed_dict={input_placeholder:images, output_placeholder:labels_dummy})
                        test_writer.add_summary(summary_val, j)
                        val_acc.append(acc)
                    print("step=", i, "val_acc=", sum(val_acc)/len(val_acc))
                    step +=1
        except tf.errors.OutOfRangeError:
            print("Done training -- Maximum epoch limit has reached")
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
