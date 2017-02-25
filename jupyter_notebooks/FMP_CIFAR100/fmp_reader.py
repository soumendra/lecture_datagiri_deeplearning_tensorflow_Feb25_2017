""" File reader """

"""
https://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
https://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

"""
import tensorflow as tf
import numpy as np

def read_my_file_format(filename_queue):
    label_bytes = 2
    height, width, depth = 32, 32, 3
    image_bytes = 32 * 32 * 3

    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)

    label = tf.cast(tf.strided_slice(record_bytes, [1], [label_bytes]), tf.int32)

    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],
    [label_bytes + image_bytes]), [depth, height, width])

    uint8image = tf.transpose(depth_major, [1, 2, 0])
    image = tf.cast(uint8image, tf.float32)
    pad_image = tf.image.pad_to_bounding_box(image, 31, 31, 94, 94)
    return pad_image, label

def train_input_pipeline(filenames, batch_size, num_epochs = None):
    with tf.name_scope("train_input"):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs= num_epochs, shuffle = True)
        example, label = read_my_file_format(filename_queue)
        min_after_dequeue = 100
        capacity = min_after_dequeue+3*batch_size
        example.set_shape([94,94,3])
        label.set_shape([1])
        example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size = batch_size, num_threads = 4, capacity = capacity,
        min_after_dequeue = min_after_dequeue)
        tf.summary.image("images", example_batch)
    return example_batch, label_batch


def eval_input_pipeline(filenames, batch_size, num_epochs = None):
    with tf.name_scope("eval_input"):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs =num_epochs, shuffle = True)
        example, label = read_my_file_format(filename_queue)
        min_after_dequeue = 64*2
        capacity = min_after_dequeue+3*batch_size
        example.set_shape([94,94,3])
        label.set_shape([1])
        example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size = batch_size, num_threads = 4, capacity = capacity,
        min_after_dequeue = min_after_dequeue)
    return example_batch, label_batch


def model_train(input_placeholder, output_placeholder , train, val, train_step, accuracy, batch_size, num_epochs=1, summary_dir = None):
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
                summary, _ = sess.run([merged, train_step], feed_dict={input_placeholder:images, output_placeholder:labels})
                train_writer.add_summary(summary, i)
                if i ==0: print ("ALL Working Fine")
                i +=1
                if i == step * int(len(train)/batch_size):
                    val_acc = []
                    for j in range(int(len(val)/(batch_size))):
                        images,labels = sess.run([eval_images, eval_labels])
                        summary_val,acc  = sess.run([merged, accuracy], feed_dict={input_placeholder:images, output_placeholder:labels})
                        test_writer.add_summary(summary_val, i+j)
                        val_acc.append(acc)
                    print("step=", i, "val_acc=", sum(val_acc)/len(val_acc))
                    step +=1
        except tf.errors.OutOfRangeError:
            print("Done training -- Maximum epoch limit has reached")
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)
