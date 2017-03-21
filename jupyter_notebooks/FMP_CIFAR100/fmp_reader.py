""" File reader """

"""
https://ischlag.github.io/2016/11/07/tensorflow-input-pipeline-for-large-datasets/
https://ischlag.github.io/2016/06/19/tensorflow-input-pipeline-example/

"""
import tensorflow as tf
import numpy as np
from metrics import *

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

def dummy(x):
    return np.float32([int(m == n) for n in x for m in range(100)]).reshape(len(x), 100)


def model_train(model, input_placeholder, output_placeholder, train, val, batch_size, num_epochs=10, save_path="/tensorflow/dr",summary_dir = None):
    """
    input:
    input_placeholder: A placeholder to feed the network with images
    output_placeholder: A placeholder to feed the true labels of the input images
    keep_prob: A placeholder to feed the network with probability value . useful for generalization and avoiding overfitting.
    drop_out: dropout need to be added
    input_image_shape: the input shape of the image, A tuple (h,w)
    actual_train: the actual training data.
    train: the oversampled train data. Incase of No oversampling it is the same as actual train
    val: list of val images locations
    labels_frame: the truth labels of the input images, index as image location.
    train_step: the optimizer to back prop the error
    accuracy: the accuracy of the model
    batch_size: the size of the batch
    num_epochs: total number of epochs
    save_path: path to save the model at different checkpoints
    summary_dir: path to write the various summaries written.

    output:
    save model at defined checkpoints, prints the val_accuracy after every epoch

    process:
    1) calculate the total number of iterations
    2) ensemble both input_pipelines defined in reader.py
    3) save all the variables
    4) merge all the summaries
    5) initiate the network
    6) in session , pass a batch of images to the network and train the model
       - calculate train_summaries and add to the summaries_dir
       - After every-epoch, run the eval pipeline for one epoch and calculate the accuracy and print it.
       - save the model
    """
    num_epochs_per_decay = 2.0
    iterations = (int(50000/batch_size))*num_epochs
    decay_steps = int(50000 / batch_size *
                      num_epochs_per_decay)

    if tf.gfile.Exists(summary_dir):
        tf.gfile.DeleteRecursively(summary_dir)


    #decay_steps = tf.constant(decay_steps, name="decay_steps")
    with tf.name_scope("cross_entropy"):
        out = tf.nn.softmax_cross_entropy_with_logits(labels=output_placeholder, logits = model)
    with tf.name_scope("total"):
        cross_entropy = tf.reduce_mean(out)
    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("train"):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_step = optimizer.minimize(cross_entropy, global_step = global_step)
        tf.add_to_collection("train_step", train_step)

    tf.summary.scalar("global_step", global_step)
    #tf.summary.scalar("learning_rate", learning_rate)
    #tf.summary.scalar("optimizer", optimizer)


    ## setting up the three pipelines
    train_images, train_labels = train_input_pipeline(train,batch_size = batch_size, num_epochs = num_epochs)

    #Pipeline for calculating training accuracy TA (training accuracy)

    TA_images , TA_labels = eval_input_pipeline(train,batch_size = batch_size, num_epochs = num_epochs+1)

    #Pipeline for calculating Validation accuracy VA (validation accuracy)
    VA_images , VA_labels = eval_input_pipeline(val, batch_size, num_epochs+1)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 20)
    merged = tf.summary.merge_all()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        training_writer = tf.summary.FileWriter(summary_dir + '/training',sess.graph)
        train_writer = tf.summary.FileWriter(summary_dir+'/train')
        valid_writer = tf.summary.FileWriter(summary_dir + '/valid')

        if tf.gfile.Exists(save_path):
            filename = tf.train.latest_checkpoint(save_path)
            print ("[Restoring the last checkpoint with filename]",filename)
            saver.restore(sess,filename)
            #train_step = tf.get_collection("train_step")[0]
            print ("model restored")
            g = sess.run(global_step)
            print ("[global step] : ", g)
            print ("starting from epoch:", int(batch_size*g/50000))

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        print ("[queue runners started]")

        try:
            print ("[Training has initiated]")
            i, step = 0 , 0
            step_tb = 0
            while not coord.should_stop():
                images, labels = sess.run([train_images, train_labels])
                summary, _  = sess.run([merged, train_step], feed_dict={input_placeholder:images, output_placeholder:dummy(labels)})
                training_writer.add_summary(summary, i)
                i +=1
                if i ==1 or i == step * int(50000/batch_size):
                    #### calculating training and testing accuracy
                    tb , t_p, t_t, v_p, v_t = metrics(step_tb, merged , sess, train_writer, valid_writer, input_placeholder, output_placeholder, val, batch_size, model, TA_images, TA_labels, VA_images, VA_labels)
                    step_tb += tb
                    train_kappa = quadratic_weighted_kappa(t_t, t_p, min_rating = 0, max_rating =99)
                    test_kappa = quadratic_weighted_kappa(v_t, v_p, min_rating = 0, max_rating =99)
                    print("step:", i, "train_kappa:", train_kappa, "valid_kappa:", test_kappa,
                    "train_acc:",accuracy_score(t_t, t_p), "valid_acc:", accuracy_score(v_t, v_p) )

                    if not tf.gfile.IsDirectory(save_path):
                        tf.gfile.MkDir(save_path)
                    checkpoint_path = os.path.join(save_path, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    tf.train.write_graph(sess.graph_def, '/tmp/fmp_model', 'train_{}.pb'.format(step))
                    tf.train.write_graph(sess.graph_def, '/tmp/fmp_model', 'train_{}.pbtxt'.format(step), as_text = True)
                    step +=1
        except tf.errors.OutOfRangeError:
            print("Done training -- Maximum epoch limit has reached")
        finally:
            coord.request_stop()

        coord.request_stop()
        coord.join(threads)


def metrics(step, merged, sess, train_writer, valid_writer, input_placeholder, output_placeholder, val, batch_size, model, TA_images, TA_labels, VA_images, VA_labels):
    """
    """
    t_p, t_t , v_p , v_t = [], [],[], []
    for j in range(int(10000/batch_size)):

        train_images, train_labels = sess.run([TA_images, TA_labels])
        valid_images, valid_labels = sess.run([VA_images, VA_labels])


        #caclutating train summary
        t_summary, train_true, train_pred = sess.run([merged, output_placeholder, tf.nn.softmax(model)], \
        feed_dict={input_placeholder: train_images, output_placeholder: dummy(train_labels)})
        train_writer.add_summary(t_summary, step+j)

        #calculating validation summary
        v_summary, valid_true, valid_pred = sess.run([merged, output_placeholder, tf.nn.softmax(model)], feed_dict={input_placeholder: valid_images, output_placeholder: dummy(valid_labels)})
        valid_writer.add_summary(v_summary, step+j)

        ## Adding all the outputs to our empty lists
        t_p.append(train_pred)
        t_t.append(train_true)
        v_p.append(valid_true)
        v_t.append(valid_pred)

    return j , np.argmax(np.concatenate(t_p), axis=1), np.argmax(np.concatenate(t_t), axis=1), np.argmax(np.concatenate(v_p), axis=1), np.argmax(np.concatenate(v_t), axis=1)



# def model_train(input_placeholder, output_placeholder , train, val, labels, logits, batch_size, num_epochs=1, summary_dir = None):
#     iterations = (int(len(train)/batch_size))*num_epochs
#
#     labels = tf.squeeze(labels, axis = 1)
#     labels = tf.cast(labels,tf.int64)
#     with tf.name_scope("cross_entropy"):
#         out = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits = logits)
#         with tf.name_scope("total"):
#             cross_entropy = tf.reduce_mean(out)
#     tf.summary.scalar("cross_entropy", cross_entropy)
#
#     with tf.name_scope("train"):
#         train_step = tf.train.AdamOptimizer(0.0002).minimize(cross_entropy)
#
#     with tf.name_scope("accuracy"):
#         with tf.name_scope("correct_prediction"):
#             correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
#         with tf.name_scope("accuracy"):
#             accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.summary.scalar("accuracy", accuracy)
#
#
#     train_images, train_labels = train_input_pipeline(train,batch_size = batch_size, num_epochs = num_epochs)
#     eval_images , eval_labels = eval_input_pipeline(val, batch_size = batch_size, num_epochs = num_epochs+1)
#
#     merged = tf.summary.merge_all()
#     init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     with tf.Session() as sess:
#         sess.run(init_op)
#         train_writer = tf.summary.FileWriter(summary_dir + '/train',sess.graph)
#         test_writer = tf.summary.FileWriter(summary_dir + '/test')
#
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(coord = coord)
#         print ("[queue runners started]")
#
#         try:
#             print ("[Training has initiated]")
#             i, step = 0 , 1
#             while not coord.should_stop():
#                 images, labels = sess.run([train_images, train_labels])
#                 summary, _ = sess.run([merged, train_step], feed_dict={input_placeholder:images, output_placeholder:labels})
#                 train_writer.add_summary(summary, i)
#                 if i ==0: print ("ALL Working Fine")
#                 i +=1
#                 if i == step * int(len(train)/batch_size):
#                     val_acc = []
#                     for j in range(int(len(val)/(batch_size))):
#                         images,labels = sess.run([eval_images, eval_labels])
#                         summary_val,acc  = sess.run([merged, accuracy], feed_dict={input_placeholder:images, output_placeholder:labels})
#                         test_writer.add_summary(summary_val, i+j)
#                         val_acc.append(acc)
#                     print("step=", i, "val_acc=", sum(val_acc)/len(val_acc))
#                     step +=1
#         except tf.errors.OutOfRangeError:
#             print("Done training -- Maximum epoch limit has reached")
#         finally:
#             coord.request_stop()
#
#         coord.request_stop()
#         coord.join(threads)
