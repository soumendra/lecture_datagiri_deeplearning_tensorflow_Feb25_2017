#this is a pipeline document 


with tf.name_scope("train_input"):
  filenames = tf.train.match_filenames_once("/datalocation/*.png", name = "filenamesholder")
  
  #create a queue of filenames
  filename_queue = tf.train.string_input_producer(filenames, num_epochs = num_epochs, shuffle = True)

with tf.name_scope("read_files"):
  # initate a file reader 
  reader = tf.WholdeFileReader()
  
  # read a file 
  key, record_string = reader.read(filename_queue)
  
  # decode the png and convert into float
  image = tf.image.decode_png(record_string, channels = 3)
  image = tf.cast(image, tf.float32)

with tf.name_scope("pre_process_image"):
  #Apply the pre_processing
  image = pre_process(image) # preprocess using tensorflow functions 

# Queue the images and feed the network with a batch of images 
with tf.name_scope("queue_images"):
  #Establish a queue (An Image storer, which stores images and feed to the network when required)
  min_after_dequeue = 100 # minimum number of images in the bin after dequeue
  capacity = min_after_dequeue + 3* batch_size # max number of images which can be stored in bin
  image.set_shape([36, 36, 3]) # set the shape of images in the queue
  key.set_shape([1]) # set the key(label) shape 
  example_batch, label_batch = tf.train.shuffle_batch([example, label], batch_size = batch_size, num_thread =4,
                                                      capacity = capacity, min_after_dequeue = min_after_dequeue)
  
  # feed example_batch and label_batch to the network

  
# How to Read Binary Files using tensorflow ?
with tf.name_scope("input_data"):
  label_bytes = 2 # CIFAR 100 has 0-99 labels #use 1 for CIFAR-10 (0-9)
  height, width, depth = 32, 32, 2
  image_bytes = 32* 32 * 3
  
  record_bytes = label_bytes + image_bytes 
  
  reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
  key, value = reader.read(filename_queue) #read filename queue.
  
  record_bytes = tf.decode_raw(value, tf.uint8)
  
  label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

   depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes],
                                             [label_bytes + image_bytes]), [depth, height, width])

    uint8image = tf.transpose(depth_major, [1, 2, 0])
    
    image = tf.cast(uint8image, tf.float32)
    
    #Image and label to the file storeage bin - tf.train.shuffle_batch
