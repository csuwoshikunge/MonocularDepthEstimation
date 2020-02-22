import tensorflow as tf
import os

def DataAugment(img, dep):
    img, dep = FlipTransform(img, dep)
    # ... other augment
    return img, dep

def FlipTransform(img, dep):
    fliprand = tf.random_uniform([], 0., 1., dtype=tf.float32)
    img = tf.cond(fliprand > 0.5, lambda: img, lambda: tf.reverse(img, [1]))
    dep = tf.cond(fliprand > 0.5, lambda: dep, lambda: tf.reverse(dep, [1]))
    return img, dep

def LoadData(filelist,
             bs,
             data_dir,
             img_size):
    f = open(filelist, 'r')
    img_dir = []
    dep_dir = []
    for line in f.readlines():
        img, dep = line.strip().split(' ')
        img = data_dir + img
        dep = data_dir + dep
        if os.path.exists(img):
            img_dir.append(img)
            dep_dir.append(dep)
    # parsing the path string to tf tensor
    img_dir_queue, dep_dir_queue = tf.train.slice_input_producer([img_dir, dep_dir], shuffle=True, capacity=1000)
    img_q = tf.read_file(img_dir_queue)
    dep_q = tf.read_file(dep_dir_queue)
    # parsing to image in tf tensor
    img_q = tf.image.decode_png(img_q, dtype=tf.uint8, channels=3)
    dep_q = tf.image.decode_png(dep_q, dtype=tf.uint16, channels=1)
    dep_q = tf.cast(dep_q, tf.float32)
    dep_q = tf.cast(dep_q / 6000., tf.float32)
    img_q.set_shape([img_size[0], img_size[1], 3])
    dep_q.set_shape([img_size[0], img_size[1], 1])
    img_q, dep_q = DataAugment(img_q, dep_q)
    return tf.train.batch([img_q, dep_q], bs)

