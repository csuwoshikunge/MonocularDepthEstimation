import os
import tensorflow as tf
from models import ResNet50UpProj
from utils import saveplot
from dataloader import LoadData
from loss import L1loss
import cv2
import argparse

parser = argparse.ArgumentParser(description='depth estimation')
parser.add_argument('--IMAGE_SIZE', type=int, default=(480,640))
parser.add_argument('--INPUT_SIZE', type=int, default=(240,320),help="the input size of network")
parser.add_argument('--RANDOM_SEED', type=int, default=256)
parser.add_argument('--LEARNING_RATE', type=float, default=1.e-3)
parser.add_argument('--STEPS', type=int, default=10000)
parser.add_argument('--BS', type=int, default=2)
parser.add_argument('--FILELIST', type=str, default='E:/datasets/NYUV2/NYUv2ParsingData/trainlist.txt')
parser.add_argument('--DATA_DIR', type=str, default='E:/datasets/NYUV2/NYUv2ParsingData/')
parser.add_argument('--RESTORE_FROM', type=str, default='./pretrained')
parser.add_argument('--LOGDIR', type=str, default='./logs/')
parser.add_argument('--GPUS', type=int, default=1)
parser.add_argument('--SAVE_INTERVAL', type=int, default=1000)
args = parser.parse_args()

def Net(input,
        seed,
        reuse,
        trainable):
    net = ResNet50UpProj(input,seed,reuse,trainable)
    output = net.build_model()
    return output

def average_gradients(tower_grads):
    avg_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grads = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grads, axis=0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        avg_grads.append(grad_and_var)
    return avg_grads

def train_single_GPU():
    tf.set_random_seed(args.RANDOM_SEED)
    img_batch, dep_batch = LoadData(args.FILELIST,args.BS,args.DATA_DIR,args.IMAGE_SIZE)
    img_batch = tf.image.resize_nearest_neighbor(img_batch,args.INPUT_SIZE)
    dep_batch = tf.image.resize_nearest_neighbor(dep_batch,args.INPUT_SIZE)
    # learning rate poly strategy
    lr_ph = tf.placeholder(tf.float32, shape=[])
    base_lr = tf.constant(args.LEARNING_RATE)
    learning_rate = tf.scalar_mul(base_lr, tf.pow(1. - lr_ph / args.STEPS, 0.9))
    # summaries
    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.image('img', img_batch, max_outputs=4)
    tf.summary.image('dep', dep_batch, max_outputs=4)
    #optimizer
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    out = Net(input=tf.cast(img_batch,tf.float32),
              seed=args.RANDOM_SEED,
              reuse=False,
              trainable=True) # with shape (bs,120,160,1)
    out = tf.image.resize_bilinear(out,args.INPUT_SIZE)
    loss = L1loss(out, dep_batch)
    grads = tf.gradients(loss, tf.trainable_variables())
    train_op = opt.apply_gradients(zip(grads, tf.trainable_variables()))
    '''
    config = tf.ConfigProto(allow_soft_placement=True, log_divice_placement=False)
    config.gpu_optimis.allow_growth = True
    sess = tf.Session(config=config)
    '''
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1000)
    '''
    loader = tf.train.Saver(varlist=restore_variables)
    loader.restore(args.RESTORE_FROM)
    '''
    # start queue
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.LOGDIR, graph=tf.get_default_graph())
    # training
    for step in range(args.STEPS):
        summary, loss_value, input_img, pred_dep, lr_value, _ = sess.run([merged,
                                                                loss,
                                                                img_batch,
                                                                out,
                                                                learning_rate,
                                                                train_op], feed_dict={lr_ph:step})
        if step%args.SAVE_INTERVAL==0:
            saver.save(sess, args.LOGDIR+'model.ckpt', global_step=step,write_meta_graph=False)
        summary_writer.add_summary(summary, step)
        print('step %d:'%step + ': loss=%s'%str(loss_value) + ', learning rate=%s'%str(lr_value))
        if step%50==0:
            saveplot(pred_dep[0,:,:,0], 'temp_dep.png')
            cv2.imwrite('temp_img.png', input_img[0,:,:,::-1])
    # end training queue
    coord.request_stop()
    coord.join(threads)

def train_multi_GPUs():
    tf.set_random_seed(args.RANDOM_SEED)
    img_batch, dep_batch = LoadData(args.FILELIST, args.BS, args.DATA_DIR, args.IMAGE_SIZE)
    img_batch = tf.image.resize_nearest_neighbor(img_batch, args.INPUT_SIZE)
    dep_batch = tf.image.resize_nearest_neighbor(dep_batch, args.INPUT_SIZE)
    # learning rate poly strategy
    lr_ph = tf.placeholder(tf.float32, shape=[])
    base_lr = tf.constant(args.LEARNING_RATE)
    learning_rate = tf.scalar_mul(base_lr, tf.pow(1. - lr_ph / args.STEPS, 0.9))
    # optimizer
    opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # it is the same as train_single_GPU to here

    #multi gpus
    tower_grads = []
    with tf.variables_scope(tf.get_variable_scope()):
        for i in range(args.NUM_GPUS):
            with tf.device('/gpu:%d'%i) as scope:
                #batch data on each gpu
                img_batch_tower = img_batch[i*args.BS :(i+1)*args.BS,:,:]
                dep_batch_tower = dep_batch[i*args.BS :(i+1)*args.BS,:,:]

                all_trainable = tf.trainable_variables()
                out = Net()
                loss = L1loss(out, dep_batch_tower)
                # if restore:
                #     restore_var = ...
                cur_grads = tf.gradients(loss, all_trainable)
                tower_grads.append(cur_grads)
    
    grads = average_gradients(tower_grads)
    train_op = opt.apply_gradients(zip(grads, all_trainable))
    # others are the same as single GPU training

if __name__=='__main__':
    train_single_GPU()