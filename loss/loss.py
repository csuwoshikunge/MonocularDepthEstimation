import tensorflow as tf

def L1loss(pred,gt):
    loss = tf.reduce_mean(tf.abs(pred - gt))
    return loss

# def berhuLoss(pred, gt):
