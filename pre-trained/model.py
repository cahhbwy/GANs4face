# coding:utf-8

from util import read_tfrecords
import tensorflow as tf
from tensorflow import layers as tl


def model(image, label):
    """
    :param image: 218x178x3
    :param label: 1
    """
    h_00 = tl.conv2d(image, 16, 3, (1, 1), "valid", activation=tf.nn.relu, name="conv2d_00")  # 216x176x16
    h_01 = tl.max_pooling2d(h_00, 2, 2, "valid", name="pool_01")  # 108x88x16
    h_02 = tl.conv2d(h_01, 64, 3, (1, 1), "same", activation=tf.nn.relu, name="conv2d_02")  # 108x88x64
    h_04 = tl.max_pooling2d(h_02, 2, 2, "valid", name="pool_03")  # 54x44x64
    h_05 = tl.conv2d(h_04, 64, 5, (1, 1), "valid", activation=tf.nn.relu, name="conv2d_05")  # 50x40x128
    h_06 = tl.conv2d(h_05, 32, 3, (1, 1), "same", activation=tf.nn.relu, name="conv2d_06")  # 50x40x128
    h_07 = tl.max_pooling2d(h_06, 2, 2, "valid", name="pool_04")  # 25x20x32
    h_08 = tl.flatten(h_07, name="flatten_08")  # 16000
    h_09 = tl.dense(h_08, 2000, activation=tf.nn.leaky_relu, name="dense_09")  # 2000
    h_10 = tl.dropout(h_09, 0.3, name="dropout_10")  # 2000
    h_11 = tl.dense(h_10, 500, name="dense_11")  # 500
    h_12 = tl.dense(h_11, 1, name="dense_12")  # 1
    loss = tf.losses.sigmoid_cross_entropy(label, h_12)
    predict = tf.cast(tf.greater(h_12, 0), tf.int64)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))
    return loss, predict, accuracy


def train(start_step=0, restore=False):
    batch_size = 64
    data_set = read_tfrecords("/home/walter/PycharmProjects/WGAN_face/pre-trained/data", batch_size)
    iterator = data_set.make_one_shot_iterator()
    next_batch = iterator.get_next()
    m_loss, m_predict, m_accuracy = model(next_batch["image"], next_batch["label"])

    tf.summary.scalar("loss", m_loss)
    merged_summary_op = tf.summary.merge_all()

    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(lr).minimize(m_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    for step in range(start_step, 10000):
        if step % 100 == 0:
            merged_summary, v_loss = sess.run([merged_summary_op, m_loss])
            print("step = %6d, loss = %f" % (step, v_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 1000 == 0:
            v_accuracy = sess.run(m_accuracy)
            print("step = %6d, accuracy = %f" % (step, v_accuracy))
            saver.save(sess, "model/model.ckpt", global_step=step)
        sess.run(op)


if __name__ == '__main__':
    train()
