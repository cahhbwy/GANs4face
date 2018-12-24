# coding:utf-8
# DCGAN

from util import *
import tensorflow as tf
from tensorflow import layers as tl


def discriminate_0(image, label, training, reuse=None):  # 128x128x3
    with tf.variable_scope("discriminate", reuse=reuse):
        d_00 = tl.conv2d(image, 24, 5, (2, 2), "same", activation=None, name="dis_00")  # 64x64x24
        d_01 = tf.nn.leaky_relu(d_00, 0.2, name="dis_01")  # 64x64x24
        d_02 = tl.conv2d(d_01, 48, 5, (2, 2), "same", activation=None, name="dis_02")  # 32x32x48
        d_03 = tl.batch_normalization(d_02, momentum=0.9, epsilon=1e-5, training=training, name="dis_03")  # 32x32x48
        d_04 = tf.nn.leaky_relu(d_03, 0.2, name="dis_04")  # 32x32x48
        d_05 = tl.conv2d(d_04, 96, 5, (2, 2), "same", activation=None, name="dis_05")  # 16x16x96
        d_06 = tl.batch_normalization(d_05, momentum=0.9, epsilon=1e-5, training=training, name="dis_06")  # 16x16x96
        d_07 = tf.nn.leaky_relu(d_06, 0.2, name="dis_07")  # 16x16x96
        d_08 = tl.conv2d(d_07, 192, 5, (2, 2), "same", activation=None, name="dis_08")  # 8x8x192
        d_09 = tl.batch_normalization(d_08, momentum=0.9, epsilon=1e-5, training=training, name="dis_09")  # 8x8x192
        d_10 = tf.nn.leaky_relu(d_09, 0.2, name="dis_10")  # 8x8x192
        d_11 = tl.flatten(d_10, name="dis_11")  # 12288
        d_12 = tl.dense(d_11, 1, activation=None, name="dis_12")  # 1
        loss = tf.losses.sigmoid_cross_entropy(label, d_12)
        return loss


def generate_0(rand_input, training):  # 1024
    with tf.variable_scope("generate"):
        g_00 = tl.dense(rand_input, 8 * 8 * 192, activation=tf.nn.leaky_relu, name="gen_00")  # 12288
        g_01 = tf.reshape(g_00, [-1, 8, 8, 192], name="gen_02")  # 8x8x192
        g_02 = tl.batch_normalization(g_01, momentum=0.9, epsilon=1e-5, training=training, name="gen_02")  # 8x8x192
        g_03 = tf.nn.relu(g_02, name="gen_03")  # 8x8x192
        g_04 = tl.conv2d_transpose(g_03, 96, 5, (2, 2), "same", activation=None, name="gen_04")  # 16x16x96
        g_05 = tl.batch_normalization(g_04, momentum=0.9, epsilon=1e-5, training=training, name="gen_05")  # 16x16x96
        g_06 = tf.nn.relu(g_05, name="gen_06")  # 16x16x96
        g_07 = tl.conv2d_transpose(g_06, 48, 5, (2, 2), "same", activation=None, name="gen_07")  # 32x32x48
        g_08 = tl.batch_normalization(g_07, momentum=0.9, epsilon=1e-5, training=training, name="gen_08")  # 32x32x48
        g_09 = tf.nn.relu(g_08, name="gen_09")  # 32x32x48
        g_10 = tl.conv2d_transpose(g_09, 24, 5, (2, 2), "same", activation=None, name="gen_10")  # 64x64x96
        g_11 = tl.batch_normalization(g_10, momentum=0.9, epsilon=1e-5, training=training, name="gen_11")  # 64x64x96
        g_12 = tf.nn.relu(g_11, name="gen_12")  # 64x64x96
        g_13 = tl.conv2d_transpose(g_12, 12, 5, (2, 2), "same", activation=None, name="gen_13")  # 128x128x12
        g_14 = tl.conv2d(g_13, 3, (1, 1), "same", activation=None, name="gen_14")  # 128x128x3
        g_15 = tf.nn.sigmoid(g_14, name="gen_15")
        return g_15


def model(batch_size, real_image, rand_input):
    discriminate = discriminate_0
    generate = generate_0
    training = tf.placeholder(tf.bool)
    fake_image = generate(rand_input, training)
    dis_loss = discriminate(real_image, tf.ones([batch_size, 1]), training) + discriminate(fake_image, tf.zeros([batch_size, 1]), training, True)
    gen_loss = discriminate(fake_image, tf.ones([batch_size, 1]), training, True)
    return training, dis_loss, gen_loss, fake_image


def train(start_step=0, restore=False):
    rand_size = 1024
    batch_size = 36

    # data_set = read_images("data/image", batch_size)
    data_set = read_tfrecords("/home/walter/PycharmProjects/WGAN_face/data", batch_size)
    iterator = data_set.make_one_shot_iterator()
    m_real_image = iterator.get_next()
    m_rand_input = tf.placeholder(tf.float32, [None, rand_size])
    m_training, m_dis_loss, m_gen_loss, m_fake_image = model(batch_size, m_real_image, m_rand_input)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=100, decay_rate=0.99)
    dis_op = tf.train.AdamOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars)
    gen_lr = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=100, decay_rate=0.99)
    gen_op = tf.train.AdamOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    for step in range(start_step, 10000):
        if step < 50:
            n_dis = 5
            n_gen = 1
        else:
            n_dis = 1
            n_gen = 2
        for _ in range(n_dis):
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            sess.run(dis_op, feed_dict={m_rand_input: v_rand_input, global_step: step, m_training: True})
        for _ in range(n_gen):
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            sess.run(gen_op, feed_dict={m_rand_input: v_rand_input, global_step: step, m_training: True})
        if step % 10 == 0:
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand_input: v_rand_input, m_training: False})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand_input: v_rand_input, m_training: False})
            visualized(v_fake_image, "sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)


if __name__ == '__main__':
    train(0, False)
