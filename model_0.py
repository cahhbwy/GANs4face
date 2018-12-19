# coding:utf-8
# DCGAN

from util import *
import tensorflow as tf
from tensorflow import layers as tl


def discriminate(image, reuse=None):
    """
    :param image: (128, 128, 3)
    :param reuse: reuse
    :return:
    """
    with tf.variable_scope("discriminate", reuse=reuse):
        d_00 = tl.conv2d(image, 24, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="dis_00")  # 64x64x24
        d_01 = tl.batch_normalization(d_00, name="dis_01")  # 64x64x24
        d_02 = tl.conv2d(d_01, 48, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="dis_02")  # 32x32x48
        d_03 = tl.batch_normalization(d_02, name="dis_03")  # 32x32x48
        d_04 = tl.conv2d(d_03, 96, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="dis_04")  # 16x16x96
        d_05 = tl.batch_normalization(d_04, name="dis_05")  # 16x16x96
        d_06 = tl.conv2d(d_05, 192, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="dis_06")  # 8x8x192
        d_07 = tl.batch_normalization(d_06, name="dis_07")  # 8x8x192
        d_08 = tl.flatten(d_07, name="dis_08")  # 12288
        d_09 = tl.dense(d_08, 1536, activation=tf.nn.relu, name="dis_09")  # 192
        d_10 = tl.dense(d_09, 1, name="dis_10")  # 1
        return d_10


def generate(rand_input):  # 1024
    with tf.variable_scope("generate"):
        g_00 = tl.dense(rand_input, 8 * 8 * 192, activation=tf.nn.leaky_relu, name="gen_00")
        g_01 = tf.reshape(g_00, [-1, 8, 8, 192], name="gen_02")  # 8x8x192
        g_02 = tl.conv2d_transpose(g_01, 96, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_02")  # 16x16x96
        g_03 = tl.batch_normalization(g_02, name="gen_03")  # 16x16x96
        g_04 = tl.conv2d_transpose(g_03, 48, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_04")  # 32x32x48
        g_05 = tl.batch_normalization(g_04, name="gen_05")  # 32x32x48
        g_06 = tl.conv2d_transpose(g_05, 24, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_06")  # 64x64x24
        g_07 = tl.batch_normalization(g_06, name="gen_07")  # 64x64x24
        g_08 = tl.conv2d_transpose(g_07, 12, 5, (2, 2), "same", activation=tf.nn.leaky_relu, name="gen_08")  # 128x128x12
        g_09 = tl.conv2d_transpose(g_08, 12, 5, (1, 1), "same", activation=tf.nn.leaky_relu, name="gen_09")  # 128x128x12
        g_10 = tl.batch_normalization(g_09, name="gen_10")  # 128x128x12
        g_11 = tl.conv2d_transpose(g_10, 3, 3, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_11")  # 128x128x3
        return g_11


def model(batch_size, real_image, rand_input):
    fake_image = generate(rand_input)
    real = discriminate(real_image)
    fake = discriminate(fake_image, True)
    dis_loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
    gen_loss = -tf.reduce_mean(fake)
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real_image + (1. - alpha) * fake_image
    gradients = tf.gradients(discriminate(interpolates, True), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), 1))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    dis_loss += 10 * gradient_penalty
    return dis_loss, gen_loss, fake_image


def train(start_step=0, restore=False):
    rand_size = 1024
    batch_size = 36

    data_set = read_images("data/image", batch_size)
    iterator = data_set.make_one_shot_iterator()
    m_real_image = iterator.get_next()
    m_rand_input = tf.placeholder(tf.float32, [None, rand_size])
    m_dis_loss, m_gen_loss, m_fake_image = model(batch_size, m_real_image, m_rand_input)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.00001, global_step=global_step, decay_steps=100, decay_rate=0.99)
    dis_op = tf.train.AdamOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars, colocate_gradients_with_ops=True)
    gen_lr = tf.train.exponential_decay(learning_rate=0.00001, global_step=global_step, decay_steps=100, decay_rate=0.99)
    gen_op = tf.train.AdamOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    for step in range(start_step, 10000):
        if step < 500:
            n_dis = 3
            n_gen = 1
        else:
            n_dis = 2
            n_gen = 1
        for _ in range(n_dis):
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            sess.run(dis_op, feed_dict={m_rand_input: v_rand_input, global_step: step})
        for _ in range(n_gen):
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            sess.run(gen_op, feed_dict={m_rand_input: v_rand_input, global_step: step})
        if step % 10 == 0:
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand_input: v_rand_input})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_rand_input = np.random.uniform(-1., 1., (batch_size, rand_size))
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand_input: v_rand_input})
            visualized(v_fake_image, "sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)


if __name__ == '__main__':
    train(0, False)
