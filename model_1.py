# coding:utf-8
# WGAN-GP

from util import *
import tensorflow as tf
from tensorflow import layers as tl


def discriminate_0(image, reuse=None):  # 128x128x3
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


def generate_0(rand_input):  # 1024
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
        g_09 = tl.batch_normalization(g_08, name="gen_09")  # 128x128x12
        g_10 = tl.conv2d(g_09, 12, 5, (1, 1), "same", activation=tf.nn.leaky_relu, name="gen_10")  # 128x128x12
        g_11 = tl.conv2d(g_10, 3, 3, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_11")  # 128x128x3
        return g_11


def discriminate_1(image, reuse=None):  # 128x128x3
    with tf.variable_scope("discriminate", reuse=reuse):
        d_00 = tl.conv2d(image, 24, 5, (1, 1), "same", activation=None, name="dis_00")  # 128x128x24
        d_01 = tl.conv2d(d_00, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_01")  # 128x128x6
        d_02 = tl.conv2d(d_01, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_02")  # 128x128x6
        d_03 = tl.conv2d(d_02, 24, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_03")  # 128x128x24
        d_04 = tl.batch_normalization(d_03, name="dis_04")  # 128x128x24
        d_05 = tf.add(d_00, d_04, name="dis_05")  # 128x128x24
        d_06 = tl.max_pooling2d(d_05, 2, 2, "same", name="dis_06")  # 64x64x24

        d_07 = tl.conv2d(d_06, 48, 5, (1, 1), "same", activation=None, name="dis_07")  # 64x64x48
        d_08 = tl.conv2d(d_07, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_08")  # 64x64x6
        d_09 = tl.conv2d(d_08, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_09")  # 64x64x6
        d_10 = tl.conv2d(d_09, 48, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_10")  # 64x64x48
        d_11 = tl.batch_normalization(d_10, name="dis_11")  # 64x64x48
        d_12 = tf.add(d_07, d_11, name="dis_12")  # 64x64x48
        d_13 = tl.max_pooling2d(d_12, 2, 2, "same", name="dis_13")  # 32x32x48

        d_14 = tl.conv2d(d_13, 96, 5, (1, 1), "same", activation=None, name="dis_14")  # 32x32x96
        d_15 = tl.conv2d(d_14, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_15")  # 32x32x6
        d_16 = tl.conv2d(d_15, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_16")  # 32x32x6
        d_17 = tl.conv2d(d_16, 96, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_17")  # 32x32x96
        d_18 = tl.batch_normalization(d_17, name="dis_18")  # 32x32x96
        d_19 = tf.add(d_14, d_18, name="dis_19")  # 32x32x96
        d_20 = tl.max_pooling2d(d_19, 2, 2, "same", name="dis_20")  # 16x16x96

        d_21 = tl.conv2d(d_20, 192, 5, (1, 1), "same", activation=None, name="dis_21")  # 16x16x192
        d_22 = tl.conv2d(d_21, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_22")  # 16x16x6
        d_23 = tl.conv2d(d_22, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_23")  # 16x16x6
        d_24 = tl.conv2d(d_23, 192, 3, (1, 1), "same", activation=tf.nn.tanh, name="dis_24")  # 16x16x192
        d_25 = tl.batch_normalization(d_24, name="dis_25")  # 16x16x192
        d_26 = tf.add(d_21, d_25, name="dis_26")  # 16x16x192
        d_27 = tl.max_pooling2d(d_26, 2, 2, "same", name="dis_27")  # 8x8x192

        d_28 = tl.flatten(d_27, name="dis_28")  # 12288
        d_29 = tl.dense(d_28, 1536, activation=tf.nn.tanh, name="dis_29")  # 1536
        d_30 = tl.dense(d_29, 1, activation=None, name="dis_30")  # 1
        return d_30


def generate_1(rand_input):  # 1024
    with tf.variable_scope("generate"):
        g_00 = tl.dense(rand_input, 1536, activation=tf.nn.tanh, name="gen_00")  # 1536
        g_01 = tl.dense(g_00, 12288, activation=tf.nn.tanh, name="gen_01")  # 12288
        g_02 = tf.reshape(g_01, [-1, 8, 8, 192], name="gen_02")  # 8x8x192
        g_03 = tf.image.resize_images(g_02, [16, 16])  # 16x16x192

        g_04 = tl.conv2d(g_03, 96, 5, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_04")  # 16x16x96
        g_05 = tl.conv2d(g_04, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_05")  # 16x16x6
        g_06 = tl.conv2d(g_05, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_06")  # 16x16x6
        g_07 = tl.conv2d(g_06, 96, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_07")  # 16x16x96
        g_08 = tf.add(g_04, g_07, name="gen_08")  # 16x16x96
        g_09 = tf.image.resize_images(g_08, [32, 32])  # 32x32x96

        g_10 = tl.conv2d(g_09, 48, 5, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_10")  # 32x32x48
        g_11 = tl.conv2d(g_10, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_11")  # 32x32x6
        g_12 = tl.conv2d(g_11, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_12")  # 32x32x6
        g_13 = tl.conv2d(g_12, 48, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_13")  # 32x32x48
        g_14 = tf.add(g_10, g_13, name="gen_14")  # 32x32x48
        g_15 = tf.image.resize_images(g_14, [64, 64])  # 64x64x48

        g_16 = tl.conv2d(g_15, 24, 5, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_16")  # 64x64x24
        g_17 = tl.conv2d(g_16, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_17")  # 64x64x6
        g_18 = tl.conv2d(g_17, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_18")  # 64x64x6
        g_19 = tl.conv2d(g_18, 24, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_19")  # 64x64x24
        g_20 = tf.add(g_16, g_19, name="gen_20")  # 64x64x24
        g_21 = tf.image.resize_images(g_20, [128, 128])  # 128x128x24

        g_22 = tl.conv2d(g_21, 12, 5, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_22")  # 128x128x12
        g_23 = tl.conv2d(g_22, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_23")  # 128x128x6
        g_24 = tl.conv2d(g_23, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_24")  # 128x128x6
        g_25 = tl.conv2d(g_24, 12, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_25")  # 128x128x12
        g_26 = tf.add(g_22, g_25, name="gen_20")  # 128x128x12

        g_27 = tl.conv2d(g_26, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_27")  # 128x128x6
        g_28 = tl.conv2d(g_27, 6, 3, (1, 1), "same", activation=tf.nn.tanh, name="gen_28")  # 128x128x6
        g_29 = tl.conv2d(g_28, 3, 3, (1, 1), "same", activation=tf.nn.sigmoid, name="gen_29")  # 128x128x3

        return g_29


def model(batch_size, real_image, rand_input):
    discriminate = discriminate_1
    generate = generate_1
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

    # data_set = read_images("data/image", batch_size)
    data_set = read_tfrecords("/home/walter/PycharmProjects/WGAN_face/data", batch_size)
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
    dis_op = tf.train.RMSPropOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars, colocate_gradients_with_ops=True)
    gen_lr = tf.train.exponential_decay(learning_rate=0.00001, global_step=global_step, decay_steps=100, decay_rate=0.99)
    gen_op = tf.train.RMSPropOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    for step in range(start_step, 10000):
        if step < 100:
            n_dis = 3
            n_gen = 1
        else:
            n_dis = 1
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
