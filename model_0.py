# coding:utf-8
# DCGAN with batch normalization
import tensorflow as tf
import numpy as np
from util import visualize, read_images, read_tfrecords
import os


def discriminator(image, training=True, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.1)
    with tf.variable_scope("discriminator", reuse=reuse):
        d_00 = tf.layers.conv2d(image, 12, 5, (2, 2), "same", kernel_initializer=ki, name="dis_00")  # 64*64*12
        # d_01 = tf.layers.batch_normalization(d_00, training=training, name="dis_01")  # 64*64*12
        d_02 = tf.nn.leaky_relu(d_00, name="dis_02")  # 64*64*12
        d_03 = tf.layers.conv2d(d_02, 24, 5, (2, 2), "same", kernel_initializer=ki, name="dis_03")  # 32*32*24
        d_04 = tf.layers.batch_normalization(d_03, training=training, name="dis_04")  # 32*32*24
        d_05 = tf.nn.leaky_relu(d_04, name="dis_05")  # 32*32*24
        d_06 = tf.layers.conv2d(d_05, 48, 5, (2, 2), "same", kernel_initializer=ki, name="dis_06")  # 16*16*48
        # d_07 = tf.layers.batch_normalization(d_06, training=training, name="dis_07")  # 16*16*48
        d_08 = tf.nn.leaky_relu(d_06, name="dis_08")  # 16*16*48
        d_09 = tf.layers.conv2d(d_08, 48, 5, (2, 2), "same", kernel_initializer=ki, name="dis_09")  # 8*8*96
        d_10 = tf.layers.batch_normalization(d_09, training=training, name="dis_10")  # 8*8*96
        d_11 = tf.nn.leaky_relu(d_10, name="dis_11")  # 8*8*96
        d_12 = tf.layers.flatten(d_11, name="dis_12")  # 6144
        d_13 = tf.layers.dense(d_12, 256, name="dis_13")  # 256
        d_14 = tf.layers.batch_normalization(d_13, training=training, name="dis_14")  # 128
        d_15 = tf.nn.leaky_relu(d_14, name="dis_15")  # 256
        d_16 = tf.layers.dense(d_15, 1, name="dis_16")  # 1
        return d_16


def generator(rand_z, training=True, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.1)
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand_z, 6144, name="gen_00")  # 6144
        g_01 = tf.reshape(g_00, [-1, 8, 8, 96], name="gen_01")  # 8*8*96
        g_02 = tf.layers.batch_normalization(g_01, training=training, name="gen_02")
        g_03 = tf.nn.relu(g_02, name="gen_03")
        g_04 = tf.layers.conv2d_transpose(g_03, 48, 5, (2, 2), "same", kernel_initializer=ki, name="gen_04")  # 16*16*48
        # g_05 = tf.layers.batch_normalization(g_04, training=training, name="gen_05")
        g_06 = tf.nn.relu(g_04, name="gen_06")
        g_07 = tf.layers.conv2d_transpose(g_06, 24, 5, (2, 2), "same", kernel_initializer=ki, name="gen_07")  # 32*32*24
        g_08 = tf.layers.batch_normalization(g_07, training=training, name="gen_08")
        g_09 = tf.nn.relu(g_08, name="gen_09")
        g_10 = tf.layers.conv2d_transpose(g_09, 12, 5, (2, 2), "same", kernel_initializer=ki, name="gen_10")  # 64*64*12
        # g_11 = tf.layers.batch_normalization(g_10, training=training, name="gen_11")
        g_12 = tf.nn.relu(g_10, name="gen_12")
        g_13 = tf.layers.conv2d_transpose(g_12, 3, 5, (2, 2), "same", activation=tf.nn.tanh, kernel_initializer=ki, name="gen_13")  # 128*128*3
        return g_13


def model(batch_size: int, real_image_uint8, rand_z_size: int):
    rand_z = tf.placeholder(tf.float32, [None, rand_z_size], name="rand_z")
    training = tf.placeholder(tf.bool)
    real_image_float = tf.divide(tf.cast(real_image_uint8, tf.float32) - 128., 128., name="real_image_float")
    fake_image_float = generator(rand_z, training)
    fake_image_uint8 = tf.cast(tf.clip_by_value(tf.cast(tf.multiply(fake_image_float + 1.0, 128.), tf.int32), 0, 255), tf.uint8)
    dis_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), discriminator(real_image_float, training)) + \
               tf.losses.sigmoid_cross_entropy(tf.zeros([batch_size, 1]), discriminator(fake_image_float, training))
    gen_loss = tf.losses.sigmoid_cross_entropy(tf.ones([batch_size, 1]), discriminator(fake_image_float, training))
    return training, rand_z, fake_image_uint8, dis_loss, gen_loss


def train(start_step, restore):
    batch_size = 64
    rand_z_size = 256

    dataset = read_tfrecords("data/tfrecords", batch_size)
    iterator = dataset.make_one_shot_iterator()
    m_real_image = iterator.get_next()

    m_training, m_rand_z, m_fake_image, m_dis_loss, m_gen_loss = model(batch_size, m_real_image, rand_z_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    gen_lr = tf.train.exponential_decay(learning_rate=0.001, global_step=global_step, decay_steps=100, decay_rate=0.90)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        dis_op = tf.train.AdamOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars)
        gen_op = tf.train.AdamOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    n_dis = 3
    n_gen = 1
    v_sample_rand = np.random.uniform(-1., 1., (256, rand_z_size))
    for step in range(start_step, 10001):
        if step % 10 == 0:
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand_z: v_rand_z, m_training: False})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand_z: v_sample_rand, m_training: False})
            image = visualize(v_fake_image, channel=3, height=128, width=128)
            image.convert("RGB").save("sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)
        for _ in range(n_dis):
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(dis_op, feed_dict={m_rand_z: v_rand_z, global_step: step, m_training: True})
        for _ in range(n_gen):
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(gen_op, feed_dict={m_rand_z: v_rand_z, global_step: step, m_training: True})


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train(0, False)
