# coding:utf-8
# WDCGAN-GP without batch normalization
import tensorflow as tf
import numpy as np
from util import visualize, read_images, read_tfrecords
import os


def discriminator(image, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.01)
    with tf.variable_scope("discriminator", reuse=reuse):
        d_00 = tf.layers.conv2d(image, 12, 3, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_00")  # 64*64*12
        d_01 = tf.layers.conv2d(d_00, 24, 3, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_01")  # 32*32*24
        d_02 = tf.layers.conv2d(d_01, 48, 3, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_02")  # 16*16*48
        d_03 = tf.layers.conv2d(d_02, 96, 3, (2, 2), "same", activation=tf.nn.leaky_relu, kernel_initializer=ki, name="dis_03")  # 8*8*96
        d_04 = tf.layers.flatten(d_03, name="dis_04")  # 6144
        d_05 = tf.layers.dense(d_04, 256, activation=tf.nn.leaky_relu, name="dis_05")  # 256
        d_06 = tf.layers.dense(d_05, 1, name="dis_06")  # 1
        return d_06


def generator(rand_z, reuse=tf.AUTO_REUSE):
    ki = tf.initializers.random_normal(stddev=0.01)
    with tf.variable_scope("generator", reuse=reuse):
        g_00 = tf.layers.dense(rand_z, 6144, activation=tf.nn.relu, name="gen_00")  # 6144
        g_01 = tf.reshape(g_00, [-1, 8, 8, 96], name="gen_01")  # 8*8*96
        g_02 = tf.layers.conv2d_transpose(g_01, 48, 3, (2, 2), "same", activation=tf.nn.relu, kernel_initializer=ki, name="gen_02")  # 16*16*48
        g_03 = tf.layers.conv2d_transpose(g_02, 24, 3, (2, 2), "same", activation=tf.nn.relu, kernel_initializer=ki, name="gen_03")  # 32*32*24
        g_04 = tf.layers.conv2d_transpose(g_03, 12, 3, (2, 2), "same", activation=tf.nn.relu, kernel_initializer=ki, name="gen_04")  # 64*64*12
        g_05 = tf.layers.conv2d_transpose(g_04, 3, 3, (2, 2), "same", activation=tf.nn.tanh, kernel_initializer=ki, name="gen_05")  # 128*128*3
        return g_05


def model(batch_size: int, real_image_uint8, rand_z_size: int):
    rand_z = tf.placeholder(tf.float32, [None, rand_z_size], name="rand_z")
    real_image_float = tf.divide(tf.cast(real_image_uint8, tf.float32) - 128., 128., name="real_image_float")
    fake_image_float = generator(rand_z)
    fake_image_uint8 = tf.cast(tf.clip_by_value(tf.cast(tf.multiply(fake_image_float + 1.0, 128.), tf.int32), 0, 255), tf.uint8)
    dis_loss = tf.reduce_mean(discriminator(fake_image_float)) - \
               tf.reduce_mean(discriminator(real_image_float))
    gen_loss = -tf.reduce_mean(discriminator(fake_image_float))
    alpha = tf.random_uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolates = alpha * real_image_float + (1. - alpha) * fake_image_float
    gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), [1, 2, 3]))  # 除第0维之外的所有维度
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    dis_loss += 10 * gradient_penalty
    return rand_z, fake_image_uint8, dis_loss, gen_loss


def train(start_step, restore):
    batch_size = 64
    rand_z_size = 256

    dataset = read_tfrecords("data/tfrecords", batch_size)
    iterator = dataset.make_one_shot_iterator()
    m_real_image = iterator.get_next()

    m_rand_z, m_fake_image, m_dis_loss, m_gen_loss = model(batch_size, m_real_image, rand_z_size)
    tf.summary.scalar("dis_loss", m_dis_loss)
    tf.summary.scalar("gen_loss", m_gen_loss)
    merged_summary_op = tf.summary.merge_all()

    dis_vars = [var for var in tf.trainable_variables() if "dis" in var.name]
    gen_vars = [var for var in tf.trainable_variables() if "gen" in var.name]

    global_step = tf.Variable(0, trainable=False)
    dis_lr = tf.train.exponential_decay(learning_rate=0.000004, global_step=global_step, decay_steps=100, decay_rate=0.90)
    gen_lr = tf.train.exponential_decay(learning_rate=0.000020, global_step=global_step, decay_steps=100, decay_rate=0.90)

    dis_op = tf.train.RMSPropOptimizer(dis_lr).minimize(m_dis_loss, var_list=dis_vars, colocate_gradients_with_ops=True)
    gen_op = tf.train.RMSPropOptimizer(gen_lr).minimize(m_gen_loss, var_list=gen_vars, colocate_gradients_with_ops=True)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter("log", sess.graph)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=10)
    if restore:
        saver.restore(sess, "model/model.ckpt-%d" % start_step)

    v_sample_rand = np.random.uniform(-1., 1., (256, rand_z_size))
    for step in range(start_step, 2001):
        if step < 5 or step % 100 == 0:
            n_dis = 100
            n_gen = 1
        else:
            n_dis = 5
            n_gen = 1
        if step % 10 == 0:
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            merged_summary, v_dis_loss, v_gen_loss = sess.run([merged_summary_op, m_dis_loss, m_gen_loss], feed_dict={m_rand_z: v_rand_z})
            print("step %6d, dis_loss = %f, gen_loss = %f" % (step, v_dis_loss, v_gen_loss))
            summary_writer.add_summary(merged_summary, step)
        if step % 100 == 0:
            v_fake_image = sess.run(m_fake_image, feed_dict={m_rand_z: v_sample_rand})
            image = visualize(v_fake_image, channel=3, height=128, width=128)
            image.convert("RGB").save("sample/%06d.jpg" % step)
            saver.save(sess, "model/model.ckpt", global_step=step)
        for _ in range(n_dis):
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(dis_op, feed_dict={m_rand_z: v_rand_z, global_step: step})
        for _ in range(n_gen):
            v_rand_z = np.random.uniform(-1., 1., (batch_size, rand_z_size))
            sess.run(gen_op, feed_dict={m_rand_z: v_rand_z, global_step: step})


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train(0, False)
