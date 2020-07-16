# coding:utf-8
# DCGAN with batch normalization
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, activations, metrics, models
from util import visualize, load_tfrecords
import datetime


def make_discriminator():
    training = layers.Input(shape=(), dtype=tf.bool, name='training')
    images = layers.Input(shape=(160, 160, 3), dtype=tf.float32, name='image')
    hidden = layers.Conv2D(filters=16, kernel_size=5, strides=(2, 2), padding='same', name='conv2d_01')(images)  # (80, 80, 16)
    hidden = layers.BatchNormalization(name='bn_01')(hidden, training=training)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2), padding='same', name='conv2d_02')(hidden)  # (40, 40, 32)
    hidden = layers.BatchNormalization(name='bn_02')(hidden, training=training)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), padding='same', name='conv2d_03')(hidden)  # (20, 20, 64)
    hidden = layers.BatchNormalization(name='bn_03')(hidden, training=training)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Flatten(name='flatten')(hidden)  # 25600
    hidden = layers.Dense(units=160, name='dense_01')(hidden)  # 160
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Dropout(rate=0.2)(hidden)
    output = layers.Dense(units=1, name='dense_02')(hidden)  # 1
    return models.Model(inputs=[images, training], outputs=[output], name='discriminator')


def make_generator(noise_length):
    noises = layers.Input(shape=(noise_length,), name='noises')  # noises_shape
    hidden = layers.Dense(units=2560, name='dense_01')(noises)  # 2560
    hidden = layers.Dense(units=25600, name='dense_02')(hidden)  # 25600
    hidden = layers.Reshape(target_shape=(20, 20, 64), name='reshape')(hidden)  # (20, 20, 64)
    hidden = layers.Conv2DTranspose(filters=32, kernel_size=5, strides=(2, 2), padding='same', activation=activations.relu, name='deconv2d_01')(hidden)  # (40, 40, 64)
    hidden = layers.Conv2DTranspose(filters=16, kernel_size=5, strides=(2, 2), padding='same', activation=activations.relu, name='deconv2d_02')(hidden)  # (80, 80, 32)
    hidden = layers.Conv2DTranspose(filters=8, kernel_size=5, strides=(2, 2), padding='same', activation=activations.relu, name='deconv2d_03')(hidden)  # (160, 160, 16)
    output = layers.Conv2D(filters=3, kernel_size=5, strides=(1, 1), padding='same', activation=activations.tanh, name='conv2d_01')(hidden)  # (160, 160, 3)
    return models.Model(inputs=[noises], outputs=[output], name='generator')


def train(start_step=0, restore=False, model_name=None):
    batch_size = 64
    noise_length = 128
    epochs = 20000
    model_name = model_name or datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sample_number = 64
    sample_noise = tf.random.normal([sample_number, noise_length])

    os.mkdir(f"sample/{model_name}")
    os.mkdir(f"model/{model_name}")
    os.mkdir(f"log/{model_name}")

    print(f"Save to {model_name}")

    tfrecords_filename_list = [os.path.join("data/tfrecords_align", filename) for filename in os.listdir("data/tfrecords_align") if filename.endswith(".tfrecords")]
    train_ds = load_tfrecords(tfrecords_filename_list, align=True)
    train_ds = train_ds.map(lambda _batch: tf.image.decode_jpeg(_batch["image_raw"]))
    train_ds = train_ds.repeat().batch(batch_size)

    model_dis = make_discriminator()
    model_gen = make_generator(noise_length)

    lr_dis = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0007, decay_steps=1000, decay_rate=0.95, staircase=False)
    lr_gen = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0010, decay_steps=1000, decay_rate=0.95, staircase=False)

    optimizer_dis = optimizers.RMSprop(learning_rate=lr_dis)
    optimizer_gen = optimizers.RMSprop(learning_rate=lr_gen)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer_gen=optimizer_gen, optimizer_dis=optimizer_dis, model_gen=model_gen, model_dis=model_dis)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, f"model/{model_name}/", max_to_keep=5)

    if restore:
        try:
            checkpoint.restore(f"model/{model_name}/ckpt-{start_step}")
            print(f"Restored from model/{model_name}/ckpt-{start_step}")
        except tf.errors.NotFoundError:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            if checkpoint_manager.latest_checkpoint:
                start_step = checkpoint.step.numpy()
                print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
            else:
                start_step = 0
                print("Initializing from scratch.")

    train_loss_dis = metrics.Mean(name='train_loss_dis')
    train_loss_gen = metrics.Mean(name='train_loss_gen')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 218, 178, 3), dtype=tf.uint8, name='image_real')
    ])
    def train_step(image_real):
        image_real = tf.image.resize_with_crop_or_pad(image_real, 160, 160)
        image_real = tf.image.convert_image_dtype(image_real, tf.float32)
        image_real = tf.subtract(tf.multiply(image_real, 2.0), 1.0)
        noises = tf.random.normal([batch_size, noise_length])
        with tf.GradientTape() as gt_dis, tf.GradientTape() as gt_gen:
            image_fake = model_gen(noises)
            output_real = model_dis([image_real, True])
            output_fake = model_dis([image_fake, True])
            loss_dis = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(output_real), output_real) + \
                       losses.BinaryCrossentropy(from_logits=True)(tf.zeros_like(output_fake), output_fake)
            loss_gen = losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(output_fake), output_fake)
        gradients_dis = gt_dis.gradient(loss_dis, model_dis.trainable_variables)
        gradients_gen = gt_gen.gradient(loss_gen, model_gen.trainable_variables)
        optimizer_dis.apply_gradients(zip(gradients_dis, model_dis.trainable_variables))
        optimizer_gen.apply_gradients(zip(gradients_gen, model_gen.trainable_variables))
        train_loss_dis(loss_dis)
        train_loss_gen(loss_gen)

    log_dis = f"log/{model_name}/dis"
    log_gen = f"log/{model_name}/gen"
    log_sample = f"log/{model_name}/sample"
    summary_writer_dis = tf.summary.create_file_writer(log_dis)
    summary_writer_gen = tf.summary.create_file_writer(log_gen)
    summary_writer_sample = tf.summary.create_file_writer(log_sample)

    train_ds_iter = iter(train_ds)
    for epoch in range(start_step, epochs):
        checkpoint.step.assign_add(1)
        image_batch = next(train_ds_iter)
        train_step(image_batch)
        if epoch % 100 == 0:
            with summary_writer_dis.as_default():
                tf.summary.scalar('Discriminator Loss', train_loss_dis.result(), step=epoch)
            with summary_writer_gen.as_default():
                tf.summary.scalar('Generator Loss', train_loss_gen.result(), step=epoch)
            print(f"Epoch {epoch}, Gen Loss: {train_loss_gen.result()}, Dis Loss: {train_loss_dis.result()}")
            samples = model_gen(sample_noise).numpy()
            img = visualize(((samples + 1.) * 127.5).astype("uint8"), height=160, width=160, channel=3)
            img.save(f"sample/{model_name}/{epoch:06d}.jpg")
            with summary_writer_sample.as_default():
                tf.summary.image("sample", tf.expand_dims(tf.convert_to_tensor(np.array(img)), 0), step=epoch)
            checkpoint_manager.save()
        train_loss_dis.reset_states()
        train_loss_gen.reset_states()
    model_gen.save(f"model/{model_name}/generator.hdf5", save_format="hdf5")
    model_dis.save(f"model/{model_name}/discriminator.hdf5", save_format="hdf5")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    train(0, False)
