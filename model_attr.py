# coding:utf-8

import os
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, activations, metrics, models
from util import load_tfrecords
import datetime

ATTRIBUTE = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
             'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
             'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
             'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
             'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']


def make_model():
    image = layers.Input(shape=(160, 160, 3), dtype=tf.float32, name='image')
    hidden = layers.Conv2D(filters=24, kernel_size=5, strides=(2, 2), padding='same', name='conv2d_01')(image)  # (80, 80, 32)
    hidden = layers.BatchNormalization(name='bn_01')(hidden)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2), padding='same', name='conv2d_02')(hidden)  # (40, 40, 64)
    hidden = layers.BatchNormalization(name='bn_02')(hidden)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Conv2D(filters=48, kernel_size=5, strides=(2, 2), padding='same', name='conv2d_03')(hidden)  # (20, 20, 48)
    hidden = layers.BatchNormalization(name='bn_03')(hidden)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.Flatten(name='flatten')(hidden)  # 20 * 20 * 48
    hidden = layers.Dense(units=1024, name='dense_01')(hidden)
    hidden = layers.Activation(activation=tf.nn.leaky_relu)(hidden)
    hidden = layers.BatchNormalization(name='bn_06')(hidden)
    output = layers.Dense(units=len(ATTRIBUTE), activation=activations.sigmoid, name='dense_02')(hidden)
    return models.Model(inputs=[image], outputs=[output], name='model_attribute')


def train(start_step=0, restore=False):
    batch_size = 64
    epochs = 10000

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tfrecords_filename_list = [os.path.join("data/tfrecords_align", filename) for filename in os.listdir("data/tfrecords_align") if filename.endswith(".tfrecords")]
    train_ds = load_tfrecords(tfrecords_filename_list, align=True)
    train_ds = train_ds.map(lambda _batch: {"image": tf.image.decode_jpeg(_batch["image_raw"]), "attribute": [_batch[attr] for attr in ATTRIBUTE]})
    train_ds = train_ds.repeat().batch(batch_size)

    model = make_model()
    lr = optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.95, staircase=False)
    optimizer = optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.99)

    checkpoint = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, f"model/", max_to_keep=5)

    if restore:
        try:
            checkpoint.restore(f"model/ckpt-{start_step}")
            print(f"Restored from model/ckpt-{start_step}")
        except tf.errors.NotFoundError:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            if checkpoint_manager.latest_checkpoint:
                start_step = checkpoint.step.numpy()
                print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
            else:
                start_step = 0
                print("Initializing from scratch.")

    loss_object = losses.BinaryCrossentropy(from_logits=False)
    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.BinaryAccuracy(name='train_accuracy')

    @tf.function(input_signature=[
        tf.TensorSpec(shape=(None, 218, 178, 3), dtype=tf.uint8, name='image'),
        tf.TensorSpec(shape=(None, len(ATTRIBUTE)), dtype=tf.int64, name='attribute')
    ])
    def train_step(images, attributes):
        images = tf.image.resize_with_crop_or_pad(images, 160, 160)
        images = tf.image.convert_image_dtype(images, tf.float32)
        images = tf.subtract(tf.multiply(images, 2.0), 1.0)
        attributes = tf.divide(tf.add(tf.cast(attributes, tf.float32), 1.0), 2.0)
        with tf.GradientTape() as gt:
            predictions = model(images)
            loss = loss_object(attributes, predictions)
        gradients = gt.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(attributes, predictions)

    log_loss = f"log/{current_time}/loss"
    summary_writer_loss = tf.summary.create_file_writer(log_loss)

    train_ds_iter = iter(train_ds)
    for epoch in range(start_step, epochs):
        checkpoint.step.assign_add(1)
        batch = next(train_ds_iter)
        train_step(batch["image"], batch["attribute"])
        if epoch % 100 == 0:
            with summary_writer_loss.as_default():
                tf.summary.scalar('Discriminator Loss', train_loss.result(), step=epoch)
            print(f"Epoch {epoch}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result() * 100}%")
            checkpoint_manager.save()
        train_loss.reset_states()
    model.save(f"model/{current_time}.hdf5", save_format="hdf5")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train(0, False)
