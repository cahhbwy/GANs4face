# coding:utf-8
import os
import tensorflow as tf
import numpy as np
from PIL import Image


def write_tfrecords(image_dir, batch_size, save_dir="data"):
    filename_list = os.listdir(image_dir)
    for i in range(0, len(filename_list), batch_size):
        filename = "%06d.tfrecords" % i
        writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, filename))
        for j in range(min(batch_size, len(filename_list) - i)):
            img = Image.open(os.path.join(image_dir, filename_list[i + j]))
            if img.size != (178, 218):
                continue
            img = img.crop((0, 20, 178, 198)).resize((128, 128))
            item = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()]))
                    }
                )
            )
            writer.write(item.SerializeToString())
        writer.close()
        print("write %s finished" % filename)


def parse_data(proto):
    features = tf.parse_single_example(
        proto, features={
            "image": tf.FixedLenFeature([], tf.string),
        }
    )
    img = tf.reshape(tf.decode_raw(features["image"], tf.uint8), (128, 128, 3))
    return tf.divide(tf.cast(img, tf.float32), 255.)


def read_tfrecords(data_dir, batch_size):
    filename_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".tfrecords")]
    data_set = tf.data.TFRecordDataset(filename_list).map(parse_data).shuffle(2048).batch(batch_size).repeat()
    return data_set


def test_tfrecords():
    dataset = read_tfrecords("data/tfrecords", 32)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    sess = tf.Session()
    img_batch = sess.run(next_batch)
    visualized(img_batch, "sample/tmp.jpg")


def center_crop_and_resize(source, destination, size=(128, 128)):
    filename_list = os.listdir(source)
    for filename in filename_list:
        img = Image.open(os.path.join(source, filename))
        if img.size[0] / size[0] < img.size[1] / size[1]:
            img = img.resize((size[0], img.size[1] * size[0] // img.size[0]))
            img = img.crop((0, (img.size[1] - size[1]) // 2, size[0], (img.size[1] - size[1]) // 2 + size[1]))
        else:
            img = img.resize((img.size[0] * size[1] // img.size[1], size[1]))
            img = img.crop(((img.size[0] - size[0]) // 2, 0, (img.size[0] - size[0]) // 2 + size[0], size[1]))
        img.save(os.path.join(destination, filename))


def read_images(image_dir, batch_size):
    filename_list = tf.constant([os.path.join(image_dir, filename) for filename in os.listdir(image_dir)])

    def _parse(filename):
        image_bytes = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_bytes, 3)
        image = tf.image.resize_images(image, (128, 128))
        return tf.divide(tf.cast(image, tf.float32), 255.)

    dataset = tf.data.Dataset.from_tensor_slices((filename_list,)).map(_parse).shuffle(2048).batch(batch_size).repeat()
    return dataset


def test_images():
    dataset = read_images("data/image", 16)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    sess = tf.Session()
    img_batch = sess.run(next_batch)
    visualized(img_batch, "sample/tmp.jpg")


def visualized(image_data, save_path):
    num_v = np.ceil(np.sqrt(len(image_data))).astype(np.int)
    num_h = np.ceil(len(image_data) / num_v).astype(np.int)
    img = np.zeros((num_v * 128, num_h * 128, 3))
    for i, image in enumerate(image_data):
        x = i // num_h * 128
        y = i % num_h * 128
        img[x:x + 128, y:y + 128, :] = image
    Image.fromarray(np.multiply(img, 255).astype(np.uint8)).save(save_path)


if __name__ == '__main__':
    # write_tfrecords(r"E:\workspace\data_set\CelebA\Img\img_align_celeba", 2048, "data")
    # test_tfrecords("data")
    # center_crop_and_resize(r"E:\workspace\data_set\CelebA\Img\img_align_celeba", "data/image", (128, 128))
    test_images()
