# coding:utf-8
import os
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw


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


def write_tfrecords(image_dir, batch_size, save_dir="data"):
    filename_list = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]
    for i in range(0, len(filename_list), batch_size):
        filename = "%06d.tfrecords" % i
        writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, filename))
        for j in range(min(batch_size, len(filename_list) - i)):
            img = Image.open(os.path.join(image_dir, filename_list[i + j]))
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


def read_tfrecords(data_dir, shape=(128, 128, 3), batch_size=32):
    def parse_data(proto):
        features = tf.parse_single_example(
            proto, features={
                "image": tf.FixedLenFeature([], tf.string),
            }
        )
        img = tf.reshape(tf.decode_raw(features["image"], tf.uint8), shape)
        img = tf.divide(tf.cast(img, tf.float32), 255.)
        return img

    filename_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".tfrecords")]
    data_set = tf.data.TFRecordDataset(filename_list).map(parse_data).repeat().shuffle(2048).batch(batch_size)
    return data_set


def test_tfrecords(path="data/tfrecords", shape=(128, 128, 3), batch_size=32):
    dataset = read_tfrecords(path, shape, batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()
    sess = tf.Session()
    img_batch = sess.run(tf.cast(tf.multiply(next_batch, 255.), tf.uint8))
    img = visualize(img_batch, height=shape[0], width=shape[1], channel=shape[2])
    img.save("sample/test.jpg")


def read_images(image_dir, shape, batch_size):
    filename_list = tf.constant([os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith(".jpg")])

    def _parse(filename):
        image_bytes = tf.read_file(filename)
        image = tf.image.decode_jpeg(image_bytes, shape[2])
        image = tf.image.resize_images(image, shape[0:2])
        return image

    dataset = tf.data.Dataset.from_tensor_slices((filename_list,)).map(_parse).repeat().shuffle(2048).batch(batch_size)
    return dataset


def test_images(shape=(128, 128, 3), batch_size=32):
    dataset = read_images("data/image", shape, batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_batch = tf.image.hsv_to_rgb(iterator.get_next())
    sess = tf.Session()
    img_batch = sess.run(next_batch)
    img = visualize(img_batch, height=shape[0], width=shape[1], channel=shape[3])
    img.save("sample/test.jpg")


def visualize(images, labels=None, label2text=str, height=20, width=20, channel=1, pad=1):
    """
    将多张图片连标签一起放置在一张图片上
    :param images: 多张图片数据，np.ndarry(dtype=np.uint8)
    :param labels: 图片对应标签，np.ndarry(dtype=np.int64)
    :param label2text: 标签转字符串函数
    :param height: 单张图片高度，int
    :param width: 单张图片宽度，int
    :param channel: 图片通道数，int
    :param pad: 图片边距，int
    :return: PIL.Image
    """
    size = len(images)
    num_h = int(np.ceil(np.sqrt(size)))
    num_v = int(np.ceil(size / num_h).astype(np.int))
    image = np.zeros((num_v * height + (num_v + 1) * pad, num_h * width + (num_h + 1) * pad, channel), np.uint8)
    for idx, img in enumerate(images):
        i = idx // num_h
        j = idx % num_v
        image[pad + i * (height + pad):pad + i * (height + pad) + height, pad + j * (width + pad):pad + j * (width + pad) + width, :] = img
    if channel == 1:
        img = Image.fromarray(image.reshape(image.shape[:-1]))
    else:
        img = Image.fromarray(image)
    if labels is not None:
        assert len(images) == len(labels)
        draw = ImageDraw.Draw(img)
        for idx, label in enumerate(labels):
            i = idx // num_h
            j = idx % num_v
            draw.text((j * (width + pad) + pad, i * (height + pad) + pad), label2text(label), fill=0)
    return img


if __name__ == '__main__':
    center_crop_and_resize("/media/monk/e/workspace/data_set/CelebA/Img/img_align_celeba", "data/image", (64, 64))
    write_tfrecords("data/image", 4096, "data/tfrecords")
    test_tfrecords("data/tfrecords", (64, 64, 3), 16)
