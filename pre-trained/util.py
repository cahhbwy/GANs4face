# coding:utf-8
import os
import tensorflow as tf
from PIL import Image


def write_tfrecords(image_pos_dir, image_neg_dir, batch_size, save_dir="data"):
    filename_pos_list = os.listdir(image_pos_dir)[:24800]
    filename_neg_list = os.listdir(image_neg_dir)[:24800]
    for i in range(0, len(filename_pos_list), batch_size):
        filename = "%06d.tfrecords" % i
        writer = tf.python_io.TFRecordWriter(os.path.join(save_dir, filename))
        for j in range(min(batch_size, len(filename_pos_list) - i)):
            img = Image.open(os.path.join(image_pos_dir, filename_pos_list[i + j]))
            if img.size != (178, 218):
                continue
            item = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[1]))
                    }
                )
            )
            writer.write(item.SerializeToString())
            img = Image.open(os.path.join(image_neg_dir, filename_neg_list[i + j]))
            if img.size != (178, 218):
                continue
            item = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img.tobytes()])),
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[0]))
                    }
                )
            )
            writer.write(item.SerializeToString())
        writer.close()
        print("write %s finished" % filename)


def parse_data(proto):
    parsed = tf.parse_single_example(
        proto, features={
            "image": tf.FixedLenFeature([], tf.string),
            "label": tf.FixedLenFeature([1], tf.int64, 0)
        }
    )
    image = tf.reshape(tf.decode_raw(parsed["image"], tf.uint8), (218, 178, 3))
    label = parsed["label"]
    return {"image": tf.divide(tf.cast(image, tf.float32), 255.), "label": label}


def read_tfrecords(data_dir, batch_size):
    filename_list = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith(".tfrecords")]
    data_set = tf.data.TFRecordDataset(filename_list).map(parse_data).shuffle(2048).batch(batch_size).repeat()
    return data_set


def test_tfrecords(data_dir):
    data_set = read_tfrecords(data_dir, 32)
    iterator = data_set.make_one_shot_iterator()
    next_batch = iterator.get_next()
    sess = tf.Session()
    batch_data = sess.run(next_batch)
    print(batch_data["image"].shape, batch_data["label"].shape)


if __name__ == '__main__':
    # write_tfrecords(r"E:\workspace\data_set\CelebA\Img\img_align_celeba", r"E:\workspace\project\WGAN_face\pre-trained\cleared_pic", 1024, "data")
    test_tfrecords("data")
