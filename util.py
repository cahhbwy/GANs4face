# coding:utf-8
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


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


def read_anno_identity(identity_path: str) -> pd.DataFrame:
    with open(identity_path) as file:
        lines = file.readlines()
        lines = [line.split() for line in lines]
        data_identity = pd.DataFrame(lines, columns=["filename", "identity"]).set_index("filename")
        data_identity = data_identity.astype(np.int64)
        return data_identity


def read_anno_attr(attr_path: str) -> pd.DataFrame:
    with open(attr_path) as file:
        count = int(file.readline())
        header = file.readline().split()
        lines = file.readlines()
        assert len(lines) == count
        lines = [line.split() for line in lines]
        data_attr = pd.DataFrame(lines, columns=["filename"] + header).set_index("filename")
        data_attr = data_attr.astype(np.int64)
        return data_attr


def read_anno_bbox(bbox_path: str) -> pd.DataFrame:
    with open(bbox_path) as file:
        count = int(file.readline())
        header = file.readline().split()[1:]
        lines = file.readlines()
        assert len(lines) == count
        lines = [line.split() for line in lines]
        data_bbox = pd.DataFrame(lines, columns=["filename"] + header).set_index("filename")
        data_bbox = data_bbox.astype(np.int64)
        return data_bbox


def read_anno_landmarks(landmarks_path: str) -> pd.DataFrame:
    with open(landmarks_path) as file:
        count = int(file.readline())
        header = file.readline().split()
        lines = file.readlines()
        assert len(lines) == count
        lines = [line.split() for line in lines]
        data_landmarks = pd.DataFrame(lines, columns=["filename"] + header).set_index("filename")
        data_landmarks = data_landmarks.astype(np.int64)
        return data_landmarks


def read_anno_partition(partition_path: str) -> pd.DataFrame:
    with open(partition_path) as file:
        lines = file.readlines()
        lines = [line.split() for line in lines]
        data_partition = pd.DataFrame(lines, columns=["filename", "partition"]).set_index("filename")
        data_partition = data_partition.astype(np.int64)
        return data_partition


def get_columns(align=True):
    if align:
        return ['identity',
                '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
                'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y',
                'partition']
    else:
        return ['identity',
                '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair',
                'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
                'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
                'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young',
                'x_1', 'y_1', 'width', 'height',
                'lefteye_x', 'lefteye_y', 'righteye_x', 'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 'leftmouth_y', 'rightmouth_x', 'rightmouth_y',
                'partition']


def read_anno(identity_path: str, attr_path: str, bbox_path: str, landmarks_path: str, partition_path: str) -> pd.DataFrame:
    data_identity = read_anno_identity(identity_path)
    data_attr = read_anno_attr(attr_path)
    data_bbox = read_anno_bbox(bbox_path)
    data_landmarks = read_anno_landmarks(landmarks_path)
    data_partition = read_anno_partition(partition_path)
    data = pd.concat([data_identity, data_attr, data_bbox, data_landmarks, data_partition], axis=1)
    return data


def read_anno_align(identity_path: str, attr_path: str, landmarks_align_path: str, partition_path: str) -> pd.DataFrame:
    data_identity = read_anno_identity(identity_path)
    data_attr = read_anno_attr(attr_path)
    data_landmarks = read_anno_landmarks(landmarks_align_path)
    data_partition = read_anno_partition(partition_path)
    data = pd.concat([data_identity, data_attr, data_landmarks, data_partition], axis=1)
    return data


def process(images_path, filename, anno):
    img_path = tf.strings.join([images_path, filename], separator="/")
    img_raw = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_raw)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return {"image": img, "filename": filename, "anno": anno}


def load_images(data: pd.DataFrame, images_path: str):
    data_list = ([images_path] * len(data), data.index.to_list(), data.values)
    dataset = tf.data.Dataset.from_tensor_slices(data_list)
    dataset = dataset.map(process)
    return dataset


def test_load_images(align=True):
    celeba_root = "/Volumes/Seagate Backup Plus Drive/E/workspace/dataset/CelebA"
    anno_list = get_columns(align)
    if align:
        data = read_anno_align(
            identity_path=f"{celeba_root}/Anno/identity_CelebA.txt",
            attr_path=f"{celeba_root}/Anno/list_attr_celeba.txt",
            landmarks_align_path=f"{celeba_root}/Anno/list_landmarks_align_celeba.txt",
            partition_path=f"{celeba_root}/Eval/list_eval_partition.txt"
        )
        dataset = load_images(data, f"{celeba_root}/Img/img_align_celeba").shuffle(100).repeat().batch(16)
        batch = next(iter(dataset))
        images = tf.image.convert_image_dtype(batch["image"], tf.uint8)
        visualize(images.numpy(), height=218, width=178, channel=3).show()
        attribute = pd.DataFrame(batch["anno"].numpy(), index=batch["filename"].numpy().astype(str), columns=anno_list)
        print(attribute)
    else:
        data = read_anno(
            identity_path=f"{celeba_root}/Anno/identity_CelebA.txt",
            attr_path=f"{celeba_root}/Anno/list_attr_celeba.txt",
            bbox_path=f"{celeba_root}/Anno/list_bbox_celeba.txt",
            landmarks_path=f"{celeba_root}/Anno/list_landmarks_celeba.txt",
            partition_path=f"{celeba_root}/Eval/list_eval_partition.txt"
        )
        dataset = load_images(data, f"{celeba_root}/Img/img_align_celeba")
        batch = next(iter(dataset))
        Image.fromarray(tf.image.convert_image_dtype(batch["image"], tf.uint8).numpy()).show()
        print(f"filename : {batch['filename'].numpy().decode()}")
        for anno, value in zip(anno_list, batch["anno"].numpy()):
            print(f"{anno} : {value}")


def make_tfrecords(data: pd.DataFrame, images_path: str, output_path, block_size: int = 8192) -> None:
    for idx in range(0, len(data), block_size):
        block = data.iloc[idx:idx + block_size]
        writer = tf.io.TFRecordWriter(f"{output_path}/{idx:08d}.tfrecords")
        for i, filename in enumerate(block.index):
            anno = block.loc[filename].to_dict()
            img_bytes = tf.io.read_file(tf.strings.join([images_path, filename], separator="/"))
            img = tf.image.decode_jpeg(img_bytes)
            shape = img.shape
            feature = {
                'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename.encode()])),
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[0]])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[1]])),
                'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[shape[2]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes.numpy()])),
                **{key: tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) for key, value in anno.items()}
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
            print(f"\r{i + idx}/{len(data)}", end="")
        writer.close()
    print("\nFinished.")


def load_tfrecords(tfrecords_filename_list, align=True):
    anno_list = get_columns(align)
    raw_dataset = tf.data.TFRecordDataset(tfrecords_filename_list)
    feature_description = {
        'filename': tf.io.FixedLenFeature([], tf.string),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channel': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        **{key: tf.io.FixedLenFeature([], tf.int64) for key in anno_list}
    }
    dataset = raw_dataset.map(lambda example_proto: tf.io.parse_single_example(example_proto, feature_description))
    return dataset


def test_tfrecords(align=True):
    if align:
        tfrecords_filename_list = [os.path.join("data/tfrecords_align", filename) for filename in os.listdir("data/tfrecords_align") if filename.endswith(".tfrecords")]
    else:
        tfrecords_filename_list = [os.path.join("data/tfrecords", filename) for filename in os.listdir("data/tfrecords") if filename.endswith(".tfrecords")]
    anno_list = get_columns(align)
    if align:
        dataset = load_tfrecords(tfrecords_filename_list)
        dataset = dataset.shuffle(1024).repeat().batch(16)
        batch = next(iter(dataset))
        image = tf.map_fn(tf.image.decode_jpeg, batch["image_raw"], tf.uint8)
        visualize(image.numpy(), height=218, width=178, channel=3).save("tmp.jpg")
        print(pd.DataFrame([batch[anno].numpy() for anno in anno_list], index=anno_list, columns=batch['filename'].numpy().astype(str)).T)
    else:
        dataset = load_tfrecords(tfrecords_filename_list)
        batch = next(iter(dataset))
        image = tf.image.decode_jpeg(batch["image_raw"])
        Image.fromarray(image.numpy()).save("tmp.jpg")
        print(f"filename : {batch['filename'].numpy().decode()}")
        for anno in anno_list:
            print(f"{anno} : {batch[anno].numpy()}")


if __name__ == '__main__':
    # test_load_images(align=False)

    celeba_root = "/Volumes/Seagate Backup Plus Drive/E/workspace/dataset/CelebA"

    # data = read_anno(
    #     identity_path=f"{celeba_root}/Anno/identity_CelebA.txt",
    #     attr_path=f"{celeba_root}/Anno/list_attr_celeba.txt",
    #     bbox_path=f"{celeba_root}/Anno/list_bbox_celeba.txt",
    #     landmarks_path=f"{celeba_root}/Anno/list_landmarks_celeba.txt",
    #     partition_path=f"{celeba_root}/Eval/list_eval_partition.txt"
    # )
    # make_tfrecords(data, f"{celeba_root}/Img/img_celeba", "data/tfrecords", 8192)
    # test_tfrecords(align=False)

    # data = read_anno_align(
    #     identity_path=f"{celeba_root}/Anno/identity_CelebA.txt",
    #     attr_path=f"{celeba_root}/Anno/list_attr_celeba.txt",
    #     landmarks_align_path=f"{celeba_root}/Anno/list_landmarks_align_celeba.txt",
    #     partition_path=f"{celeba_root}/Eval/list_eval_partition.txt"
    # )
    # make_tfrecords(data, f"{celeba_root}/Img/img_align_celeba", "data/tfrecords_align", 8192)
    test_tfrecords(align=True)
