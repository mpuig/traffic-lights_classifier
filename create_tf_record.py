r"""Convert the Traffic Lights dataset to TFRecord for object_detection.

Example usage:
    python create_tf_record.py --data_dir=rosbag_images \
        --output_dir=out
"""

import PIL.Image
import io
import tensorflow as tf
import sys
import os
import glob
import json
import random

# This is needed since the notebook is stored in the udacity folder.
sys.path.append("..")
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw traffic lights dataset.')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/traffic-lights_label_map.pbtxt',
                    'Path to label map proto')
FLAGS = flags.FLAGS

def dict_to_tf_example(data,
                       label_map_dict,
                       image_subdirectory):
    """Convert JSON derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    data: dict holding fields for a single image (obtained by
        running dataset_util.recursive_parse_xml_to_dict)
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
        dataset directory holding the actual image data.

    Returns:
        example: The converted tf.Example.

    Raises:
        ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """


    img_path = os.path.join(image_subdirectory, data['filename'])
    with tf.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG')

    width, height = data['image_w_h'] # Image size

    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    for box in data['objects']:
        x,y,w,h = box['x_y_w_h']
        xmins.append(float(x / width))
        xmaxs.append(float((x + w) / width))
        ymins.append(float(y / height))
        ymaxs.append(float((y + h) / height))
        class_name = box['label']
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map_dict[class_name])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example

def create_tf_record(examples,
                    label_map_dict,
                    image_dir,
                    output_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    counter = 0
    for idx,example in enumerate(examples):
        if idx % 100 == 0:
            print('On image %d of %d' % (idx, len(examples)))
        with open(example) as data_file:
            tf_annotation = dict_to_tf_example(json.load(data_file), label_map_dict, image_dir)
        writer.write(tf_annotation.SerializeToString())
    writer.close()


def main(_):
    data_dir = FLAGS.data_dir
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    print('Reading from Traffic Lights dataset.')

    examples_list = [name for name in glob.glob(os.path.join('rosbag_images', 'annotations', '*.json'))]

    # Test images are not included in the downloaded data set, so we shall perform
    # our own split.
    random.seed(42)
    random.shuffle(examples_list)
    num_examples = len(examples_list)
    num_train = int(0.7 * num_examples)
    train_examples = examples_list[:num_train]
    val_examples = examples_list[num_train:]
    print('%d training and %d validation examples.' %
               (len(train_examples), len(val_examples)))

    train_output_path = os.path.join(FLAGS.output_dir, 'traffic-lights_train.record')
    val_output_path = os.path.join(FLAGS.output_dir, 'traffic-lights_val.record')

    create_tf_record(train_examples, label_map_dict, data_dir, train_output_path)
    create_tf_record(val_examples, label_map_dict, data_dir, val_output_path)

if __name__ == '__main__':
    tf.app.run()