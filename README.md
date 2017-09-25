# traffic-lights_classifier


## Sources of information
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md

Running locally: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md
 1. Install: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md
 2. Prepare inputs: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/preparing_inputs.md
 3. Training Pipeline: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md


## Steps to install

### 1. Clone the Tensorflow/models repo

```
git clone https://github.com/tensorflow/models tensorflow-models
```

### 2. Clone the Traffic Lights classifier repc

```
git clone https://github.com/mpuig/traffic-lights_classifier
```

### 3. Setup environment

The Tensorflow Object Detection API uses Protobufs to configure model and training parameters. Before the framework can be used, the Protobuf libraries must be compiled. This should be done by running the following command from the tensorflow/models/research/ directory:

```
# From tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

```

When running locally, the tensorflow/models/research/ and slim directories should be appended to PYTHONPATH. This can be done by running the following from tensorflow/models/research/:

```
# From tensorflow/models/research/
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

### 4. Create the tf records

```
# from traffic-lights_classifier/ directory

python create_tf_record.py \
  --data_dir=rosbag_images \
  --output_dir=data \
  --label_map_path=data/tl_label_map.pbtxt
```

### 5. Downloading Faster RCNN COCO-pretrained Model for Transfer Learning

Training a state of the art object detector from scratch can take days, even when using multiple GPUs! In order to speed up training, we'll take an object detector trained on a different dataset (COCO), and reuse some of it's parameters to initialize our new model.

Download the COCO-pretrained Faster R-CNN with Resnet-101 model. Unzip the contents of the folder and copy the model.ckpt* files into your GCS Bucket.

```
# from traffic-lights_classifier/ directory

wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz

tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz

export YOUR_MODEL=faster_rcnn_resnet101_tl
```

### 6. Train locally

A local training job can be run with the following command:

```
mkdir train
python ../tensorflow-models/research/object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=data/${YOUR_MODEL}.local.config \
    --train_dir=./train


mkdir eval
python ../tensorflow-models/research/object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=data/${YOUR_MODEL}.local.config \
    --checkpoint_dir=./train \
    --eval_dir=./eval

tensorboard --logdir=${PATH_TO_MODEL_DIRECTORY}

```


### 7. Export the inference graph

After your model has been trained, you should export it to a Tensorflow graph proto:

```
export CHECKPOINT_NUMBER=0

# If using AWS, download from server
scp -i "udacity.pem" ubuntu@ec2-34-240-221-165.eu-west-1.compute.amazonaws.com:/home/ubuntu/traffic-lights_classifier/model${CHECKPOINT_NUMBER}.tar.gz .

tar zxvf model${CHECKPOINT_NUMBER}.tar.gz

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path data/${YOUR_MODEL}.local.config \
    --trained_checkpoint_prefix ./train/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory output_inference_graph-${CHECKPOINT_NUMBER}.pb
```
Afterwards, you should see a graph named output_inference_graph.pb.



### 8. launch jupyter notebook to test the model
```
jupyter notebook object_detection.ipynb
``

