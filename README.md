# traffic-lights_classifier


## Sources of information
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_pets.md

Running locally: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/running_locally.md
 1. Install: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/installation.md
 2. Prepare inputs: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/preparing_inputs.md
 3. Training Pipeline: https://github.com/tensorflow/models/blob/master/object_detection/g3doc/configuring_jobs.md


## Steps to install
1. Clone the github.com/tensorflow/models
2. Clone the github.com/mpuig/traffic-lights_classifier inside tensorflow/models
3. Setup environment:

```
# From tensorflow/models/ directory
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
```

4. Install Google Cloud SDK to run the learning process.

```
# My local setup
export PATH="$PATH:~/Downloads/google-cloud-sdk/bin/"

# Name of your Google Cloud Bucket to be used for the process
export YOUR_GCS_BUCKET=udacity-traffic-lights
```

5. Create the tf records

```
python create_tf_record.py \
  --data_dir=rosbag_images \
  --output_dir=out \
  --label_map_path=data/traffic-lights-label_map.pbtxt
```


6. Export tf records to the google cloud platform bucket:
gsutil cp out/traffic-lights_train.record gs://${YOUR_GCS_BUCKET}/data/train.record
gsutil cp out/traffic-lights_val.record gs://${YOUR_GCS_BUCKET}/data/val.record
gsutil cp data/traffic-lights-label_map.pbtxt gs://${YOUR_GCS_BUCKET}/data/label_map.pbtxt


7. Downloading a COCO-pretrained Model for Transfer Learning

Training a state of the art object detector from scratch can take days, even when using multiple GPUs! In order to speed up training, we'll take an object detector trained on a different dataset (COCO), and reuse some of it's parameters to initialize our new model.

Download the COCO-pretrained Faster R-CNN with Resnet-101 model. Unzip the contents of the folder and copy the model.ckpt* files into your GCS Bucket.

Choose a pretrained Model:
  - Faster RCNN
  - SSD Mobilenet
  - etc

### Faster RCNN
```
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz

tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz

gsutil cp faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/

sed -i "s|PATH_TO_BE_CONFIGURED|"gs://${YOUR_GCS_BUCKET}"/data|g" \
    config/faster_rcnn_resnet101_coco_11_06_2017.config

gsutil cp config/faster_rcnn_resnet101_tl.config gs://${YOUR_GCS_BUCKET}/data/faster_rcnn_resnet101_tl.config

export YOUR_MODEL=faster_rcnn_resnet101_tl
```

### SSD Mobilenet
```
wget http://storage.googleapis.com/download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_11_06_2017.tar.gz

tar -xvf ssd_mobilenet_v1_coco_11_06_2017.tar.gz

gsutil cp ssd_mobilenet_v1_coco_11_06_2017/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/

sed -i "s|PATH_TO_BE_CONFIGURED|"gs://${YOUR_GCS_BUCKET}"/data|g" \
    config/ssd_mobilenet_v1_coco.config

gsutil cp config/ssd_mobilenet_v1_coco.config gs://${YOUR_GCS_BUCKET}/data/ssd_mobilenet_v1_coco.config

export YOUR_MODEL=faster_rcnn_resnet101_tl
```



8. Train
```
# From tensorflow/models/

gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%s` \
    --job-dir=gs://${YOUR_GCS_BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region europe-west1 \
    --config config/cloud.yml \
    -- \
    --train_dir=gs://${YOUR_GCS_BUCKET}/train \
    --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/${YOUR_MODEL}.config


gcloud ml-engine jobs submit training `whoami`_object_detection_eval_`date +%s` \
    --job-dir=gs://${YOUR_GCS_BUCKET}/train \
    --packages dist/object_detection-0.1.tar.gz,dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region europe-west1 \
    --scale-tier BASIC_GPU \
    -- \
    --checkpoint_dir=gs://${YOUR_GCS_BUCKET}/train \
    --eval_dir=gs://${YOUR_GCS_BUCKET}/eval \
    --train_dir=gs://${YOUR_GCS_BUCKET}/train \
    --pipeline_config_path=gs://${YOUR_GCS_BUCKET}/data/${YOUR_MODEL}.config
```

9. Download the trained model
```
# From tensorflow/models
export CHECKPOINT_NUMBER=0
gsutil cp gs://${YOUR_GCS_BUCKET}/train/model.ckpt-${CHECKPOINT_NUMBER}.* .
```

10. Export the inference graph
```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path config/ssd_mobilenet_v1_coco.config \
    --trained_checkpoint_prefix model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory output_inference_graph.pb
```

11. launch jupyter notebook to test the model
```
jupyter notebook object_detection.ipynb
``