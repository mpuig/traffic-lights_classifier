# traffic-lights_classifier (steps for using Google Cloud Platform)

### 3. Install Google Cloud SDK to run the learning process.

### 5. Setup environment:

```
export PATH="$PATH:~/Downloads/google-cloud-sdk/bin/"

# From tensorflow/models/ directory
export PYTHONPATH=$PYTHONPATH:`pwd`/research:`pwd`/slim

# Name of your Google Cloud Bucket to be used for the process
export YOUR_GCS_BUCKET=udacity-traffic-lights
```

### 5.1. Export tf records to the google cloud platform bucket:
```
# from traffic-lights_classifier/ directory

gsutil cp out/traffic-lights_train.record gs://${YOUR_GCS_BUCKET}/data/tl_train.record
gsutil cp out/traffic-lights_val.record gs://${YOUR_GCS_BUCKET}/data/tl_val.record
gsutil cp data/traffic-lights-label_map.pbtxt gs://${YOUR_GCS_BUCKET}/data/tl_label_map.pbtxt
gsutil cp faster_rcnn_resnet101_coco_11_06_2017/model.ckpt.* gs://${YOUR_GCS_BUCKET}/data/
gsutil cp config/${YOUR_MODEL}.config gs://${YOUR_GCS_BUCKET}/data/${YOUR_MODEL}.config
```

### 6. Train on Google Cloud
```
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

### 6.1. Download the trained model
```
export CHECKPOINT_NUMBER=0
gsutil cp gs://${YOUR_GCS_BUCKET}/train/model.ckpt-${CHECKPOINT_NUMBER}.* ./train/
```

### 7. Export the inference graph

After your model has been trained, you should export it to a Tensorflow graph proto:

```
python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path config/${YOUR_MODEL}.config \
    --trained_checkpoint_prefix ./train/model.ckpt-${CHECKPOINT_NUMBER} \
    --output_directory output_inference_graph.pb
```
Afterwards, you should see a graph named output_inference_graph.pb.