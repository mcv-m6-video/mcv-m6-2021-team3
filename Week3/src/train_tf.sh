#!/bin/sh

# MOBILENET 640
python ./models/research/object_detection/model_main_tf2.py --pipeline_config_path="/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/mobilenet640/pipeline.config" --model_dir="/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/mobilenet640/trained" --alsologtostderr


#RESNET 640
python ./models/research/object_detection/model_main_tf2.py --pipeline_config_path="/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/resnet640/pipeline.config" --model_dir="/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/resnet640/trained" --alsologtostderr


#EFFICIENTDET 640
python ./models/research/object_detection/model_main_tf2.py --pipeline_config_path="/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/efficientdetd1/pipeline.config" --model_dir="/home/josep/shared_dir/Week3bis/src/models/research/object_detection/checkpoints/efficientdetd1/trained" --alsologtostderr
