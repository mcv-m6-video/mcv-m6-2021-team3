Download TF models from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Execute with
 
```
python models/research/object_detection/model_main_tf2.py --pipeline_config_path="./models/research/object_detection/checkpoints/SSDResNet152V1FPN1024x1024(RetinaNet152)/pipeline.config" \
    --model_dir="./models/research/object_detection/checkpoints/SSDResNet152V1FPN1024x1024(RetinaNet152)/checkpoints" \
    --alsologtostderr \
    --num_train_steps=20 \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps=1
```