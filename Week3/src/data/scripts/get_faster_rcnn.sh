
url=https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
echo 'Downloading' $url$f && wget $url$f && mv *.pkl data/weights/faster_rcnn.pkl

#url=https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
#echo 'Downloading' $url$f && wget $url$f && mv *.yaml data/model_config/faster_rcnn.yaml