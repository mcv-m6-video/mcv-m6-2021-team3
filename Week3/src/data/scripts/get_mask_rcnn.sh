
url=https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x/139653917/model_final_2d9806.pkl
echo 'Downloading' $url$f && wget $url$f && mv *.pkl data/weights/mask_rcnn.pkl