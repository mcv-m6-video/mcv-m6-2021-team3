
url=https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x/139173657/model_final_68b088.pkl
echo 'Downloading' $url$f && wget $url$f && mv *.pkl data/weights/faster_rcnn.pkl