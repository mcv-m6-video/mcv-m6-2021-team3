
url=https://dl.fbaipublicfiles.com/detectron2/COCO-Detection/retinanet_R_101_FPN_3x/190397697/model_final_971ab9.pkl
echo 'Downloading' $url$f && wget $url$f && mv *.pkl data/weights/retinanet.pkl