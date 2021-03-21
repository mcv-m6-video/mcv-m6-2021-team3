url=https://github.com/ultralytics/yolov3/releases/download/v9.1/yolov3.pt
echo 'Downloading' $url$f && wget $url$f && mv *.pt data/weights/yolov3.pt