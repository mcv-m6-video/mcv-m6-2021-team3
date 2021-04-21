import os
from os.path import join
import glob
from datasets.ai_city import AICity
from config.config import Config
from utils.utils import read_video_file

def main(args):

    aicity = AICity(args)
    aicity.track(args.seqs)

    #print(args.data_path)

    '''for path in glob.glob(join(args.data_path,'AICity/train/S04/*/*.avi')):
        read_video_file(path)'''

if __name__ == "__main__":
    main(Config().get_args())

'''
Seq: S03 kalman kmeans
         num_frames      idf1       idp       idr  precision    recall
c010           1357  0.405001  0.479717  0.401401   0.941854  0.925108
c011            766  0.496000  0.562064  0.448276   0.943526  0.761958
c012            153  0.254144  0.197425  0.601307   0.224168  0.836601
c013            642  0.633577  0.711475  0.576361   0.944895  0.774236
c014           1743  0.499736  0.466667  0.569311   0.673770  0.861378
c015             17  1.000000  1.000000  1.000000   1.000000  1.000000
OVERALL        4678  0.474426  0.485643  0.504049   0.760259  0.855199
'''