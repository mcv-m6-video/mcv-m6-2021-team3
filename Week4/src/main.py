import os
from os.path import join
from utils.ai_city import AICity
from config.config import Config

def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    aicity = AICity(args)
    

if __name__ == "__main__":
    main(Config().get_args())