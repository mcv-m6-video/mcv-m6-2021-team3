import os
from os.path import join
from utils.ai_city import AICity
from utils.utils import write_json_file
from config.config import Config


def main(args):

    os.makedirs(join(args.output_path, str(args.task)), exist_ok=True)

    aicity = AICity(args)
    aicity.train_val_split()
        

if __name__ == "__main__":
    main(Config().get_args())
