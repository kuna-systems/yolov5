import argparse
import os
from tqdm import tqdm
import shutil


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_test_txt', help='path to the txt file with test dataset')
    parser.add_argument('--destination_path', help='path to destination folder')
    opt = parser.parse_args()
    
    return opt


def main(path_test_txt, destination_path):
    os.makedirs(destination_path, exist_ok=True)
    
    with open(path_test_txt) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            path = line.strip()
            shutil.copy(path, destination_path)
            if os.path.exists(os.path.splitext(path)[0] + '.txt'):
                shutil.copy(os.path.splitext(path)[0] + '.txt', destination_path)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt.path_test_txt, opt.destination_path)
