import cocopp
import argparse

import os

def main():
    parser = argparse.ArgumentParser(description='Run COCO post-processing (cocopp) for a given directory.')
    parser.add_argument('result_folder', type=str, help='Result folder name')

    args = parser.parse_args()

    dirs = [os.path.join(args.result_folder, dir) for dir in os.listdir(args.result_folder) if dir.split('.')[-1] != 'pickle']
    cocopp.main(dirs)

if __name__ == '__main__':
    main()
