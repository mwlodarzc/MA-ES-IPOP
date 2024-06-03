import cocopp
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run COCO post-processing (cocopp) for a given directory.')
    parser.add_argument('result_folder', type=str, help='Result folder name')

    args = parser.parse_args()

    cocopp.main(args.result_folder)

if __name__ == '__main__':
    main()
