import os
import cv2
from tqdm import tqdm
import argparse

semantic_kitti_dir = '/gpfswork/rech/xqt/uyl37fq/data/semantic_kitti/dataset'


def read_list_imgs(root_dir):
    img_files = []
    for root, dirs, files in tqdm(os.walk(root_dir)):
        for file in files:
            if "edge" not in file and "image_2" in root:
                if "jpg" in file or "png" in file:
                    img_files.append({
                        "path": root,
                        'file': file
                    })
    return img_files


def detect_edge(root, file, low_thres, high_thres):
    # Conver image to grayscale
    filename, ext = os.path.splitext(file)
    img = cv2.imread(os.path.join(root, file), 0)
    edges = cv2.Canny(img, high_thres/3, high_thres)
    out_filename = filename + "_edge" + ext 
    out_filename = os.path.join(root, out_filename)
    # print(out_filename)
    cv2.imwrite(out_filename, edges)


def main(lo, hi):
    img_files = read_list_imgs(semantic_kitti_dir)
    for item in tqdm(img_files):
        detect_edge(item['path'], item['file'], lo, hi)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract edges')
    parser.add_argument("--lo", type=float, default=85, help="Low threshold for Canny edge detector")
    parser.add_argument("--hi", type=float, default=255, help="High threshold for Canny edge detector")
    args = parser.parse_args()
    print(args.lo, args.hi)
    main(args.lo, args.hi)
