from __future__ import print_function, division

import argparse
import cv2
import glob
import os
from typing import List


parser = argparse.ArgumentParser(description="pyOrbSLAM")
parser.add_argument("--pathToSequence", default="./00", help="path to sequence")
parser.add_argument("--pathToVocabulary", default="Vocabulary/ORBvoc.txt", help="path to vocabulary")
parser.add_argument("--pathToSettings", default="configs/KITTI00-02.yaml", help="path to settings")

args = parser.parse_args()

def read_all_lines(filename: str) -> List[str]:
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines

def LoadImages(pathToSeq):

    imageLeft = sorted(glob.glob(os.path.join(args.pathToSequence, "image_2/*")))
    imageRight = sorted(glob.glob(os.path.join(args.pathToSequence, "image_3/*")))
    vTimestamps = read_all_lines(os.path.join(args.pathToSequence, "times.txt"))

    return imageLeft, imageRight, vTimestamps


def main(pathToVocabulary, pathToSettings, pathToSequence):

    leftImages, rightImages, timeStamps = LoadImages(pathToSequence)
    nImages = len(leftImages)

    # Create SLAM system. It initializes all system threads and gets ready to process frames.
    #ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true)
    #load
    return 0



if __name__ == "__main__":


    main(args.pathToVocabulary, args.pathToSettings, args.pathToSequence)




