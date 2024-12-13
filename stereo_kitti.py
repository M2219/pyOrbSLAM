from __future__ import print_function, division

import argparse
import cv2
import glob
import os
from typing import List
from pyDBoW.TemplatedVocabulary import TemplatedVocabulary

parser = argparse.ArgumentParser(description="pyOrbSLAM")
parser.add_argument("--pathToSequence", default="./00", help="path to sequence")
parser.add_argument("--pathToVocabulary", default="Vocabulary/ORBvoc.txt", help="path to vocabulary")
parser.add_argument("--pathToSettings", default="configs/KITTI00-02.yaml", help="path to settings")

args = parser.parse_args()


class SLAM(object):
    def __init__(self, pathToVocabulary, pathToSettings, mode, visualization) -> None:
        super(SLAM, self).__init__()

        self.mode = mode
        self.visualization = visualization

        # Load ORB Vocabulary



        print("initialization finally done!")



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

    vocabulary = TemplatedVocabulary(k=5, L=3, weighting="TF_IDF", scoring="L1_NORM")
    vocabulary.load_from_text_file("./Vocabulary/ORBvoc.txt")



    return 0



if __name__ == "__main__":


    main(args.pathToVocabulary, args.pathToSettings, args.pathToSequence)




