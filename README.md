# pyOrbSLAM2

This repository provides a Pythonic implementation of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

# Note:
* The repository has written to keep the pythonic version as close as possible to the original repository. Therefore, some modules may run slower than the c++ version.
However, modules can be changes to efficiently increase the speed.


# To increase the speed

* Follow [pyslam](https://github.com/luigifreda/pyslam/tree/master) practices which is a complete pythonic SLAM module.
* Replace the ```compute_stereo_matches()``` method in the Frame class with a GPU-accelerated stereo matching algorithm to enhance frame creation performance.
* Light weight matchers such as [LightGlue](https://github.com/cvg/LightGlue) can also be used alternatively.
* Optimize the functions implemented in the ```ORBMatcher.py``` module for better efficiency and performance.


# Installation

The code was tested on Ubuntu 22.04 and Python 3.10

```
cd g2o-python
pip install .

cd .. && cd pyORBExtractor
bash build.sh

Download and extract [ORBvoc.txt](https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary) inside the Vocabulary
```
### Dependencies

```
pip install opencv-python
pip install ordereddict

Install pypangolin from [Pangolin](https://github.com/stevenlovegrove/Pangolin/tree/master)
```

## Data Preparation
* [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

## Run

```
python stereo_kitti.py --pathToSequence PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER --pathToSettings ./configs/KITTI_X.yaml

```
