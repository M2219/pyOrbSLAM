# pyOrbSLAM2

This repository provides a pythonic implementation of [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2).

# Note:
* The repository has tried to keep the pythonic version as close as possible to the original repository. Therefore, some modules may run slower than the c++ version.
However, modules can be changed to efficiently increase the speed.
* The current version only supports the stereo tracking

# To increase the speed

* Follow [pyslam](https://github.com/luigifreda/pyslam/tree/master) practices which is a complete pythonic SLAM repo.
* Replace the ```compute_stereo_matches()``` method in the Frame class with a GPU-accelerated stereo matching algorithms to enhance frame creation performance.
* Lightweight matchers such as [LightGlue](https://github.com/cvg/LightGlue) can also be used alternatively.
* Optimize the functions implemented in the ```ORBMatcher.py``` module for better efficiency and performance.


# Installation

The code was tested on Ubuntu 22.04 and Python 3.10

```
cd g2o-python
pip install .

cd .. && cd pyORBExtractor
bash build.sh

```
```mkdir Vocabulary``` 
Download and extract [ORBvoc.txt](https://github.com/raulmur/ORB_SLAM2/tree/master/Vocabulary) inside the ```Vocabulary```

### Dependencies

```
pip install opencv-python
pip install ordereddict

```
Install ```pypangolin``` from [Pangolin](https://github.com/stevenlovegrove/Pangolin/tree/master)

## Data Preparation
* [KITTI Odometry](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

## Run

```
python stereo_kitti.py --pathToSequence PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER --pathToSettings ./configs/KITTI_X.yaml

```
Replace KITTI_X.yaml with the related sequence config
