# Fast Growcut algorithm (C++)

Implemented fast grow cut algorithm based on "An Effective Interactive Medical Image Segmentation Method Using Fast GrowCut" (see [link](https://nac.spl.harvard.edu/files/nac/files/zhu-miccai2014.pdf)), generalized for multi-class n-dimensional data with multiple features.

## Installation

Run setup file

```bash
python setup.py build
sudo python setup.py install
```

## Usage

**Fast Growcut using shortest path**

Implemented in [growcut_fast](./growcut/growcut_fast.cpp), callable by 

```growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, newSeg = True, verbose = True)```

where

**datatensor** : can be n-dimensional data with M features, which means d1 x d2 x d3 ... x dn x M

**seedsdata**: initial n-dimensional seeds data, d1 x d2 x d3 x .. x dn.

**labPre** : converged n-dimensional label results, d1 x d2 x d3 x .. x dn. Can be first initialized as an array with all zero entries, or a previous result.

**distPre** : converged n-dimensional shortest distance results (to the closest class), d1 x d2 x d3 x .. x dn. Can be first initialized as an array with all zero entries, or a previous result.

**newSeg** : set to True for first time GrowCut segmentation. Can be set to False for later implementation of interactive adjustment.

### Test example

Test examples can be found in [test_growcut](./test/test_growcut.py), including inferences for teeth/pelvis annotation refinement.

### Github reference

1. Fibonacci heap package ([link](https://github.com/robinmessage/fibonacci))