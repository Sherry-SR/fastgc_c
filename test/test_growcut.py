""" Tests for the growcut module """

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.filters import median_filter

from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__),'..')))

import growcut_fast

def test_growcut_fast_teeth():

    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001_label_tri.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_c/results"

    img = nib.load(imgpath)
    imgdata = np.squeeze(img.get_fdata()[120:580, 10:350, 110:340])
    label = nib.load(labelpath)
    labeldata = np.squeeze(label.get_fdata()[120:580, 10:350, 110:340])
    seedsdata = np.zeros(labeldata.shape)

    nlabels = np.unique(labeldata)
    for i in nlabels:
        mask = labeldata == i
        for j in range(labeldata.shape[2]):
            mask[:,:,j] = binary_erosion(mask[:,:,j])
            mask[:,:,j] = binary_erosion(mask[:,:,j])
            mask[:,:,j] = binary_erosion(mask[:,:,j])
            mask[:,:,j] = binary_erosion(mask[:,:,j])
        seedsdata = seedsdata + mask * (i+1)
    
    labPre = np.zeros(labeldata.shape)
    distPre = np.zeros(labeldata.shape)

    start = time.time()
    labPre = growcut_fast.growcut_cpu(imgdata, seedsdata, labPre, distPre, True, True)
    end = time.time()
    print("time used:", end - start, "seconds")

    nib.save(nib.Nifti1Image(imgdata.astype(float), img.affine), join(savepath, "original_img_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata.astype(float), img.affine), join(savepath, "original_label_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata.astype(float), img.affine), join(savepath, "seedsdata_teeth.nii.gz"))
    nib.save(nib.Nifti1Image(labPre.astype(float), img.affine), join(savepath, "fast_growcut_results_teeth.nii.gz"))
    print("results saved!")

def test_growcut_fast_pelvis():

    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/pelvis03_ct.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/pelvis03-ct_label.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_c/results"

    img = nib.load(imgpath)
    imgdata = np.squeeze(img.get_fdata()[80:460, 210:420, :])
    label = nib.load(labelpath)
    labeldata = np.squeeze(label.get_fdata()[80:460, 210:420, :])
    seedsdata = np.zeros(labeldata.shape)

    imgdata = median_filter(imgdata, size=5)

    nlabels = np.unique(labeldata)
    for i in nlabels:
        mask = labeldata == i
        for j in range(labeldata.shape[2]):
            mask[:,:,j] = binary_erosion(mask[:,:,j])
            mask[:,:,j] = binary_erosion(mask[:,:,j])
            mask[:,:,j] = binary_erosion(mask[:,:,j])
            mask[:,:,j] = binary_erosion(mask[:,:,j])
        seedsdata = seedsdata + mask * (i+1)
    
    labPre = np.zeros(labeldata.shape)
    distPre = np.zeros(labeldata.shape)

    start = time.time()
    labPre = growcut_fast.growcut_cpu(imgdata, seedsdata, labPre, distPre, True, True)
    end = time.time()
    print("time used:", end - start, "seconds")

    nib.save(nib.Nifti1Image(imgdata.astype(float), img.affine), join(savepath, "original_img_pelvis.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata.astype(float), img.affine), join(savepath, "original_label_pelvis.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata.astype(float), img.affine), join(savepath, "seedsdata_pelvis.nii.gz"))
    nib.save(nib.Nifti1Image(labPre.astype(float), img.affine), join(savepath, "fast_growcut_results_pelvis.nii.gz"))
    print("results saved!")

#test_growcut_fast_pelvis()
test_growcut_fast_teeth() 