""" Tests for the growcut module """

import pytest
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
import nibabel as nib
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage.filters import median_filter, gaussian_filter

from os import makedirs
from os.path import dirname, join, abspath, exists
sys.path.append(abspath(join(dirname(__file__),'..')))

import growcut_fast
from test_helper import util_bbox

def test_growcut_fast_list(filepath = None, savepath = None):
    CT_max = 2000
    CT_min = 0
    with open(filepath) as f:
        for line in f:
            name, imgpath, labelpath = line.split()
            label = nib.load(labelpath)
            labeldata_all = np.copy(label.get_fdata())

            bbox = util_bbox(binary_dilation(labeldata_all, iterations=4))

            img = nib.load(imgpath)
            imgdata = np.copy(img.get_fdata()[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]])
            labeldata = np.copy(labeldata_all[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]])
            
            seedsdata = np.zeros(labeldata.shape)

            imgdata[imgdata > CT_max] = CT_max
            imgdata[imgdata < CT_min] = CT_min
            imgdata = (imgdata - CT_min) / (CT_max - CT_min)

            gx, gy, gz = np.gradient(gaussian_filter(imgdata, sigma = 1))
            imggradient = np.sqrt(np.power(gx,2) + np.power(gy,2) + np.power(gz,2))
            imggradient = (imggradient - imggradient.min()) / (imggradient.max() - imggradient.min())
    
            datatensor = np.stack((imgdata, imggradient), axis = -1)

            nlabels = np.unique(labeldata)
            for i in nlabels:
                mask = labeldata == i
                for j in range(labeldata.shape[2]):
                    mask[:,:,j] = binary_erosion(mask[:,:,j], iterations=4)
                seedsdata = seedsdata + mask * (i+1)

            labPre = np.zeros(labeldata.shape)
            distPre = np.zeros(labeldata.shape)
            #distPath = np.zeros(labeldata.shape+(len(labeldata.shape),))

            start = time.time()
            #labPre = growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, distPath, True, True)
            labPre = growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, True, False)
            end = time.time()
            print("time used:", end - start, "seconds")

            labPre = labPre - 1
            labeldata_all[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = np.copy(labPre)
            
            subjpath = join(savepath, name)
            if not exists(subjpath):
                makedirs(subjpath)

            nib.save(nib.Nifti1Image((img.get_fdata()).astype(float), img.affine), join(subjpath, "original_img_whole.nii.gz"))
            nib.save(nib.Nifti1Image((img.get_fdata()[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]).astype(float), img.affine), join(subjpath, "original_img.nii.gz"))
            nib.save(nib.Nifti1Image(imgdata.astype(float), img.affine), join(subjpath, "processed_img.nii.gz"))
            nib.save(nib.Nifti1Image(imggradient.astype(float), img.affine), join(subjpath, "img_gradient.nii.gz"))
            nib.save(nib.Nifti1Image((label.get_fdata()).astype(float), img.affine), join(subjpath, "original_label_whole.nii.gz"))
            nib.save(nib.Nifti1Image(labeldata.astype(float), img.affine), join(subjpath, "original_label.nii.gz"))
            nib.save(nib.Nifti1Image(seedsdata.astype(float), img.affine), join(subjpath, "seedsdata.nii.gz"))
            nib.save(nib.Nifti1Image(labeldata_all.astype(float), img.affine), join(subjpath, "fast_growcut_results_all.nii.gz"))
            nib.save(nib.Nifti1Image(labPre.astype(float), img.affine), join(subjpath, "fast_growcut_results.nii.gz"))
            nib.save(nib.Nifti1Image(distPre.astype(float), img.affine), join(subjpath, "fast_growcut_dist.nii.gz"))
            #nib.save(nib.Nifti1Image(distPath.astype(float), img.affine), join(subjpath, "fast_growcut_path.nii.gz"))
            print(name + " results saved!")

def test_growcut_fast_teeth():

    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/teeth0001_label_tri.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_c/results/teeth"

    img = nib.load(imgpath)
    imgdata = img.get_fdata()[120:580, 10:350, 110:340]
    label = nib.load(labelpath)
    labeldata = label.get_fdata()[120:580, 10:350, 110:340]
    seedsdata = np.zeros(labeldata.shape)

    CT_max = 2000
    CT_min = 0
    imgdata[imgdata > CT_max] = CT_max
    imgdata[imgdata < CT_min] = CT_min
    imgdata = (imgdata - CT_min) / (CT_max - CT_min)

    gx, gy, gz = np.gradient(gaussian_filter(imgdata, sigma = 1))
    imggradient = np.sqrt(np.power(gx,2) + np.power(gy,2) + np.power(gz,2))
    datatensor = np.stack((imgdata, imggradient), axis = -1)

    nlabels = np.unique(labeldata)
    for i in nlabels:
        mask = labeldata == i
        for j in range(labeldata.shape[2]):
            mask[:,:,j] = binary_erosion(mask[:,:,j], iterations=4)
        seedsdata = seedsdata + mask * (i+1)
    labPre = np.zeros(labeldata.shape)
    distPre = np.zeros(labeldata.shape)
    #distPath = np.zeros(labeldata.shape+(len(labeldata.shape),))

    start = time.time()
    #labPre = growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, distPath, True, True)
    labPre = growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, True, True)
    end = time.time()
    print("time used:", end - start, "seconds")
    
    labPre = labPre - 1

    nib.save(nib.Nifti1Image(imgdata.astype(float), img.affine), join(savepath, "original_img.nii.gz"))
    nib.save(nib.Nifti1Image(imggradient.astype(float), img.affine), join(savepath, "original_gradient.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata.astype(float), img.affine), join(savepath, "original_label.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata.astype(float), img.affine), join(savepath, "seedsdata.nii.gz"))
    nib.save(nib.Nifti1Image(labPre.astype(float), img.affine), join(savepath, "fast_growcut_results.nii.gz"))
    nib.save(nib.Nifti1Image(distPre.astype(float), img.affine), join(savepath, "fast_growcut_dist.nii.gz"))
    #nib.save(nib.Nifti1Image(distPath.astype(float), img.affine), join(savepath, "fast_growcut_path.nii.gz"))
    print("results saved!")

def test_growcut_fast_pelvis():

    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/pelvis03_ct.nii.gz"
    labelpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/data/pelvis03-ct_label.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_c/results/pelvis"

    img = nib.load(imgpath)
    imgdata = img.get_fdata()[80:460, 210:420, :]
    label = nib.load(labelpath)
    labeldata = label.get_fdata()[80:460, 210:420, :]
    seedsdata = np.zeros(labeldata.shape)

    CT_max = 2000
    CT_min = 0
    imgdata[imgdata > CT_max] = CT_max
    imgdata[imgdata < CT_min] = CT_min
    imgdata = (imgdata - CT_min) / (CT_max - CT_min)

    gx, gy, gz = np.gradient(gaussian_filter(imgdata, sigma = 1))
    imggradient = np.sqrt(np.power(gx,2) + np.power(gy,2) + np.power(gz,2))
    imggradient = (imggradient - imggradient.min()) / (imggradient.max() - imggradient.min())
    
    datatensor = np.stack((imgdata, imggradient), axis = -1)

    nlabels = np.unique(labeldata)
    for i in nlabels:
        mask = labeldata == i
        for j in range(labeldata.shape[2]):
            mask[:,:,j] = binary_erosion(mask[:,:,j], iterations=4)
        seedsdata = seedsdata + mask * (i+1)

    labPre = np.zeros(labeldata.shape)
    distPre = np.zeros(labeldata.shape)
    #distPath = np.zeros(labeldata.shape+(len(labeldata.shape),))

    start = time.time()
    #labPre = growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, distPath, True, True)
    labPre = growcut_fast.growcut_cpu(datatensor, seedsdata, labPre, distPre, True, True)
    end = time.time()
    print("time used:", end - start, "seconds")

    labPre = labPre - 1

    nib.save(nib.Nifti1Image(imgdata.astype(float), img.affine), join(savepath, "original_img.nii.gz"))
    nib.save(nib.Nifti1Image(imggradient.astype(float), img.affine), join(savepath, "original_gradient.nii.gz"))
    nib.save(nib.Nifti1Image(labeldata.astype(float), img.affine), join(savepath, "original_label.nii.gz"))
    nib.save(nib.Nifti1Image(seedsdata.astype(float), img.affine), join(savepath, "seedsdata.nii.gz"))
    nib.save(nib.Nifti1Image(labPre.astype(float), img.affine), join(savepath, "fast_growcut_results.nii.gz"))
    nib.save(nib.Nifti1Image(distPre.astype(float), img.affine), join(savepath, "fast_growcut_dist.nii.gz"))
    #nib.save(nib.Nifti1Image(distPath.astype(float), img.affine), join(savepath, "fast_growcut_path.nii.gz"))
    print("results saved!")

test_growcut_fast_list(filepath='/home/SENSETIME/shenrui/data/pelvis_resampled/dataset_all.txt', savepath='/home/SENSETIME/shenrui/data/pelvis_results')
#test_growcut_fast_pelvis()
#test_growcut_fast_teeth()