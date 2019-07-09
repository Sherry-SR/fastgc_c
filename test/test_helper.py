import pytest
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import itertools
from os.path import join

def test_checkpath(SAVE_RESULTS = False):

    imgpath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_c/results/pelvis/pelvis_d4_g/fast_growcut_path.nii.gz"
    savepath = "/home/SENSETIME/shenrui/Dropbox/SenseTime/fastgc_c/results/pelvis/pelvis_d4_g"

    img = nib.load(imgpath)
    imgdata = (img.get_fdata()).astype(np.int)

    path = np.zeros((imgdata.shape)[0:3])
    idx = (93, 110, 135)
    pathlist = [idx]
    path[idx] = 1
    while np.sum(imgdata[idx[0], idx[1], idx[2], :]) > 0:
        idx = tuple(imgdata[idx[0], idx[1], idx[2], :])
        pathlist.insert(0, idx)
        path[idx] = 1

    print(pathlist)
    if SAVE_RESULTS:
        nib.save(nib.Nifti1Image(path.astype(float), img.affine), join(savepath, "test_checkpath.nii.gz"))
        print("results saved!")

def util_bbox(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(reversed(range(N)), N-1):
        nonezero = np.any(img, axis = ax)
        out.append(np.where(nonezero)[0][[0, -1]])
    return out