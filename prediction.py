import nibabel as nib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import mlflow
import tensorflow as tf
import unetmodel

VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include

INPUT_SHAPE = 64
IMG_SIZE = 64

# DEFINE seg-areas
SEGMENT_CLASSES = {
    0: 'NOT tumor',
    1: 'NECROTIC/CORE',  # or NON-ENHANCING tumor CORE
    2: 'EDEMA',
    3: 'ENHANCING'  # original 4 -> converted into 3 later
}





def predictByPath(model, path):
    X = np.empty((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 1))

    vol_path = os.path.join(path)
    flair = nib.load(vol_path).get_fdata()


    for j in range(VOLUME_SLICES):
        X[j, :, :, 0] = cv2.resize(flair[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))

    return model.predict(X / np.max(X), verbose=1)


def showPredicts(model, path, start_slice=60):
    # path = f"BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_{case}"
    origImage = nib.load(path).get_fdata()
    p = predictByPath(model, path)

    core = p[:, :, :, 1]
    edema = p[:, :, :, 2]
    enhancing = p[:, :, :, 3]

    plt.figure(figsize=(18, 50))
    f, axarr = plt.subplots(1, 5, figsize=(18, 50))

    for i in range(5):  # for each image, add brain background
        axarr[i].imshow(cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray",
                        interpolation='none')

    axarr[0].imshow(cv2.resize(origImage[:, :, start_slice + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axarr[0].title.set_text('Original image flair')
    axarr[1].imshow(p[start_slice, :, :, 1:4], cmap="Reds", interpolation='none', alpha=0.3)
    axarr[1].title.set_text('all classes')
    axarr[2].imshow(edema[start_slice, :, :], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[2].title.set_text(f'{SEGMENT_CLASSES[1]} predicted')
    axarr[3].imshow(core[start_slice, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[3].title.set_text(f'{SEGMENT_CLASSES[2]} predicted')
    axarr[4].imshow(enhancing[start_slice, :, ], cmap="OrRd", interpolation='none', alpha=0.3)
    axarr[4].title.set_text(f'{SEGMENT_CLASSES[3]} predicted')
    # plt.show()
    plt.savefig("static/pred_plot.png", bbox_inches='tight', pad_inches=0)
