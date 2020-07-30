import os
import sys
import cv2
import numpy as np
from . import config
from imutils import paths


def get_filePath_fileName_fileExt(filename):
    filepath, tmpfilename = os.path.split(filename)
    shotname, extension = os.path.splitext(tmpfilename)
    return filepath, shotname, extension


def load_dataset(datasetPath):
    imagePaths = list(paths.list_images(datasetPath))
    data = []
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (config.RESIZE_WH, config.RESIZE_WH))
        data.append(image)
    return np.array(data, dtype="float32")
    