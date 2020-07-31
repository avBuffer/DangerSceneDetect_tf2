import os
import sys
import cv2
import numpy as np
import imutils
import random
from tensorflow.keras.models import load_model
from core import config
from core import utils
from imutils import paths


if __name__ == '__main__':
    # load the trained model from disk
    print("[INFO] loading model...")
    model = load_model(config.MODEL_PATH)

    print("[INFO] predicting...")
    safePaths = list(paths.list_images(config.SAFE_PATH))
    accidentPaths = list(paths.list_images(config.ACCIDENT_PATH))
    firePaths = list(paths.list_images(config.FIRE_PATH))
    robberyPaths = list(paths.list_images(config.ROBBERY_PATH))

    imagePaths = safePaths + accidentPaths + firePaths + robberyPaths
    random.shuffle(imagePaths)
    imagePaths = imagePaths[:config.SAMPLE_SIZE]

    for idx, imagePath in enumerate(imagePaths):
    	filename = os.path.split(imagePath)
        image = cv2.imread(imagePath)
        output = image.copy()

        image = cv2.resize(image, (config.RESIZE_WH, config.RESIZE_WH))
        image = image.astype("float32") / 255.0
        
        preds = model.predict(np.expand_dims(image, axis=0))[0]
        label_id = np.argmax(preds)
        label = config.CLASSES[label_id]

        text = label if label == "Safe" else ("WARNING! %s!") % label
        text_color = (0, 255, 0) if label == "Safe" else (0, 0, 255)
        output = imutils.resize(output, width=500)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 5)

        out_file = os.path.sep.join([config.OUTPUT_IMAGE_PATH, filename])
        cv2.imwrite(out_file, output)
        print('idx=', idx, 'imagePath=', imagePath, 'out_file=', out_file)
