import os
import numpy as np
import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from core.lr import LearningRateFinder
from core.net import DetectionNet
from core.resnet import ResnetBuilder
from core import config
from core.utils import *

matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--lr-find", type=int, default=0,
                    help="whether or not to find optimal learning rate")
    args = vars(ap.parse_args())

    print("[INFO] loading data...")
    safeData = load_dataset(config.SAFE_PATH)
    accidentData = load_dataset(config.ACCIDENT_PATH)
    fireData = load_dataset(config.FIRE_PATH)
    robberyData = load_dataset(config.ROBBERY_PATH)
    print('safeData.len=', len(safeData), 'accidentData.len=', len(accidentData),
          'fireData.len=', len(fireData), 'robberyData.len=', len(robberyData))

    safeLabels = np.zeros((safeData.shape[0],))
    accidentLabels = np.ones((accidentData.shape[0],))
    fireLabels = np.full((fireData.shape[0],), 2)
    robberyLabels = np.full((robberyData.shape[0],), 3)
    print('safeLabels.len=', len(safeLabels), 'accidentLabels.len=', len(accidentLabels),
          'fireLabels.len=', len(fireLabels), 'robberyLabels.len=', len(robberyLabels))

    data = np.vstack([safeData, accidentData, fireData, robberyData])
    labels = np.hstack([safeLabels, accidentLabels, fireLabels, robberyLabels])
    data /= 255
    print('data.len=', len(data), 'labels.len=', len(labels))

    # perform one-hot encoding on the labels and account for skew in the labeled data
    labels = to_categorical(labels, num_classes=config.CLASS_NUM)
    classTotals = labels.sum(axis=0)
    classWeight = classTotals.max() / classTotals

    # construct the training and testing split
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=config.TEST_SPLIT, random_state=42)
    print('trainX.len=', len(trainX), 'trainY.len=', len(trainY), 'testX.len=', len(testX), 'testY.len=', len(testY))

    # initialize the training data augmentation object
    aug = ImageDataGenerator(rotation_range=30, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,
                             shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

    print("[INFO] compiling model...")
    opt = SGD(lr=config.INIT_LR, momentum=0.9, decay=config.INIT_LR / config.NUM_EPOCHS)
    
    if os.path.exists(config.MODEL_PATH):
        print("[INFO] loading from ", config.MODEL_PATH)
        model = load_model(config.MODEL_PATH)
    else:
        print("[INFO] NET_TYPE=", config.NET_TYPE)
        if not 'resnet' in config.NET_TYPE:
            model = DetectionNet.build(width=config.RESIZE_WH, height=config.RESIZE_WH, depth=3, classes=config.CLASS_NUM)
        else:
            if config.NET_TYPE == 'resnet18':
                model = ResnetBuilder.build_resnet_18((3, config.RESIZE_WH, config.RESIZE_WH), config.CLASS_NUM)
            elif config.NET_TYPE == 'resnet34':
                model = ResnetBuilder.build_resnet_34((3, config.RESIZE_WH, config.RESIZE_WH), config.CLASS_NUM)
            elif config.NET_TYPE == 'resnet50':
                model = ResnetBuilder.build_resnet_50((3, config.RESIZE_WH, config.RESIZE_WH), config.CLASS_NUM)
            elif config.NET_TYPE == 'resnet101':
                model = ResnetBuilder.build_resnet_101((3, config.RESIZE_WH, config.RESIZE_WH), config.CLASS_NUM)
            else:
                model = ResnetBuilder.build_resnet_152((3, config.RESIZE_WH, config.RESIZE_WH), config.CLASS_NUM)

    if config.CLASS_NUM == 2:
        crossentropy_type = "binary_crossentropy"
    else:
        crossentropy_type = "categorical_crossentropy"    
    print("[INFO] crossentropy_type=", crossentropy_type)
    
    model.compile(loss=crossentropy_type, optimizer=opt, metrics=["accuracy"])
    #print(model.summary())

    # check to see if we are attempting to find an optimal learning rate
    # before training for the full number of epochs
    if args["lr_find"] > 0:
        # initialize the learning rate finder and then train with learning rates ranging from 1e-10 to 1e+1
        print("[INFO] finding learning rate...")
        lrf = LearningRateFinder(model)
        lrf.find(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), 1e-10, 1e+1,
                 stepsPerEpoch=np.ceil((trainX.shape[0] / float(config.BATCH_SIZE))),
                 epochs=20, batchSize=config.BATCH_SIZE, classWeight=classWeight)

        lrf.plot_loss()
        plt.savefig(config.LRFIND_PLOT_PATH)

        # gracefully exit the script so we can adjust our learning rates in the config 
        # and then train the network for our full set of epochs
        print("[INFO] learning rate finder complete, examine plot and adjust learning rates before training")
        sys.exit(0)

    print("[INFO] training network...")
    checkpointer = ModelCheckpoint(os.path.join(config.CKPT_PATH, 'danger_scene_{epoch:03d}-{val_loss:.4f}.hdf5'),
                                   monitor='val_loss', verbose=1, save_weights_only=False, period=config.SAVE_PERIOD)

    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=config.BATCH_SIZE), validation_data=(testX, testY),
                            steps_per_epoch=trainX.shape[0] // config.BATCH_SIZE, epochs=config.NUM_EPOCHS,
                            class_weight=classWeight, verbose=1, initial_epoch=0, callbacks=[checkpointer])

    print("[INFO] evaluating network...")
    predictions = model.predict(testX, batch_size=config.BATCH_SIZE)
    classify_report = classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=config.CLASSES)
    print('classify_report=', classify_report)

    print("[INFO] serializing network to '{}'...".format(config.MODEL_PATH))
    model.save(config.MODEL_PATH)

    # construct a plot that plots and saves the training history
    N = np.arange(0, config.NUM_EPOCHS)
    plt.style.use("ggplot")
    plt.figure()

    plt.plot(N, H.history["loss"], label="train_loss")
    plt.plot(N, H.history["val_loss"], label="val_loss")
    plt.plot(N, H.history["accuracy"], label="train_acc")
    plt.plot(N, H.history["val_accuracy"], label="val_acc")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(config.TRAINING_PLOT_PATH)
