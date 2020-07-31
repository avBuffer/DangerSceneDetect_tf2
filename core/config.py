import os

NET_TYPE = 'resnet'

SAFE_PATH = "data/spatial_envelope_256x256_static_8outdoorcategories"
ACCIDENT_PATH = os.path.sep.join(["data", "Robbery_Accident_Fire_Database2", "Accident"])
FIRE_PATH = os.path.sep.join(["data", "Robbery_Accident_Fire_Database2", "Fire"])
ROBBERY_PATH = os.path.sep.join(["data", "Robbery_Accident_Fire_Database2", "Robbery"])

CLASSES = ["Safe", "Accident", "Fire", "Rebbery"]
CLASS_NUM = len(CLASSES)

RESIZE_WH = 128
TRAIN_SPLIT = 0.75
TEST_SPLIT = 0.25

INIT_LR = 1e-2
BATCH_SIZE = 64
NUM_EPOCHS = 10000

MODEL_PATH = os.path.sep.join(["ckpts", ("danger_detect_%s_%s.model" % (NET_TYPE, str(NUM_EPOCHS)))])
LRFIND_PLOT_PATH = os.path.sep.join(["ckpts", "lrfind_plot.png"])
TRAINING_PLOT_PATH = os.path.sep.join(["ckpts", "training_plot.png"])

OUTPUT_IMAGE_PATH = os.path.sep.join(["result", "examples"])
SAMPLE_SIZE = 50
