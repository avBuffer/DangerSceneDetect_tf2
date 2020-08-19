import os

NET_TYPE = 'net' #'resnet50'

CLASSES = ["Safe", "Accident", "Fire", "Robbery"]
CLASS_NUM = len(CLASSES)

RESIZE_WH = 128
TEST_SPLIT = 0.15

INIT_LR = 1e-2
BATCH_SIZE = 2
NUM_EPOCHS = 1000
SAVE_PERIOD = 50

DATA_PATH = "D:/datasets/DangerScenes"
SAFE_PATH = os.path.sep.join([DATA_PATH, "spatial_envelope_256x256_static_8outdoorcategories"])
ACCIDENT_PATH = os.path.sep.join([DATA_PATH, "Robbery_Accident_Fire_Database2", "Accident"])
FIRE_PATH = os.path.sep.join([DATA_PATH, "Robbery_Accident_Fire_Database2", "Fire"])
ROBBERY_PATH = os.path.sep.join([DATA_PATH, "Robbery_Accident_Fire_Database2", "Robbery"])

CKPT_PATH = "ckpts"
if not os.path.exists(CKPT_PATH):
    os.makedirs(CKPT_PATH)

MODEL_PATH = os.path.sep.join([CKPT_PATH, ("danger_detect_%s_%s.model" % (NET_TYPE, str(NUM_EPOCHS)))])
LRFIND_PLOT_PATH = os.path.sep.join([CKPT_PATH, ("lrfind_plot_%s_%s.png" % (NET_TYPE, str(NUM_EPOCHS)))])
TRAINING_PLOT_PATH = os.path.sep.join([CKPT_PATH, ("training_plot_%s_%s.png" % (NET_TYPE, str(NUM_EPOCHS)))])

SAMPLE_SIZE = 50
OUTPUT_IMAGE_PATH = os.path.sep.join(["result", ("%s-%s_%s" % (NET_TYPE, str(NUM_EPOCHS), str(SAMPLE_SIZE)))])
if not os.path.exists(OUTPUT_IMAGE_PATH):
    os.makedirs(OUTPUT_IMAGE_PATH)
