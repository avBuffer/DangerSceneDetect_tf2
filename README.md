# DangerSceneDetect_tf2
A Dangerous Scene Detection included fire, accident and robbery etc. by using keras in Tensorflow2.

## Download Dangerous dataset
Download Robbery_Accident_Fire_Database2 and spatial_envelope_256x256_static_8outdoorcategories.

```
Extract dataset into data floder, which should have the following basic structure.
```

### data path

1) data/Robbery_Accident_Fire_Database2

1.1) data/Robbery_Accident_Fire_Database2/Accident

1.1) data/Robbery_Accident_Fire_Database2/Fire

1.1) data/Robbery_Accident_Fire_Database2/Robbery

2) data/spatial_envelope_256x256_static_8outdoorcategories

## Train method

```bashrc
$ python train.py or python train.py --lr-find 1
```

## Predict method

```bashrc
$ python predict.py
```

## CMD shell

```bashrc
$ chmod 777 cmd.sh
$ ./cmd.sh
```