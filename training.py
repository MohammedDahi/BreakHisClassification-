# USAGE
# python train_model.py

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")


from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adagrad
from keras import optimizers
from keras.utils import np_utils
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from cancernet import CancerNet

from keras.applications import vgg16, inception_v3, resnet50, mobilenet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import keras

from keras.callbacks import ModelCheckpoint


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# initialize our number of epochs, initial learning rate, and batch
# size
NUM_EPOCHS = 50
INIT_LR = 1e-2
BS = 32
size = 224
# determine the total number of image paths in training, validation,
# and testing directories
trainPaths = list(paths.list_images("/home/mohammed/DeepLearning/patches_224/train"))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images("/home/mohammed/DeepLearning/patches_224/test")))
totalTest = len(list(paths.list_images("/home/mohammed/DeepLearning/patches_224/test")))


# account for skew in the labeled data
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = np_utils.to_categorical(trainLabels)

classTotals = trainLabels.sum(axis=0)
classWeight = classTotals.max() / classTotals
print("je suis class weight rak fahem", classWeight)
# initialize the training training data augmentation object
trainAug = ImageDataGenerator(
    rescale=1 / 255.0)
#trainAug = ImageDataGenerator(
#    rescale=1 / 255.0,
#    rotation_range=20,
#    zoom_range=0.05,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    shear_range=0.05,
#    horizontal_flip=True,
#    vertical_flip=True,
#    fill_mode="nearest")

# initialize the validation (and testing) data augmentation object
valAug = ImageDataGenerator(rescale=1 / 255.0)

# initialize the training generator
trainGen = trainAug.flow_from_directory(
    "/home/mohammed/DeepLearning/patches_224/train",
    class_mode="categorical",
    target_size=(size, size),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)

# initialize the validation generator
valGen = valAug.flow_from_directory(
    "/home/mohammed/DeepLearning/patches_224/test",
    class_mode="categorical",
    target_size=(size, size),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# initialize the testing generator
testGen = valAug.flow_from_directory(
    "/home/mohammed/DeepLearning/patches_224/test",
    class_mode="categorical",
    target_size=(size, size),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)

# initialize our CancerNet model and compile it
#model = inception_v3.InceptionV3(weights=None, include_top=False, input_shape= (size,size,3))
#
#x = model.output
#x = keras.layers.GlobalAveragePooling2D()(x)
##x = keras.layers.Dropout(0.9)(x)
#predictions = keras.layers.Dense(2, activation= 'softmax')(x)
#model = keras.models.Model(inputs = model.input, outputs = predictions)
#opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
opt = optimizers.SGD(lr=INIT_LR, decay=INIT_LR/NUM_EPOCHS,momentum=0.9,nesterov=False)

model = CancerNet.build(width=size, height=size, depth=3,
    classes=2)
#opt = Adagrad(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
    metrics=["accuracy"])

print("TOTAL TRAIN", totalTrain)
print("TOTAL TEST", totalTest)

# fit the model
path="model-patcheses224-%d-%d"% (BS,size)
os.mkdir(path)

filepath = path+"/saved-model-{epoch:02d}-{val_acc:.2f}.hdf5"
def myprint(s):
    with open(path+'/modelsummary.txt','w+') as f:
        print(s, file=f)

model.summary(print_fn=myprint)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]
H = model.fit_generator(
    trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    class_weight=classWeight,
    callbacks=callbacks_list,
    epochs=NUM_EPOCHS)

# reset the testing generator and then use our trained model to
# make predictions on the data
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict_generator(testGen,
    steps=(totalTest // BS) + 1)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")

# show a nicely formatted classification report
print(classification_report(testGen.classes, predIdxs,
    target_names=testGen.class_indices.keys()))

# compute the confusion matrix and and use it to derive the raw
# accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
print(total)
print(cm[0,0])
acc = (cm[0,0] + cm[1,1]) / total
sensitivity = cm[0,0] / (cm[0,0] + cm[0,1])
specificity = cm[1,1] / (cm[1,0] + cm[1,1])


# show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print(acc)
print(sensitivity)
print(specificity)


