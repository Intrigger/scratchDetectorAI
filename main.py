import keras.saving.saved_model_experimental
import numpy
import tensorflow.python.keras.models
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt



print("SCRATCH DETECTOR AI")

to_fit = False



if to_fit:

    model = None

    scratched_ready = os.listdir('scratched_ready/')
    nonscratched_ready = os.listdir('nonscratched_ready/')

    scratched_data = []

    for image in scratched_ready:
        im = Image.open("scratched_ready" + "/" + image)
        im_arr = np.array(im)
        scratched_data.append((im_arr, 'scratched'))

    nonscratched_data = []

    for image in nonscratched_ready:
        im = Image.open("nonscratched_ready" + "/" + image)
        im_arr = np.array(im)
        nonscratched_data.append((im_arr, 'nonscratched'))

    x_train = scratched_data + nonscratched_data
    random.shuffle(x_train)

    y_train = [[0, 1] if _[1] == 'nonscratched' else [1, 0] for _ in x_train]
    x_train = [_[0] for _ in x_train]

    x_train = np.array(x_train)
    y_train = np.array(y_train)


    model = Sequential()

    img_size = (256, 256)

    model.add(Conv2D(128, (3, 3), input_shape=(256, 256, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(64, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(512, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(2))
    model.add(Activation("softmax"))

    # 128 - 64 - 128 - 64 - 64 - 512 - 2 : 66%  (128X128px)
    # 128 - 64 - 128 - 64 - 64 - 512 - 2 : ?%  (256px256px)

    epochs = 15

    optimizer = 'Adam'

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, batch_size=16)

    model.save("ai_scratch_detector.h5")


    #

    scratched_test = os.listdir('scratched_test_ready/')
    nonscratched_test = os.listdir('nonscratched_test_ready/')

    scratched_test_data = []

    for image in scratched_test:
        im = Image.open("scratched_test_ready" + "/" + image)
        im_arr = np.array(im)
        scratched_test_data.append((im_arr, 'scratched'))

    nonscratched_test_data = []

    for image in nonscratched_test:
        im = Image.open("nonscratched_test_ready" + "/" + image)
        im_arr = np.array(im)
        nonscratched_test_data.append((im_arr, 'nonscratched'))

    x_test = scratched_test_data + nonscratched_test_data
    random.shuffle(x_test)

    y_test = [[0, 1] if _[1] == 'nonscratched' else [1, 0] for _ in x_test]
    x_test = [_[0] for _ in x_test]

    x_test = np.array(x_test)
    y_test = np.array(y_test)


    loss, acc = model.evaluate(x_test, y_test)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

else:
    model = tensorflow.keras.models.load_model("ai_scratch_detector.h5")

    scratched_test = os.listdir('scratched_test_ready/')
    nonscratched_test = os.listdir('nonscratched_test_ready/')

    scratched_test_data = []

    for image in scratched_test:
        im = Image.open("scratched_test_ready" + "/" + image)
        im_arr = np.array(im)
        scratched_test_data.append((im_arr, 'scratched'))

    nonscratched_test_data = []

    for image in nonscratched_test:
        im = Image.open("nonscratched_test_ready" + "/" + image)
        im_arr = np.array(im)
        nonscratched_test_data.append((im_arr, 'nonscratched'))

    x_test = scratched_test_data + nonscratched_test_data
    random.shuffle(x_test)

    y_test = [[0, 1] if _[1] == 'nonscratched' else [1, 0] for _ in x_test]
    x_test = [_[0] for _ in x_test]

    x_test = np.array(x_test)
    y_test = np.array(y_test)


    loss, acc = model.evaluate(x_test, y_test)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

    answers = model.predict(x_test)
    answers = ['scratched' if max(_) == _[0] else 'nonscratched' for _ in answers]

    w, h = 6, 6

    fig, axes = plt.subplots(w, h)

    fig.set_size_inches(10, 10)
    fig.subplots_adjust(0.1, 0.1)

    cur = 0

    for y in range(h):
        for x in range(w):
            axes[y, x].get_xaxis().set_visible(False)
            axes[y, x].get_yaxis().set_visible(False)

            axes[y, x].imshow(x_test[y * w + x], cmap='gray')
            axes[y, x].set_title(answers[y * w + x])

    plt.show()