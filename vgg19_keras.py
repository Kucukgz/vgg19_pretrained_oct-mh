# %% Libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import skimage
from skimage.io import imread
import pandas as pd
from keras.applications.vgg19 import VGG19
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Flatten


# images
traning_path = 'data/training'
test_path = 'data/testing'

report = 'Report.xlsx'

# learning
LEARNING_RATE = 0.0001
BATCH_SIZE = 1
RESIZE = 224

# get vision_score. That means y_train
all_data = pd.read_excel(report)
all_data = pd.DataFrame(all_data)
vision_score = all_data['Pre-op Vision']


# %% load data
def loading_data(path):
    dataset_3d = list(os.listdir(path))
    dataset_3d = sorted(dataset_3d, key=lambda v: v.upper())

    # get how many plane it has
    image_n_planes = np.zeros((len(dataset_3d), 1), dtype=np.int)
    for i in range(0, len(dataset_3d)):
        im = Image.open(os.path.join(path + '/', dataset_3d[i]))
        image_n_planes[i] = im.n_frames
    dataset_n_planes = image_n_planes.sum()

    # get dataset as a name of plane
    dataset = []
    for i in range(0, len(dataset_3d)):
        for j in range(0, image_n_planes[i].item()):
            # full name with total row number
            dataset.append(str(dataset_3d[i])+','+str(j))
    return dataset, dataset_n_planes


# %%
def vis_load(first):
    first = first.split(".tif", 1)
    vis_score_index = (all_data['Original Filename'] == first[0]).argmax()
    vis_score = all_data['Pre-op Vision'][vis_score_index]
    return vis_score


def read_images(path, num_img, dataset):  # num_images = resim sayimiz
    array = np.zeros((num_img, 224, 224, 3))
    array2 = np.zeros([num_img])
    for i in range(num_img):
        image_name = dataset[i]
        # get 3d_name(i) and 2d_plane_nr(j)
        first = image_name.split(',')[0]
        second = int(image_name.split(',')[1])
        # vis_score
        array2[i] = vis_load(first)
        # load image
        image_path = os.path.join(path, first)
        img = skimage.img_as_ubyte(skimage.io.imread(image_path))
        img = np.moveaxis(img, 0, -1)
        img = img[:, :, second]
        img = cv2.resize(img, (224, 224))
        image = np.expand_dims(img, axis=2)
        image = np.repeat(image, 3, axis=2)
        array[i] = image
        i += 1
    return array, array2


# train data preprocessing
training_dataset, number_img = loading_data(traning_path)
x_train, y_train = read_images(traning_path, number_img, training_dataset)

# test data preprocessing
testing_dataset, number_img = loading_data(test_path)
x_test, y_test = read_images(test_path, number_img, testing_dataset)

numberOfClass = 100

y_train = to_categorical(y_train, numberOfClass)
y_test = to_categorical(y_test, numberOfClass)

# %% visualize
# plt.imshow(x_train[12,:].reshape(224, 224, 3), cmap='gray')

plt.figure()
plt.imshow(x_train[12, :].reshape(224, 224, 3), cmap='gray')
plt.axis("off")
plt.show()

# %% vgg19

vgg = VGG19(include_top=False, weights="imagenet", input_shape=(224, 224, 3))

print(vgg.summary())

vgg_layer_list = vgg.layers
print(vgg_layer_list)

model = Sequential()
for layer in vgg_layer_list:
    model.add(layer)

print(model.summary())

for layer in model.layers:
    layer.trainable = False

# fully con layers
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(numberOfClass, activation="softmax"))

print(model.summary())


model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

hist = model.fit(x_train, y_train, validation_split=0.2,
                 epochs=100, batch_size=BATCH_SIZE)

# %%  model save
model.save_weights("vgg19_model_saving.h5")

# %%
plt.plot(hist.history["loss"], label="train loss")
plt.plot(hist.history["val_loss"], label="val loss")
plt.legend()
plt.show()
plt.savefig('loss.png')

plt.figure()
plt.plot(hist.history["accuracy"], label="train acc")
plt.plot(hist.history["val_accuracy"], label="val acc")
plt.legend()
plt.show()
plt.savefig('accuracy.png')

# %% load
import json, codecs
with codecs.open("transfer_learning_vgg19_cfar10.json","r", encoding="utf-8") as f:
    n = json.loads(f.read())

plt.plot(n["acc"], label="train acc")
plt.plot(n["val_acc"], label="val acc")
plt.legend()
plt.show()

# %% save
with open('transfer_learning_vgg19_cfar10.json', 'w') as f:
    json.dump(hist.history, f)
