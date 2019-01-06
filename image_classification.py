# coding: utf-8
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
import shutil, os
import keras
from sklearn.svm import SVC
from keras.utils.np_utils import to_categorical


image_root = './images/'
files = os.listdir(image_root)
attributes = pd.read_csv('./attribute_list.csv')

if not os.path.exists('./result'):
    os.mkdir('./result')

############################# Step1. Compute the sharpness of picture #############################
# Compute the sharpness of a picture using the method proposed by Pech-Pacheco in 2000.
# 《Diatom autofocusing in brightfield microscopy: a comparative study》
def compute_sharpness(f):
    img = cv2.imread(os.path.join(image_root, '%d.png'%f ))
    return cv2.Laplacian(img, cv2.CV_64F).var()
attributes['sharpness'] = attributes['file_name'].apply(compute_sharpness)

## Exclude the blurred pictures
## In our method, a picture with less than 30 sharpness value can be considered blurred.
attributes['head'] = ~((attributes['hair_color']==-1) & (attributes['eyeglasses']==-1) & (attributes['smiling']==-1) & (attributes['young']==-1) & (attributes['human']==-1))
attributes['usable'] = (attributes['sharpness']>30) & attributes['head']

## Pictures with -1 for all attributes can be excluded.
plt.figure(figsize=(10,7))
plt.subplot(121)
demo_image = np.random.choice((attributes[(attributes['head']==True) & (attributes.sharpness<10)].file_name.values))
plt.imshow(cv2.imread(os.path.join(image_root, '%d.png' % demo_image ))[:, :, ::-1])
plt.title('Demo of blurred picture,\n Sharpness = %.2f' %  attributes[attributes.file_name==demo_image].sharpness.iloc[0], fontsize=20)
plt.subplot(122)
demo_image = np.random.choice((attributes[(attributes['head']==True) & (attributes.sharpness>300)].file_name.values))
plt.imshow(cv2.imread(os.path.join(image_root, '%d.png' % demo_image ))[:, :, ::-1])
plt.title('Demo of clear picture,\n Sharpness = %.2f' % attributes[attributes.file_name==demo_image].sharpness.iloc[0], fontsize=20)
plt.savefig('./result/demo_sharpness.png', bbox_inches ='tight')
############################# Step2. Split images as train, validation, test #############################
usable_images = attributes[attributes.usable==True][['file_name', 'hair_color', 'eyeglasses', 'smiling', 'young', 'human']]
usable_images.reset_index(drop=True, inplace=True)

file_names = usable_images.file_name.values.tolist()
np.random.shuffle(file_names)

total = len(file_names)
train_images = file_names[:int(total*0.8)]
validation_images = file_names[int(total*0.8): int(total*0.9)]
test_images = file_names[int(total*0.9):]
print 'Train images: %d' % len(train_images)
print 'Validation images: %d' % len(validation_images)
print 'Test images: %d' % len(test_images)

############################# Step3. Extract CNN Features for transfer learning. #############################
## images meaf shfit
def my_preprocessing(x):
    x = x[:, :, ::-1]
#     x[:, :,  0] -= 103.939
#     x[:, :,  1] -= 116.779
#     x[:, :,  2] -= 123.68
    return x/255.0

def CNN(model, file_list, suffix):
    attri = ['smiling', 'young', 'eyeglasses', 'human', 'hair_color']
    X = []
    y = dict()
    for item in attri:
        y[item] = []
    for name in file_list:
        img = cv2.imread(os.path.join(image_root, '%d.png' % name)).astype('float64')
        img = my_preprocessing(img)
        feature_map = model.predict(img.reshape(1,256,256,3)).ravel()
        X.append(feature_map)
        for item in attri:
            temp = attributes[attributes.file_name==int(name)][item].iloc[0]
            if item == 'hair_color':
                y[item].append(temp)
            else:
                y[item].append(1 if temp==1 else 0)
    for item in attri:
        y[item] = np.array(y[item])
    return np.array(X), y

model_feature = keras.applications.InceptionV3(include_top=False, input_shape=(256, 256, 3))
model_feature = keras.models.Model(model_feature.input, keras.layers.GlobalAveragePooling2D()(model_feature.output))

X, y = CNN(model_feature, train_images, 'train')
X_validation, y_validation = CNN(model_feature, validation_images, 'val')
X_test, y_test = CNN(model_feature, test_images, 'test')

