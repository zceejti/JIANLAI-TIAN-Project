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