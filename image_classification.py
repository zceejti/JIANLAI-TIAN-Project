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


############################# Step４. Train different models for different tasks. #############################

def MLP(input_shape, output=1, activation='sigmoid'):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(512, activation='relu',input_shape = input_shape))
    model.add(keras.layers.Dense(output,))
    model.add(keras.layers.Activation(activation))
    return model
class AccHistory(keras.callbacks.Callback):
    def __init__(self):
        pass
    def on_train_begin(self, logs={}):
        self.train_acc = []
        self.train_loss = []
        self.val_acc = []
        self.val_loss = []
        self.epoch = 1
    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
history = AccHistory()
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=2)

for item in ['smiling', 'young', 'eyeglasses', 'human', 'hair_color']:
    print 'Traning model for %s predition...' % item
    optimizer = keras.optimizers.SGD(lr = 0.008, momentum=0.9, decay=0.001, nesterov = True)
    if item=='hair_color':
        _X_train = X[np.array(y['hair_color'])!=-1, :]
        _y_train = np.array(y[item])
        _y_train = _y_train[_y_train!=-1]
        _X_val = X_validation[np.array(y_validation['hair_color'])!=-1, :]
        _y_val = np.array(y_validation[item])
        _y_val = _y_val[_y_val!=-1]
        _X_test = X_test[np.array(y_test['hair_color'])!=-1, :]
        _y_test = np.array(y_test[item])
        _y_test = _y_test[_y_test!=-1]
        model = MLP((X.shape[1],), output=6, activation='softmax')
        model.compile(optimizer = optimizer,loss = 'categorical_crossentropy',metrics=['accuracy'])
        model.fit(_X_train, to_categorical(_y_train), batch_size=256, epochs=500, callbacks = [early_stopping, history],
              validation_data=[_X_val, to_categorical(_y_val)], verbose=0)
        print 'Accuracy:',model.evaluate(_X_test, to_categorical(_y_test), verbose=0)[1]
    else:
        model = MLP((X.shape[1],))
        model.compile(optimizer = optimizer,loss = 'binary_crossentropy',metrics=['accuracy'])
        model.fit(X, np.array(y[item]).reshape(-1,1), batch_size=256, epochs=500, callbacks = [early_stopping, history],
                  validation_data=[X_validation, np.array(y_validation[item]).reshape(-1,1)], verbose=0)
        print 'Accuracy:',model.evaluate(X_test, np.array(y_test[item]).reshape(-1,1), verbose=0)[1]
    
    plt.figure(figsize=(18,7))
    plt.subplot(121)
    plt.plot(history.train_acc, '--*', label='Train Acc')
    plt.plot(history.val_acc, '-s',label='Val Acc')
    plt.legend(fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.title('Acc of %s prediction model.' % item, fontsize=20 )

    plt.subplot(122)
    plt.plot(history.train_loss, '--*', label='Train Loss')
    plt.plot(history.val_loss, '-s',label='Val Loss')
    plt.legend(fontsize=18)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss of %s prediction model.' % item , fontsize=20)
    plt.savefig('./result/history_%s_model.png' % item, bbox_inches ='tight')
    
    print '#'*80
## SVM 
## In this Experiment, The performance of SVM and MLP is similar. But MLP runs faster
##
# def svm_gridsearch(X_train, y_train, X_val, y_val):
#     acc = []
#     for C in [1.0, 4.0, 8.0, 16.0]:
#         for gamma in [0.0004, 0.001, 0.002]:
#             clf = SVC(C = C, gamma= gamma)
#             clf.fit(X_train, y_train)
#             acc.append([[C, gamma], clf.score(X_val, y_val)])
#     acc.sort(key=lambda x:x[1], reverse = True)
#     return acc[0][0]

# for item in ['smiling', 'young', 'eyeglasses', 'human', 'hair_color']:
#     print 'Training SVM  for %s predition...' % item
#     if item=='hair_color':
#         _X_train = X[np.array(y['hair_color'])!=-1, :]
#         _y_train = np.array(y[item])
#         _y_train = _y_train[_y_train!=-1]
#         _X_val = X_validation[np.array(y_validation['hair_color'])!=-1, :]
#         _y_val = np.array(y_validation[item])
#         _y_val = _y_val[_y_val!=-1]
#         _X_test = X_test[np.array(y_test['hair_color'])!=-1, :]
#         _y_test = np.array(y_test[item])
#         _y_test = _y_test[_y_test!=-1]
#         C, gamma = svm_gridsearch(_X_train, _y_train, _X_val, _y_val)
#         clf = SVC(C = C, gamma = gamma)
#         clf.fit(_X_train, _y_train)
#         print 'Accuracy:', clf.score(_X_test, _y_test)
#     else:
#         C, gamma = svm_gridsearch(X, y[item], X_validation, y_validation[item])
#         clf = SVC(C = C, gamma = gamma)
#         clf.fit(X, y[item])
#         print 'Accuracy:', clf.score(X_test, y_test[item])
