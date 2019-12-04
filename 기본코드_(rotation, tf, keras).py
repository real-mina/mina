# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:25:20 2019

@author: stu8
"""

# 고쳤음 

################### 로데티션 ##########################################
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# 이미지 조정(로테이션)
data_datagen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        fill_mode = 'nearest')

import os
###### 파일 경로 확인이 제일 중요
a = len(next(os.walk("C:/project/test2/191204_3차정제/psychoda"))[2])

for i in range(1,a):
    img = load_img('C:/project/test2/191204_3차정제/psychoda/psychoda ({}).jpg'.format(i))
    x = img_to_array(img)
    x = x.reshape((1,)+x.shape)
    i = 0
    for batch in data_datagen.flow(x,save_to_dir='c:/project/test2/191204_3차정제/test',
                                   save_prefix = '1', save_format='jpg') : # 저장할때 파일 이름, 형식 지정
        i += 1
        if i > 11:
            break




################### tensorflow ###############################################

import glob
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#############''' 바꿔줘야 하는 부분  : 파일 경로 '''
caltech_dir = "C:/project/test2/191204_3차정제/"
####################################################
categories = ["ant","psychoda","mouse","rice","cockroach"]
nb_class = len(categories)
image_w = 64
image_h = 64
pixels = image_w * image_h * 3
X = []
Y = []
for idx, cat in enumerate(categories):
	label = [0 for i in range(nb_class)]
	label[idx] = 1
	image_dir = caltech_dir+"/"+cat
	files = glob.glob(image_dir+"2_10/*.jpg")
	# print(files)
	for i, f in enumerate(files):
		img = Image.open(f)
		img = img.convert("RGB")
		img = img.resize((image_w,image_h))
		data = np.asarray(img)
		X.append(data)
		Y.append(label)
#		if i % 10 == 0:
#			print(i,"\n",data)


X = np.array(X)
X.shape
Y = np.array(Y)
Y.shape
X_train, X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
image_data = (X_train, X_test, Y_train,Y_test )
np.save("c:/data/project_data_191204_test4.npy",image_data)        # npz로 하지 않는 이유?
       # npz로 하지 않는 이유?

X_train, X_test, Y_train,Y_test = np.load("c:/data/project_data_191204_test4.npy",allow_pickle=True)
#X_train.shape
#X_test.shape
#Y_train.shape
#Y_test.shape

x = tf.placeholder(tf.float32,[None,64,64,3])
y = tf.placeholder(tf.float32,[None,5])

w1 = tf.Variable(tf.random_normal([3,3,3,32],stddev=0.01))
L1 = tf.nn.conv2d(x,w1,strides=[1,1,1,1],padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME') 


w2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2 = tf.nn.conv2d(L1,w2,strides=[1,1,1,1],padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w3 = tf.Variable(tf.random_normal([3,3,64,64],stddev=0.01))
L3 = tf.nn.conv2d(L2,w3,strides=[1,1,1,1],padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


w4 = tf.Variable(tf.random_normal([8*8*64,256],stddev=0.01))
L4 = tf.reshape(L3,[-1,8*8*64])
L4 = tf.nn.relu(tf.matmul(L4,w4))

w5 = tf.Variable(tf.random_normal([256,5],stddev=0.01))
model = tf.matmul(L4,w5)
hypothesis = tf.nn.softmax(model)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=y))
# optimizer = tf.train.GradentDescentOptimizer(learning_rate=0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


sess = tf.Session()
sess.run(tf.global_variables_initializer())
batch_size = 100    # 학습 수

for epoch in range(1,11):
	avg_cost = 0
	for i in range(int(np.ceil(len(X_train)/batch_size))):
		x_ = X_train[batch_size*i : batch_size*(i+1)]
		y_ = Y_train[batch_size*i : batch_size*(i+1)]
		_,cost_val = sess.run([optimizer,cost],feed_dict={x:x_,y:y_})
		avg_cost += cost_val
		
	print('Epoch:','%04d'%(epoch),'cost: ','{:.9f}'.format(avg_cost/len(X_train)))


is_correct = tf.equal(tf.argmax(hypothesis,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
print("정확도 : ",sess.run(accuracy,feed_dict={x:X_test,y:Y_test}))

plt.imshow(X_test[100],cmap='Greys')
data = X_test[100].reshape([1,64,64,3])
print("Prediction : ",sess.run(tf.argmax(hypothesis,1),feed_dict={x:data}))
categories[3]

##### 확인

img = Image.open("C:/project/spider/spider (18).jpg")
plt.imshow(img)
plt.imshow(img.resize([64,64]))
img = img.convert("1")
data = img.resize([64,64])
data = np.asarray(data)
#data.shape

data = data.reshape(1,64,64,3)
#data.shape
print("Prediction : ",sess.run(tf.argmax(hypothesis, 1), feed_dict={x:data}))
categories[0]



########################### keras ########################################

#수총 집 데이터 
X_train, X_test, y_train,y_test = np.load("c:/data/project_data_191204_test4.npy",allow_pickle=True)

#38377
len(X_train)#30701
len(X_test)#7676
len(y_train)
len(y_test)

import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.layers import Conv2D,MaxPooling2D

# scale 작업 해주기(0~1 사이 값으로 바꿔주기)
X_train = X_train.astype('float32')/225
X_test = X_test.astype('float32')/225

y_train.shape   # (30701, 11)
y_test.shape    # (7676, 11)

# 1층
model = Sequential()
model.add(Conv2D(32,(3,3),padding='same',input_shape=(64,64,3)))
model.summary()
model.add(Activation('relu'))

# 2층
model.add(Conv2D(32,(3,3)))
model.summary()
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.summary()
model.add(Dropout(0.25))    # 과적합을 피하기 위해 25%는 0으로 채우기

# 3층
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))    # 과적합을 피하기 위해 25%는 0으로 채우기
model.summary()

# 펼치기
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(5))        # y_train값의 개수와 맞춰준다.
model.add(Activation('softmax'))    # 최종적인 출력 함수

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

len(X_train)
len(X_test)

# 학습
hist = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))    # epochs : 반복 수


import glob
labels = ["ant","psychoda","mouse","rice","cockroach"]

files = glob.glob("c:/data/test2.jpg")
X = []
for i, f in enumerate(files):
    img = Image.open(f)
    img = img.convert("RGB")
    img = img.resize((32,32))
    data = np.asarray(img)
    X.append(data)
X = np.array(X)
X = X.astype('float32')/225

r = model.predict(X, batch_size=32)
res = r[0]
for i, acc in enumerate(res):
    print(labels[i],"=",int(acc*100))
print("예측결과 : ", labels[res.argmax()])
