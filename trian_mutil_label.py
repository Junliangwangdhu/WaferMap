from __future__ import absolute_import, division

from keras.models import Model

from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Activation
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint
import keras
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.model_selection import train_test_split

import numpy as np
from lib.layers_train import ConvOffset2D_train
from keras import backend as K
import tensorflow as tf

import os,time



class Train_mix():

    def __init__(self,datapath,lr=0.0001,fullnet_num=128,conv_num=32,deconv_size=(3,3)):
        self.datapath=datapath
        self.lr = lr
        self.fullnet_num = fullnet_num
        self.conv_num = conv_num
        self.deconv_size = deconv_size

    def auc1(self,inputs,pre):
        inputs_T=inputs.T
        pre_T=pre.T
        #for i in range(len(inputs_T)):
        acc=np.mean(np.equal(inputs_T,pre_T).astype(np.float64),axis=1)
        return acc

    def lrelu(self,x, leak=0.2, name="lrelu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def acc_myself(self,y_true, y_pre):
        y_pre = tf.round(y_pre)
        r = tf.equal(y_true, y_pre)
        r = tf.cast(r, tf.float32)
        r = tf.reduce_sum(r, axis=1)
        d = tf.zeros_like(r, dtype=tf.float32) + 8
        c = tf.equal(r, d)
        c = tf.cast(c, tf.float32)

        return tf.divide(tf.reduce_sum(c), tf.cast(tf.size(c), tf.float32))

    def build_model(self,inputs_shape,classes=8,trainable=True):
        bn_axis=3

        inputs=Input(shape=inputs_shape)
        x = ConvOffset2D_train(1, name='conv_1_offset')(inputs)
        x = Conv2D(self.conv_num, (3, 3), strides=(2, 2), padding='same', name='conv_1', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name='batch_normalization_1')(x)
        x = Activation('relu', name='activation_1')(x)

        # Conv_2 layer
        x = ConvOffset2D_train(32, name='conv_2_offset')(x)
        x = Conv2D(self.conv_num*2, (3, 3), strides=(2, 2), padding='same', name='conv_2', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name='batch_normalization_2')(x)
        x = Activation('relu', name='activation_2')(x)

        # Conv_3 layer
        x = ConvOffset2D_train(64, name='conv_3_offset')(x)
        x = Conv2D(self.conv_num*4, (3, 3), strides=(2, 2), padding='same', name='conv_3', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name='batch_normalization_3')(x)
        x = Activation('relu', name='activation_3')(x)

        # Conv_4 layer
        x = ConvOffset2D_train(128, name='conv_4_offset')(x)
        x = Conv2D(self.conv_num*8, (3, 3), padding='same', name='conv_4', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name='batch_normalization_4')(x)
        x = Activation('relu', name='activation_4')(x)

        # Conv_5 layer
        x = ConvOffset2D_train(256, name='conv_5_offset')(x)
        x = Conv2D(self.conv_num*4, (3, 3), strides=(2, 2), padding='same', name='conv_5', trainable=trainable)(x)
        x = BatchNormalization(axis=bn_axis, name='batch_normalization_5')(x)
        x = Activation('relu', name='activation_5')(x)

        # Pooling layer
        x = GlobalAveragePooling2D()(x)

        # fc layer
        outputs = Dense(classes, activation='sigmoid', name='fc', trainable=trainable)(x)

        return inputs, outputs

    def start_train(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        K.tensorflow_backend.set_session(session)

        data=np.load(os.path.join(self.datapath))
        trainx=data["arr_0"]
        trainy = data["arr_1"]

        trainx=np.expand_dims(trainx,axis=-1)

        data_shape=trainx.shape[1:]

        inputs, outputs = self.build_model(data_shape, classes=trainy.shape[-1], trainable=True)

        model = Model(inputs=inputs, outputs=outputs)
        model.summary()

        loss = keras.losses.binary_crossentropy
        #loss = keras.losses.categorical_crossentropy
        optimizer = keras.optimizers.SGD(lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss=loss, optimizer=optimizer, metrics=[self.acc_myself])

        x_train, x_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.2, random_state=10)

        t=time.strftime('%Y-%b-%d_%H-%M-%S')

        checkpointer = ModelCheckpoint(r'save_model/'+ 'model_{0}.h5'.format(t),
                                       monitor='val_acc_myself',
                                       verbose=1, save_best_only=True,period=1
                                       )
        self.lossHistory=LossHistory()

        batch_size=32
        model.fit(x_train[:1000], y_train[:1000], batch_size=batch_size,
                  epochs=5000, validation_data=(x_test[:1000], y_test[:1000]), verbose=2,
                  callbacks=[TensorBoard(log_dir=r'data/TensorBoard/'+t,
                                         histogram_freq=0,
                                         write_graph=True,
                                         write_images=True), checkpointer,self.lossHistory])


class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc_myself'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc_myself'))
        # 每五秒按照当前容器里的值来绘图
        # if int(time.time()) % 5 == 0:
        #     self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        #     self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        #     self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        #     self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc_myself'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc_myself'))
        # 每五秒按照当前容器里的值来绘图
        # if int(time.time()) % 5 == 0:
        #     self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        #     self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        #     self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        #     self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        #plt.xlabel(type)
        plt.legend(loc="upper right")
        plt.savefig(r'picture/Cache/'+ '{0}_draw.jpg'.format(label))
        plt.close()

    # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')
