from keras.models import Sequential, Model
from keras.layers import Input, TimeDistributed
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Masking
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization

def load_model(options):
     if options.model == 'bibn':
         model = build_model(options, bi = 1, lbn = 1)
     elif options.model == 'bi':
         model = build_model(options, bi = 1, lbn = 1)
     elif options.model == 'bn':
         model = build_model(options, bi = 1, lbn = 1)
     else:
         model = build_model(options, bi = 1, lbn = 1)
     if options.verbose > 0:
         model.summary()

     if options.loadweight != 'None':
         model.load_weights(options.loadweight,by_name=True)
         print('...model wieght of pre-trained one has been loadded...', options.loadweight)
     return model

def visual_model(options, conv1, conv2, conv3, fc, dr):

     image_input = Input(shape=options.inputshape[1:])

     i = Convolution2D(conv1, 3, 3, border_mode='valid')(image_input)
     i = Activation('relu')(i)
     i = MaxPooling2D(pool_size=(2, 2))(i)
     i = Dropout(dr)(i)

     i = Convolution2D(conv2, 3, 3, border_mode='valid')(i)
     i = Activation('relu')(i)
     i = MaxPooling2D(pool_size=(2, 2))(i)
     i = Dropout(dr)(i)

     if conv3:
         i = Convolution2D(conv3, 3, 3, border_mode='valid')(i)
         i = Activation('relu')(i)
         i = MaxPooling2D(pool_size=(2, 2))(i)
         i = Dropout(dr)(i)

     i = Flatten()(i)
     i = Dense(fc)(i)
     i = Activation('relu')(i)
     imodel = Model(input=[image_input], output=[i])

     if options.verbose > 0:
        imodel.summary()

     return imodel

def temporal_model(options, imodel, dr, lstm1, bi, lstm2, lbn):
     video_input = Input(shape=options.inputshape)
     v = TimeDistributed(imodel, name='Vis')(video_input)
     v = Dropout(dr)(v)
     v = Masking()(v)

     v = LSTM(lstm1, return_sequences=True)(v)
     if bi:
         v = LSTM(lstm1, return_sequences=True, go_backwards=True)(v)
     if lbn:
         v = BatchNormalization(mode=2)(v)

     v = LSTM(lstm2)(v)
     if lbn:
         v = BatchNormalization(mode=2)(v)

     v1 = Dense(options.nb_classes, activation='softmax', name='out_words')(v)
     v2 = Dense(5, activation='softmax', name='out_views')(v)

     vmodel = Model(input=[video_input], output=[v1,v2])

     if options.verbose > 0:
        imodel.summary()

     return vmodel

def build_model(options, conv1=128, conv2=256, conv3=0, fc=32, dr=0.4, lstm1=128, bi=0, lstm2=128, lbn=0):

     imodel = visual_model(options, conv1, conv2, conv3, fc, dr)
     vmodel = temporal_model(options, imodel, dr, lstm1, bi, lstm2, lbn)

     model = vmodel

     return model


