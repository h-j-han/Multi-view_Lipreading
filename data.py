
import numpy as np
from PIL import Image
import csv
import sys
import os.path
from six.moves import cPickle
import gzip
from keras.utils import np_utils
from scipy.misc import imresize

def get_class_name():
    class_name =  ["1 7 3 5 1 6 2 6 6 7",
             "4 0 2 9 1 8 5 9 0 4",
             "1 9 0 7 8 8 0 3 2 8",
             "4 9 1 2 1 1 8 5 5 1",
             "8 6 3 5 4 0 2 1 1 2",
             "2 3 9 0 0 1 6 7 6 4",
             "5 2 7 1 6 1 3 6 7 0",
             "9 7 4 4 4 3 5 5 8 7",
             "6 3 8 5 3 9 8 5 6 5",
             "7 3 2 4 0 1 9 9 5 0",
             "Excuse me",
             "Goodbye",
             "Hello",
             "How are you",
             "Nice to meet you",
             "See you",
             "I am sorry",
             "Thank you",
             "Have a good time",
             "You are welcome"]
    return class_name


def load_data(options):
     # load data
     X_train = np.zeros((5,840,36,3,options.imsize,options.imsize))
     X_val = np.zeros((5,360,36,3,options.imsize,options.imsize))
     X_test = np.zeros((5,360,36,3,options.imsize,options.imsize))

     v_train = np.zeros((5,840))
     v_val = np.zeros((5,360))
     v_test = np.zeros((5,360))

     V_train = np.zeros((5,840,5))
     V_val = np.zeros((5,360,5))
     V_test = np.zeros((5,360,5))

     y_train = np.zeros((5,840))
     y_val = np.zeros((5,360))
     y_test = np.zeros((5,360))

     Y_train = np.zeros((5,840,10))
     Y_val = np.zeros((5,360,10))
     Y_test = np.zeros((5,360,10))
     options.view = 1
     for i in range(5):
          options.view = i+1
          (X_train[i], y_train[i], Y_train[i]),(X_val[i], y_val[i], Y_val[i]),(X_test[i], y_test[i], Y_test[i]) = load_data_view(options)

          v_train[i] = options.view-1
          v_val[i] = options.view-1
          v_test[i] = options.view-1

          V_train[i] = np_utils.to_categorical(v_train[i],5)
          V_val[i] = np_utils.to_categorical(v_val[i],5)
          V_test[i] = np_utils.to_categorical(v_test[i],5)

     X_data = np.concatenate((X_train,X_val,X_test),axis =1)
     v_data = np.concatenate((v_train,v_val,v_test),axis =1)
     y_data = np.concatenate((y_train,y_val,y_test),axis =1)

     X_train = np.concatenate(X_train,axis=0)
     X_val = np.concatenate(X_val,axis=0)
     V_train = np.concatenate(V_train,axis=0)
     V_val = np.concatenate(V_val,axis=0)
     Y_train = np.concatenate(Y_train,axis=0)
     Y_val = np.concatenate(Y_val,axis=0)

     return (X_train,Y_train,y_train,V_train,v_train),(X_val,Y_val,y_val,V_val,v_val),(X_test,Y_test,y_test,V_test,v_test),(X_data,y_data,v_data)

def load_data_view(options):
    # INPUT
     color = 'color'
     (X_train, y_train) = load_ovs2_data(color=color,speakers=[1,2,3,10,11,12,13,18,19,20,21,22,23,24,25,27,33,35,36,37,38,39,45,46,47,48,50,53], imsize=options.imsize, view=str(options.view))
     (X_val, y_val) = load_ovs2_data(color=color,speakers=[4,5,7,14,16,17,28,31,32,40,41,42], imsize=options.imsize, view=str(options.view))
     (X_test, y_test) = load_ovs2_data(color=color,speakers=[6,8,9,15,26,30,34,43,44,49,51,52], imsize=options.imsize, view=str(options.view))

     X_train = X_train.astype('float32') / 255
     X_val = X_val.astype('float32') / 255
     X_test = X_test.astype('float32') / 255



     if len(X_train.shape) == 5:
         options.data_fs = X_train.shape[1] # max number of sequence
         nch, options.img_rows, options.img_cols = X_train.shape[2], X_train.shape[3], X_train.shape[4]

         options.inputshape = (options.data_fs, nch, options.img_rows, options.img_cols)
         X_train = X_train.reshape(X_train.shape[0], options.data_fs, nch, options.img_rows, options.img_cols)
         X_val = X_val.reshape(X_val.shape[0], options.data_fs, nch, options.img_rows, options.img_cols)

     Y_train = np_utils.to_categorical(y_train, options.nb_classes)
     Y_val = np_utils.to_categorical(y_val, options.nb_classes)
     Y_test = np_utils.to_categorical(y_test, options.nb_classes)


     return (X_train, y_train, Y_train),(X_val, y_val, Y_val),(X_test, y_test, Y_test)



def load_ovs2_data(speakers=[], view='1', utters=[], letters=[], pad='yes', imsize=0, color='', max_seq=36):
     #ipdb.set_trace()
     import hashlib
     sha = hashlib.sha1(str(speakers)+'xx'+view+'xx'+str(imsize)+'xx'+str(color)+'xx'+str(max_seq))
     tfilename = './data/cache/'+sha.hexdigest()+'.pgz'
     #ipdb.set_trace()
     (X, y) = load_tm_data(tfilename)
     if X is not None:
         # print ('readed from cache: '+tfilename)
         return (X, y)

     # if Xraw is None:
     (Xraw, Yraw) = load_raw_data(view, color)
     #     print ('file loaded')
     print ('OULUVS2 view:', view, speakers, color)
     if imsize>0:
         if len(Xraw.shape)==3:
             Xraw = imgresizeall(Xraw, imsize)
         else:
             Xraw = imgresizeall2(Xraw, imsize) # (31383,3,40,50) --> (31383,3,imsize,imsize)

     X, y = [], []
     idx = 0

     if pad=='yes':

         # max_seq = 36
         for i in range(Xraw.shape[0]):
             if (speakers==[] or Yraw[i][1] in speakers) and (utters==[] or Yraw[i][2] in utters) and (letters==[] or Yraw[i][0] in letters):
                 X.append(Xraw[i]) #Xraw.shape = (3,40,40)
                 if Yraw[i][3]==Yraw[i][4]: # last seq.
                     for pi in range(Yraw[i][4], max_seq):
                         # X.append( (np.ones((Xraw.shape[1], Xraw.shape[2])) * 0).astype('uint8') )
                         X.append( (np.ones(Xraw.shape[1:]) * 0).astype('uint8') )
                     y.append(Yraw[i][0])
                     idx += 1

         # print(idx, (idx,max_seq,Xraw.shape[1],Xraw.shape[2]))
         X = np.asarray(X).reshape((idx,max_seq,)+Xraw.shape[1:])
         y = np.asarray(y).reshape((idx,))

     y = y-10
     # if XYraw is not None:
     # return (X, y, Xraw, Yraw)

     with gzip.open(tfilename, 'wb') as f:
         print ('write into cache: '+tfilename)
         cPickle.dump((X, y), f)

     return (X, y)



def path_data_pkl(view,color=''):
     return './data/ouluvs2allv'+view+'cropped'+color+'.pgz'


def load_tm_data(path):
     if os.path.exists(path):
         with gzip.open(path, 'rb') as f:
             if sys.version_info < (3,):
                 (X, Y) = cPickle.load(f)
             else:
                 (X, Y) = cPickle.load(f, encoding='bytes')
 #            print (len(X), 'samples loaded')
             return (X, Y)
     else:
         return (None, None)


def load_raw_data(view='1',color=''):
     if os.path.exists(path_data_pkl(view,color)):
         with gzip.open(path_data_pkl(view,color), 'rb') as f:
             if sys.version_info < (3,):
                 (X, Y) = cPickle.load(f)
             else:
                 (X, Y) = cPickle.load(f, encoding='bytes')
 #            print (len(X), 'samples loaded')
             return (X, Y)
     else:
         raise Exception('Please check the data file path: ' + path_data_pkl(view,color))

def imgresizeall(X, nb_size=40):
     newX = []
     for i in range(X.shape[0]):
         newX.append( imresize(X[i], (nb_size,nb_size), interp='bilinear') )

     return np.asarray(newX).reshape((X.shape[0],nb_size,nb_size))

def imgresizeall2(X, nb_size=40):
     newX = []
     for i in range(X.shape[0]):
         for j in range(X.shape[1]):
             newX.append( imresize(X[i][j], (nb_size,nb_size), interp='bilinear') )

     return np.asarray(newX).reshape((X.shape[0],X.shape[1],nb_size,nb_size))


