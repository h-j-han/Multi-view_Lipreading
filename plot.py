import numpy as np
import matplotlib.pyplot as plt
import itertools
from tsne import bh_sne
from keras.models import Model
from PIL import Image
from data import get_class_name
from sklearn.metrics import confusion_matrix

plt.ion()
def plot_lrcurve(X, D=None, figsize=(10,10)):
         if D==None:
                 D = min( [len(X[i]) for i in X] )
         print D

         #Multi task for only view classification loss
         X['acc'] = X['out_words_acc']
         X['val_acc'] = X['val_out_words_acc']

         t = [i for i in range(len(X['acc'][:D]))]

         # red dashes, blue squares and green triangles
         plt.figure(figsize=figsize)
         plt.subplot(211)
         plt.title('Blue(train), Red(Test)')
         plt.ylabel('loss')
         plt.plot(t, X['loss'][:D], 'b-', t, X['val_loss'][:D], 'r-') #, t, t**3, 'g^')

         plt.subplot(212)
         plt.ylabel('accuracy')
         plt.xlabel('epoch')
         plt.plot(t, X['acc'][:D], 'b-', t, X['val_acc'][:D], 'r-') #, t, t**3, 'g^')

         plt.show()


def plot_confusion_matrix(model,options,X_test,y_test,score1,score2,score3):
     class_name = get_class_name()
     tnames = class_name[10:]+['']

     proba = model.predict(X_test[0], verbose=0, batch_size=options.batchsize)
     yhat = proba[0].argmax(axis=-1)
     r = {'yhat':yhat, 'y':y_test[0]}
     plot_title = options.abbv + '_' +str(options.epoch) +'_acc:'+str(round(score1[3], 4))
     show_confusion_matrix(r, figsize=(6,6), target_names=tnames, title=plot_title)

     proba = model.predict(X_test[0], verbose=0, batch_size=options.batchsize/2)
     yhat = proba[0].argmax(axis=-1)
     r = {'yhat':yhat, 'y':y_test[0]}
     plot_title = options.abbv + '_' +str(options.epoch) +'_acc:'+str(round(score2[3], 4))
     show_confusion_matrix(r, figsize=(6,6), target_names=tnames, title=plot_title)

     proba = model.predict(X_test[0], verbose=0, batch_size=options.batchsize*2)
     yhat = proba[0].argmax(axis=-1)
     r = {'yhat':yhat, 'y':y_test[0]}
     plot_title = options.abbv + '_' +str(options.epoch) +'_acc:'+str(round(score3[3], 4))
     show_confusion_matrix(r, figsize=(6,6), target_names=tnames, title=plot_title)

def show_confusion_matrix(r, mode=0, cmap=plt.cm.Oranges, figsize=(8,8), verbose=0, target_names=None,title='Confusionmatrix2'):

     idx = int(max(r['y'])+2)
     target_names = target_names[:idx]

     # Compute confusion matrix
     cm = confusion_matrix(r['y'], r['yhat'])
     np.set_printoptions(precision=2)
     plt.figure(figsize=figsize)
     plt.imshow(cm, interpolation='nearest', cmap=cmap)
     plt.title(title)
     tick_marks = np.arange(len(target_names)-1)
     plt.xticks(tick_marks, target_names, rotation=90)
     plt.yticks(tick_marks, target_names)
     plt.tight_layout()
     plt.ylabel('True Labels')
     plt.xlabel('Predicted Labels')


     thresh = cm.max() / 2.
     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, cm[i, j],
              horizontalalignment="center",
              color="white" if cm[i, j] > thresh else "black")

     plt.colorbar()
     plt.grid(True)
     plt.savefig('confusion22.eps', format='eps')

def plot_tsne(model,options,X_data,Y_data,y_data):
     class_name = get_class_name()
     tnames = class_name[10:]+['']
     dataset = np.zeros((5,840+360+360))
     dataset[:,0:840] = 0
     dataset[:,840:840+360] = 1
     dataset[:,840+360:840+360+360] = 2
     im_model = Model(input=model.input, output=model.get_layer('lstm_3').output)

     x_data = np.zeros((5,X_data.shape[1],128))
     for i in range(5):
         x_data[i] = im_model.predict(X_data[i]) #default=32
         #x_data[i] = im_model.predict(X_data[i])

     x_data = np.asarray(x_data).astype('float64')
     vis_data = bh_sne(np.concatenate(x_data,axis=0))
     plt.ion()
     cat_dataset = np.concatenate(dataset,axis=0)

     for i in range(3):
          k = i
          if k ==0:
              dataset_name = 'train'
          elif k ==1:
              dataset_name = 'val'
          elif k ==2:
              dataset_name = 'test'

          plt.figure()
          classes = 'phrase'
          num_class = 10
      #plt.scatter(vis_data[start:end,0], vis_data[start:end,1], c=y_data[start:end], cmap=plt.cm.get_cmap("jet", num_class))
          formatter = plt.FuncFormatter(lambda i, *args: tnames[int(i)])
          plt.scatter(vis_data[cat_dataset==k,0], vis_data[cat_dataset==k,1], c=np.concatenate(y_data,axis=0)[cat_dataset==k], cmap=plt.cm.get_cmap("jet", num_class))
          plt.colorbar(ticks=range(num_class),format=formatter)
          plt.clim(-0.5, num_class-0.5)
          plt.title('%s_class:%s'%(dataset_name,classes))
          plt.show()

          plt.figure()
          classes = 'view'
          num_class = 5
      #plt.scatter(vis_data[start:end,0], vis_data[start:end,1], c=yp_data[start:end]-class_start, cmap=plt.cm.get_cmap("jet", num_class))
          plt.scatter(vis_data[cat_dataset==k,0], vis_data[cat_dataset==k,1], c=np.concatenate(Y_data,axis=0)[cat_dataset==k], cmap=plt.cm.get_cmap("jet", num_class))
          plt.colorbar(ticks=range(num_class) )
          plt.clim(-0.5, num_class-0.5)
          plt.title('%s_class:%s'%(dataset_name,classes))
          plt.show()

