from __future__ import print_function
from optparse import OptionParser
import sys

def default(str):
     return str + ' [Default: %default]'

def parsecmd():
     parser = OptionParser("python "+sys.argv[0]+" <options>")
     parser.add_option('-e', '--epoch', help=default('number of max epoch'), default=200, type="int")
     parser.add_option('-m', '--model', help=default('name of the model'), default='bibn', type="string")
     parser.add_option('-b', '--batchsize', help=default('size of the batch'), default=64, type='int')
     parser.add_option('-w', '--savepath', help=default('path where to save the model weight.'), default='./weight/', type='string')
     parser.add_option('-d', '--loadweight', help=default('path where to load the pre-trained model weight. if None then do not load'), default='None', type='string')
     parser.add_option('-s', '--imsize', help=default('size of image'), default=20, type='int')
     parser.add_option('-l', '--lr', help=default('learning rate'), default=0.001, type='float')
     parser.add_option('-u', '--multiw', help=default('multi task weight'), default=0.7, type='float')
     parser.add_option('-v', '--verbose', help=default('verbose option'), default=1, type='int')

     options, tmp = parser.parse_args(sys.argv[1:])
     options.nb_classes = 10
     return options

def title(options):
     options.abbv = 'M'
     options.abbv += options.model
     options.abbv += '_s'
     options.abbv += str(options.imsize)
     options.abbv += '_e'
     options.abbv += str(options.epoch)
     options.abbv += '_b'
     options.abbv += str(options.batchsize)
     print('Multi-view VSR using Multi-task learning ' + options.abbv)

def print_performance_frontal(model,options,X_val,Y_val,V_val,X_test,Y_test,V_test):
      print('[view 0] Val')
      score = model.evaluate(X_val, [Y_val,V_val], verbose=0, batch_size=options.batchsize)
      print('phrase acc \t', round(score[3], 4), '\tLoss:', round(score[1], 4))
      print('view   acc \t', round(score[4], 4), '\tLoss:', round(score[2], 4))

      print('[view 0] Test') #X_test[0] for frontal
      score3 = model.evaluate(X_test[0], [Y_test[0],V_test[i]], verbose=0, batch_size=options.batchsize*2)
      print('bns*2 acc \t', round(score3[3], 4), '\tLoss:', round(score3[1], 4))
      score1 = model.evaluate(X_test[0], [Y_test[0],V_test[0]], verbose=0, batch_size=options.batchsize)
      print('bns   acc \t', round(score1[3], 4), '\tLoss:', round(score1[1], 4))
      score2 = model.evaluate(X_test[0], [Y_test[0],V_test[0]], verbose=0, batch_size=options.batchsize/2)
      print('bns/2 acc \t', round(score2[3], 4), '\tLoss:', round(score2[1], 4))

      return score, score1,score2

def print_performance_multi(model,options,X_val,Y_val,V_val,X_test,Y_test,V_test):
      print('[view 0] Val')
      score = model.evaluate(X_val, [Y_val,V_val], verbose=0, batch_size=options.batchsize)
      print('phrace acc \t', round(score[3], 4), '\tLoss:', round(score[1], 4))
      print('view   acc \t', round(score[4], 4), '\tLoss:', round(score[2], 4))
           # test
      for i in reversed(range(5)):
          print('[view %d] Test' % i) #X_test[0] for frontal
          score3 = model.evaluate(X_test[i], [Y_test[i],V_test[i]], verbose=0, batch_size=options.batchsize*2)
          print('bns*2 acc\t', round(score3[3], 4), '\tLoss:', round(score3[1], 4))
          score1 = model.evaluate(X_test[i], [Y_test[i],V_test[i]], verbose=0, batch_size=options.batchsize)
          print('bns   acc\t', round(score1[3], 4), '\tLoss:', round(score1[1], 4))
          score2 = model.evaluate(X_test[i], [Y_test[i],V_test[i]], verbose=0, batch_size=options.batchsize/2)
          print('bns/2 acc\t', round(score2[3], 4), '\tLoss:', round(score2[1], 4))

      return score, score1,score2,score3

