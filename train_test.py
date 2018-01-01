
import numpy as np
from tools import parsecmd, title, print_performance_multi
from data import load_data
from model import load_model
from plot import plot_lrcurve, plot_tsne, plot_confusion_matrix
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
import time

np.random.seed(1338)

if __name__ == "__main__":
    options = parsecmd()
    title(options)
    start_time = time.clock()
    (X_train,Y_train,y_train,V_train,v_train),(X_val,Y_val,y_val,V_val,v_val),(X_test,Y_test,y_test,V_test,v_test),(X_data,y_data,v_data) = load_data(options)
    print('data loading  time ... ' + str(round(int(time.clock() - start_time)/60,1)))
    model = load_model(options)

    optimizer = Adam(lr=options.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',loss_weights=[1., options.multiw], optimizer=optimizer, metrics=['accuracy'])

    if options.epoch > 0:
        start_time = time.clock()
        options.wfullpath = options.savepath + '_W' + options.abbv + '.h5'
        checkpointer = ModelCheckpoint(filepath=options.wfullpath, monitor='val_loss', verbose=options.verbose, save_best_only=True)
        callbacks=[checkpointer]

        hist = model.fit(X_train, [Y_train,V_train], batch_size=options.batchsize, nb_epoch=options.epoch,
                    verbose=options.verbose, validation_data=(X_val, [Y_val,V_val]), callbacks=callbacks)

        print('training time ... ' + str(round(int(time.clock() - start_time)/60,1)))
        plot_lrcurve(hist.history,figsize=(12,5))

    [ _ , score1, score2,score3 ] = print_performance_multi(model,options,X_val,Y_val,V_val,X_test,Y_test,V_test)
    plot_confusion_matrix(model,options,X_test,y_test,score1,score2,score3)
    plot_tsne(model,options,X_data,v_data,y_data)





