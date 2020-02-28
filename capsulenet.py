"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.
The current version maybe only works for TensorFlow backend. Actually it will be straightforward to re-write to TF code.
Adopting to other backends should be easy, but I have not tested this. 

Usage:
       python capsulenet.py
       python capsulenet.py --epochs 50
       python capsulenet.py --epochs 50 --routings 3
       ... ...
       
Result:
    Validation accuracy > 99.5% after 20 epochs. Converge to 99.66% after 50 epochs.
    About 110 seconds per epoch on a single GTX1070 GPU card
    
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Keras`
"""

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from utils import combine_images
from PIL import Image
from keras.utils import plot_model
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.applications.vgg16 import VGG16


K.set_image_data_format('channels_last')
np.random.seed(1337)
#Diretorios de treino teste e validação
train_dir='/home/boaro/Área de Trabalho/teste/80_10_10/split3/train/'
test_dir='/home/boaro/Área de Trabalho/teste/80_10_10/split3/test/'
val_dir='/home/boaro/Área de Trabalho/teste/80_10_10/split3/valid/'


#definição da rede neural de cápsula
def CapsNet(input_shape, n_class, routings):
    """
    A Capsule Network on MNIST.
    :param input_shape: data shape, 3d, [width, height, channels]
    :param n_class: number of classes
    :param routings: number of routing iterations
    :return: Two Keras Models, the first one used for training, and the second one for evaluation.
            `eval_model` can also be used for training.
    """
    x= layers.Input(shape=input_shape)

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=3, padding='valid', activation='relu', name='conv1')(x)
    #conv1 = layers.Conv2D(filters=64, kernel_size=9, strides=1, padding='same', activation='relu', name='conv2')(conv1)
    #conv1 = Dropout(0.5)(conv1)


    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    #primarycaps = PrimaryCap(conv1, dim_capsule=16, n_channels=16, kernel_size=9, strides=2, padding='valid')
    #primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=8, kernel_size=5, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,
                             name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

    # Models for training and evaluation (prediction)
    train_model = models.Model(x, out_caps)
    eval_model = models.Model(x, out_caps)


    return train_model, eval_model

def margin_loss(y_true, y_pred):
    """
    Margin loss for Eq.(4). When y_true[i, :] contains not just one `1`, this loss should work too. Not test it.
    :param y_true: [None, n_classes]
    :param y_pred: [None, num_capsule]
    :return: a scalar loss value.
    """
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))

    return K.mean(K.sum(L, 1))


def train(model, args):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param data: a tuple containing training and testing data, like `((x_train, y_train), (x_test, y_test))`
    :param args: arguments
    :return: The trained model
    """
    # unpacking the data
    #(x_train, y_train), (x_test, y_test) = data
    

    # callbacks
    log = callbacks.CSVLogger(args.save_dir + '/log.csv')
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    tb = callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs',
                               batch_size=args.batch_size, histogram_freq=int(args.debug))
    checkpoint = callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc',
                                           save_best_only=True, save_weights_only=True, verbose=1)
    lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))
    #callbacks
    class_weight = {0: 2.7, 1: 1.} # peso para o aprendizado class weight=(qtd_imagens_da_maior_classe/qtd_da_menor classe)

    # compile the model
    model.compile(optimizer=optimizers.Adam(lr=args.lr),
                  loss=margin_loss,
                  metrics={'capsnet': 'accuracy'})

    """
    # Training without data augmentation:
    model.fit([x_train, y_train], [y_train, x_train], batch_size=args.batch_size, epochs=args.epochs,
              validation_data=[[x_test, y_test], [y_test, x_test]], callbacks=[log, tb, checkpoint, lr_decay])
    """

    # Begin: Training with data augmentation ---------------------------------------------------------------------#
        # shift up to 2 pixel for MNIST

    datagen = ImageDataGenerator(rescale=1.0/255.0)

    #gerando os dados para treino
    train_gen=datagen.flow_from_directory(
        directory=train_dir,
        target_size=(600,408),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=args.batch_size)
        
        

        # shift up to 2 pixel for MNIST

    datagen = ImageDataGenerator(rescale=1.0/255.0) #normalização da base
    #gerando os dados para validao
    val_gen = datagen.flow_from_directory(
        directory=val_dir,target_size=(600,408),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=args.batch_size)
    
    train_samples=len(train_gen.filenames)
    valid_samples=len(val_gen.filenames)
        

    # Training with data augmentation. If shift_fraction=0., also no augmentation.
    #inicio do treino
    history = model.fit_generator(generator=train_gen,
                        steps_per_epoch=int(train_samples/args.batch_size),
                        epochs=args.epochs,
                        validation_data=val_gen,
                        validation_steps=int(valid_samples/args.batch_size),
                        callbacks=[log, tb, checkpoint, lr_decay],
                        class_weight=class_weight)
    # End: Training with data augmentation -----------------------------------------------------------------------#

    ''' import json
    f=open('history.json', 'w')
    f.write(json.dumps(history.history))
    f.close()
    '''
    model.save_weights(args.save_dir + '/trained_model.h5')
    print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)

    from utils import plot_log
    plot_log(args.save_dir + '/log.csv', show=True)

    return model


def get_metric(y_pred,y_true):
    tp = fp =  tn =  fn = 0
    for x in range(y_pred.shape[0]):
        if (y_pred[x] == 0 and y_true[x] == 0):
            tn+=1
        elif (y_pred[x]==1 and y_true[x] == 0):
            fp+=1
        elif (y_pred[x] == 0 and y_true[x] == 1):
            fn+=1
        elif (y_pred[x] == 1 and y_true[x] ==1):
            tp+=1

    return calc_metric(tp,fp,tn,fn)
    # print("{} pixel(s) de massa correto(s) - tp:{} , fp:{}, tn:{} , fn:{}".format(tp,tp,fp,tn,fn))

def calc_metric(tp,fp,tn,fn):
    dice = (2.0 * tp) / ((2.0 * tp) + fp + fn)
    jaccard = (1.0 * tp) / (tp + fp + fn) 
    sensitivity = (1.0 * tp) / (tp + fn)
    specificity = (1.0 * tn) / (tn + fp)
    accuracy = (1.0 * (tn + tp)) / (tn + fp + tp + fn)
    auc = 1 - 0.5 * (((1.0 * fp) / (fp + tn)) + ((1.0 * fn) / (fn + tp)))
    
    return dice,jaccard,sensitivity,specificity,accuracy,auc

def test(model, args):
        # shift up to 2 pixel for MNIST
    #gerando dados de test
    datagen = ImageDataGenerator(rescale=1.0/255.0)

    test_gen = datagen.flow_from_directory(
        directory=test_dir,target_size=(600,408),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=1,
        shuffle=False)
    test_samples = len(test_gen.filenames)

    y_test = [] #classes correspondente das imagens
    for i in range(0, test_samples): #separação das imagens para teste da rede
        test_img, test_label = next(test_gen)
        y_test.append(int(test_label[0][1]))

    y_pred= model.predict_generator(test_gen, len(y_test))
    #print(np.argmax(y_pred,1),y_test)
    
    '''print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])'''

    #calcula as metricas 
    
    from sklearn.metrics import classification_report,confusion_matrix
    confusion=confusion_matrix(y_test,np.argmax(y_pred,1))
    report= classification_report(y_test,np.argmax(y_pred,1))
    print(report)
    print(confusion)
    dice,jaccard,sensitivity,specificity,accuracy,auc=get_metric(np.argmax(y_pred,1),y_test)
    print('dice = {}, jaccard = {}, sensitivity={}, specificity={}, accuracy={}, auc={}'.format(dice,jaccard,sensitivity,specificity,accuracy,auc))

    '''img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon.png")
    print()
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    plt.show()'''


def manipulate_latent(model, data, args):
    print('-'*30 + 'Begin: manipulate' + '-'*30)
    x_test, y_test = data
    index = np.argmax(y_test, 1) == args.digit
    number = np.random.randint(low=0, high=sum(index) - 1)
    x, y = x_test[index][number], y_test[index][number]
    x, y = np.expand_dims(x, 0), np.expand_dims(y, 0)
    noise = np.zeros([1, 10, 16])
    x_recons = []
    for dim in range(16):
        for r in [-0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25]:
            tmp = np.copy(noise)
            tmp[:,:,dim] = r
            x_recon = model.predict([x, y, tmp])
            x_recons.append(x_recon)

    x_recons = np.concatenate(x_recons)

    img = combine_images(x_recons, height=16)
    image = img*255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + '/manipulate-%d.png' % args.digit)
    print('manipulated result saved to %s/manipulate-%d.png' % (args.save_dir, args.digit))
    print('-' * 30 + 'End: manipulate' + '-' * 30)



if __name__ == "__main__":
    import os
    import argparse
    from keras.preprocessing.image import ImageDataGenerator
    from keras import callbacks

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
                        #0.2
    parser.add_argument('--lr_decay', default=0.25, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.5, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',default='no',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default='./result/trained_model.h5',
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # define model
    model, eval_model = CapsNet(input_shape=(600,408,3),
                                                  n_class=2,
                                                  routings=args.routings)
    model.summary()

    #verificar como fazer para passar treino
    train(model=model, args=args)

    # train or test
    #if args.weights is not None:  # init the model weights with provided one
    #    model.load_weights(args.weights)
        # model2=load_model(args.weights)

    #não entra nesse IF
    #if args.testing is None:
    #    train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)
    #else:  # as long as weights are given, will run testing
    #    if args.weights is None:
    #        print('No weights are provided. Will test using random initialized weights.')
        #manipulate_latent(manipulate_model, (x_test, y_test), args)
    test(model=eval_model, args=args)
