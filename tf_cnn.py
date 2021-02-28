#
# Functions to generate data, create, compile, train, and predict a MobileNet model
#
# (c) 2021 Thelma Panaiotis, Jean-Olivier Irisson, GNU General Public License v3


# general libraries
import os
# disable tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# import ipdb    # debugging, use ipdb.set_trace() in the code

import numpy as np

# image manipulation
import lycon
from imgaug import augmenters as iaa

# convert labels into binary format
from sklearn.preprocessing import MultiLabelBinarizer

# tensorflow stuff
import tensorflow as tf
from tensorflow.keras import utils, layers, optimizers, losses, callbacks 
import tensorflow_hub as hub
import tensorflow_addons as tfa

# # explicitly allow memory to grow
# # this is necessary to run from the command line apparently...
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class DataGenerator(utils.Sequence):
    """
    Generates data batches
        
    Args:
        img_paths (list/ndarray, str): paths to images
        labels (list/ndarray, str): image labels (as text)
        classes (list, str): all classes present in the data set
            NB: `labels` may only contain a aubset of classes with some datasets
                Specifying `classes` ensure the result is consistent
        batch_size (int): number of images in a batch
        input_dims (tuple, int): dimensions of the input images for the network
        shuffle (bool) : whether to shuffle inputs between epochs
                         (usually True for training, False otherwise)
        augment (bool) : whether to use data augmentation on the input
                         (usually True for training, False otherwise)
        preserve_size (bool) : when True, small images are not scaled up
    
    Returns:
        a batch of `batch_size` images (4D ndarray) and one-hot encoded labels (2D ndarray)

    """
    def __init__(self, img_paths, labels, classes, batch_size=32, input_dims=(224, 224, 3),
                 shuffle=False, augment=False, preserve_size=False):
        'Initialization of settings'
        # initialize constants
        self.img_paths     = img_paths 
        self.labels        = labels
        self.batch_size    = batch_size
        self.input_dims    = input_dims
        # TODO why do we need the third dimension?
        self.shuffle       = shuffle
        self.augment       = augment
        self.preserve_size = preserve_size
        # initialise the ont-hot encoder
        mlb = MultiLabelBinarizer(classes=classes)
        self.class_encoder = mlb
        
        self.on_epoch_end()

    def __len__(self):
        'Compute the number of batches to cover the dataset in one epoch'
        return int(np.ceil(len(self.img_paths) / self.batch_size))
        # NB: use np.ceil instead of np.floor to be sure to see all items
        #     (important for the test set in particular)

    def on_epoch_end(self):
        'Update indexes after each epoch'
        # reinitialise indexes
        self.indexes = np.arange(len(self.img_paths))
        # and, if chosee, shuffle them between epochs, to make sure the batches
        # are not always the same
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
                
    def padding_value(self, img):
        'Compute value to use to pad an image, as the median value of border pixels'
        # get height and width of image
        h,w = img.shape[0:2]
        
        # concatenate border pixels in an array
        borders = np.concatenate((
            img[:, 0],         # left column
            img[:, w-1],       # right column
            img[0, 1:w-2],     # top line without corners
            img[h-1, 1:w-2],   # bottom line without corners        
        ), axis=0)
        
        # compute the median
        pad_value = np.median(borders)
        
        return pad_value
    
    def augmenter(self, images):
        """
        Define a data augmenter which doses horizontalf flip (50% chance),
        vertical flip (50% chance), zoom and shear
        """
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    # TODO what happens when images are zoomed and there is not enough padding?
                    shear=(-15, 15),  # shear by -15 to +15 degrees
                    mode='edge', # pad images with border picels
                ),
            ],
            random_order=True # apply these transformations in random order
        )
        return seq(images=images)

    def __getitem__(self, index):
        'Generate one batch of data'
                
        # pick indexes of images for this batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # select and load images from this batch
        batch_paths = [self.img_paths[i] for i in indexes]
        batch_orig_images = [lycon.load(p)/255 for p in batch_paths]
        
        # resize images to the input dimension of the network
        batch_prepared_images = []
        input_size = self.input_dims[0]
        for img in batch_orig_images:
            h,w = img.shape[0:2]
            
            # compute largest dimension (hor or ver)
            dim_max = max(h,w)
            
            # if size is not preserved or image is larger than input_size, resize image to input_size
            if not(self.preserve_size) or (dim_max > input_size):
                # resize image so that largest dim is now equal to input_size
                img = lycon.resize(
                    img, 
                    height = max(h*input_size//dim_max,1), 
                    width  = max(w*input_size//dim_max,1), 
                    interpolation=lycon.Interpolation.AREA
                )
                h,w = img.shape[0:2]
            
            # create a square, empty output, of desired dimension, filled with padding value
            pad_value = self.padding_value(img)
            img_square = np.full(self.input_dims, pad_value)
            
            # compute number of pixels to leave blank 
            offset_ver = int((input_size-h)/2) # on top and bottom of image
            offset_hor = int((input_size-w)/2) # on left and right of image
            
            # replace pixels by input image
            img_square[offset_ver:offset_ver+h, offset_hor:offset_hor+w] = img
            batch_prepared_images.append(img_square)
        
        # convert to array of images        
        batch_prepared_images = np.array([img for img in batch_prepared_images], dtype='float32')
        # TODO review for speed, possibly
        
        # augment images
        if self.augment == True:
            batch_prepared_images = self.augmenter(batch_prepared_images)
            
        # extract the labels corresponding to the selected indexes
        batch_labels = [self.labels[i] for i in indexes]
        batch_encoded_labels = self.class_encoder.fit_transform([[l] for l in batch_labels])
        
        # return reshaped images and labels
        return batch_prepared_images,batch_encoded_labels


def Create(fc_layers_nb, fc_layers_size, fc_layers_dropout,
           classif_layer_size, classif_layer_dropout,
           train_fe=False, summary=True):

    """
    Generates a CNN model. 
    
    Args:
        fc_layers_nb (int): number of fully connected layers 
        fc_layers_size (int): size of fully connected layers 
        fc_layers_dropout (float): dropout of fully connected layers 
        classif_layer_size (int): size of classification layer
                                  (i.e. number of classes)
        classif_layer_dropout (float): dropout of classification layer
        train_fe (bool): whether to train the feature extractor (True) or only
            classification head (False)
        summary (bool): whether to show a model summary
    
    Returns:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model
        
    """
    
    # Initiate empty model
    model = tf.keras.Sequential()
    
    # MobileNet V2 feature extractor
    # fe_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    fe_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    fe_layer = hub.KerasLayer(fe_url, input_shape=(224, 224, 3))
    # set feature extractor trainability
    fe_layer.trainable = train_fe
    model.add(fe_layer)
    
    # Fully connected layers
    if fc_layers_nb:
        for i in range(fc_layers_nb):
            if fc_layers_dropout:
                model.add(layers.Dropout(fc_layers_dropout))
            model.add(layers.Dense(fc_layers_size, activation='relu'))
    # TODO is it normal to have dropout *before* the first dense layer?
    
    # Classification layer
    if classif_layer_dropout:
        model.add(layers.Dropout(classif_layer_dropout))
    model.add(layers.Dense(classif_layer_size))
    # TODO no softmax?

    if summary:
        model.summary()

    return model


def Compile(model, initial_lr, steps_per_epoch, lr_method='constant',
            decay_rate=None, loss='cce'):
    """
    Compiles a CNN model. 
    
    Args:
        model (tf.keras.Sequential): CNN model to compile
        lr_method (str): method for learning rate. 'constant' for a constant learning rate, 'decay' for a decay
        initial_lr (float): initial learning rate. If lr_method is 'constant', set learning rate to this value
        steps_per_epochs (int): number of training steps at each epoch. Usually number_of_epochs // batch_size
        decay_rate (float): rate for learning rate decay
        loss (str): method to compute loss.
          'cce' for CategoricalCrossentropy
          (see https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalCrossentropy),
          'sfce' for SigmoidFocalCrossEntropy
          (see https://www.tensorflow.org/addons/api_docs/python/tfa/losses/SigmoidFocalCrossEntropy),
          usefull for unbalanced classes
    
    Returns:
        model (tensorflow.python.keras.engine.sequential.Sequential): compiled CNN model
        
    """
    # TODO if lr_method='decay', decay_rate in mandatory

    # Define learning rate
    if lr_method == 'decay':
        lr = tf.keras.optimizers.schedules.InverseTimeDecay(
            initial_lr, steps_per_epoch, decay_rate)
        # NB: decayed learning rate computed as 
        #   initial_learning_rate / (1 + decay_rate * step / decay_step)
        # TODO how should decay_step be chosen??
    else: # Keep constant learning rate
        lr = initial_lr
    
    # Define optimizer
    optimizer = optimizers.Adam(learning_rate=lr)
    
    # Define loss
    if loss == 'cce':
        # loss = losses.CategoricalCrossentropy(from_logits=True,reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
        loss = losses.CategoricalCrossentropy(from_logits=True, reduction=losses.Reduction.AUTO)
        # TODO consider using https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy to avoid having to one-hot encode the labels
    elif loss == 'sfce':
        loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,reduction=losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    # Compile model
    model.compile(
      optimizer=optimizer,
      loss=loss,
      metrics='accuracy'
    )
    
    return model


def Train(model, train_batches, valid_batches,
          batch_size, epochs, class_weight=None, output_dir='.', workers=1):
    """
    Trains a CNN model. 
    
    Args:
        model (tf.keras.Sequential): CNN model to train
        train_batches (datasets.DataGenerator): batches of data for training
        valid_batches (datasets.DataGenerator): batches of data for validation
        batch_size (int): size of batches
        epochs (int): number of epochs to train for
        class_weight (dict): weights for classes
        output_dir (str): directory where to save model weights

    Returns:
        history (tf.keras.callbacks.History) that contains loss and accuracy for
        the traning and validation dataset.
    """
    
    # Set callbacks to save model weights when validation loss improves
    checkpoint_path = os.path.join(output_dir, "checkpoint.{epoch:03d}.ckpt")
    checkpoint_callback = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        save_weights_only=True,
        save_freq='epoch',
        verbose=1
    )
    
    # Detect if some weights already exist and restart from there
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is not None:
        model.load_weights(latest)
        # get epoch from the weights file name
        initial_epoch = int(latest.split('.')[1])
        print("Restart training after epoch " + str(initial_epoch))
    else :
        initial_epoch = 0
    # TODO check if this works correctly with learning rate decay
    
    # Fit the model
    history = model.fit(
        x=train_batches,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        initial_epoch=initial_epoch,
        validation_data=valid_batches,
        class_weight=class_weight,
        workers=workers
    )
        
    return history


def Predict(model, batches, batch_size=None, classes=None, output_dir='.', workers=1):
    """
    Predict batches from a CNN model
    
    Args:
        model (tensorflow.python.keras.engine.sequential.Sequential): CNN model
        batches (datasets.DataGenerator): batches of data to predict
        batch_size (int): number of images per batch
        classes (list): None or list of class names; when None, the function
            returns the content of the classification layer 
        output_dir (str): directory where to find the saved model weights
        workers (int): number of CPU workers to prepare data

    
    Returns:
        prediction (ndarray): with as many rows as input and, as columns:
            logits when `classes` is None
            class names when `classes` is given
        
    """
    
    # Load last saved weights to CNN model
    latest = tf.train.latest_checkpoint(output_dir)
    if latest is None:
        raise RuntimeError('No model checkpoints available')
    else :
        model.load_weights(latest)

    # Predict all batches
    prediction = model.predict(batches, batch_size=batch_size, workers=workers)
    # NB: pred is an array with:
    # - as many lines as there are items in the batches to predict
    # - as many columns as there are classes
    # and it contains the models' logits (output of the classification layer)
    
    if classes is not None:
        # convert it to predicted classes
        prediction = np.array(classes)[np.argmax(prediction, axis=1)]

    return prediction