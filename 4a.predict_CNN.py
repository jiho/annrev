#!/usr/bin/env python
#
# Train and predict a MobileNet based CNN
#
# (c) 2021 Thelma Panaiotis, Jean-Olivier Irisson, GNU General Public License v3


print('Import libraries') ## ----

# general libraries
import os
# disable tensorflow messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.tfhub_modules/')
import math

# import ipdb    # debugging, use ipdb.set_trace() in the code

# data science libraries
import pandas as pd
import numpy as np

# CNN specific functions
import tf_cnn as cnn
# from importlib import reload
# reload(tf_cnn)


print('Set options') ## ----

# Data
img_dir = os.path.expanduser('~/datasets/regent_ptB/cropped/')
batch_size = 64       # size of CNN batches
augment = True        # whether to use data augmentation (during training)
use_class_weight = True # whether to use weights inversely proportional to class freq
workers = 10          # number of parallel threads for data generators

# Model setup
train_fe = True       # whether to train the feature extractor
                      # if False, only fully connected layer(s) and classification layer will be trained.
fc_layers_sizes = [1792, 896] # size(s) of fully connected layer(s)
fc_layers_dropout = 0.4       # drop-out rate before fully connected layers
classif_layer_dropout = 0.2   # drop-out rate before classification layer

# Training
lr_method = 'constant'# learning rate evolution: 'decay' for a decaying learning rate
                      #                          'constant' for a constant learning rate
initial_lr = 0.001    # initial learning rate
decay_rate = 0.97     # rate of learning rate decay
loss = 'cce'          # loss function: 'cce' for categorical cross entropy
                      #                'sfce' for sigmoid focal cross entropy
epochs = 50           # number of epochs to train for

# Saving of weights and training history
output_dir = os.path.expanduser('~/datasets/regent_ptB/mobilenet/')
# create the directory if it does not exist 
os.makedirs(output_dir, exist_ok=True)


print('Prepare datasets') ## ----

# read input data
df = pd.read_csv('data/regent_data.tsv.gz', sep='\t', usecols=['objid', 'taxon_detailed', 'set'])
# df = pd.read_csv('data/miniregent_data.tsv.gz', sep='\t', usecols=['objid', 'taxon_detailed', 'set'])

# pick which taxonomic level to use
df = df.rename(columns={'taxon_detailed': 'taxon'})

# define path to images 
df['img_path'] = img_dir + df.objid.astype(str) + '.png'

# split train, val, test sets
dfg = df.groupby('set')
df_train = dfg.get_group('train')
df_val   = dfg.get_group('val')
df_test  = dfg.get_group('test')

# count nb of examples per taxon
taxa_counts = df_train.groupby('taxon').size()
taxa_counts

# list taxa
taxa = taxa_counts.index.to_list()

# generate class weights
# i.e. a dict with format { class number : class weight }
if use_class_weight:
    max_count = np.max(taxa_counts)
    class_weight = {}
    for idx,count in enumerate(taxa_counts.items()):
        class_weight.update({idx : math.sqrt(max_count / count[1])})
else:
    class_weight = None

# define numnber of plankton classes to train on
nb_taxa = len(taxa)

# define data generators
train_batches = cnn.DataGenerator(
    img_paths=df_train['img_path'].values, labels=df_train['taxon'].values,
    classes=taxa, batch_size=batch_size, augment=augment, shuffle=True)

val_batches   = cnn.DataGenerator(
    img_paths=df_val['img_path'].values, labels=df_val['taxon'].values,
    classes=taxa, batch_size=batch_size, augment=False, shuffle=False)
# NB: do not suffle or augment data for validation, it is useless
test_batches  = cnn.DataGenerator(
    img_paths=df_test['img_path'].values, labels=df_test['taxon'].values,
    classes=taxa, batch_size=batch_size, augment=False, shuffle=False)
total_batches = cnn.DataGenerator(
    img_paths=df['img_path'].values, labels=df['taxon'].values,
    classes=taxa, batch_size=batch_size, augment=False, shuffle=False)


print('Prepare model') # ----

# define CNN
my_cnn = cnn.Create(
    # add fully connected layer(s)
    fc_layers_sizes=fc_layers_sizes, 
    fc_layers_dropout=fc_layers_dropout, 
    # add a classification layer
    classif_layer_size=nb_taxa, 
    classif_layer_dropout=classif_layer_dropout, 
    # fine-tune the feature extractor
    train_fe=train_fe,
    # show summary after model creation
    summary=True
)

# compile CNN
my_cnn = cnn.Compile(
    my_cnn, 
    initial_lr, 
    steps_per_epoch=len(train_batches)//epochs,
    # TODO review this
    lr_method=lr_method, 
    decay_rate=decay_rate, 
    loss=loss
)
# TODO review this part


print('Train model') # ----

# train CNN
history = cnn.Train(
    model=my_cnn,
    train_batches=train_batches,
    valid_batches=val_batches,
    batch_size=batch_size,
    epochs=epochs,
    class_weight=class_weight,
    output_dir=output_dir,
    workers=workers
)

# TODO get epoch from train history


# extract training history in a readable format
h_df = pd.DataFrame(history.history)
h_df['epoch'] = history.epoch
h_df = h_df.rename(columns={'loss': 'train_loss', 'accuracy' : 'train_accuracy'})
# write or append to file
history_file = os.path.join(output_dir, 'train_history.tsv')
if os.path.exists(history_file) :
    h_df.to_csv(history_file, sep='\t', mode='a', header=False, index=False)
else :
    h_df.to_csv(history_file, sep='\t', index=False)


print('Evaluate model on whole dataset') # ----

pred = cnn.Predict(
    model=my_cnn,
    batches=total_batches,
    batch_size=batch_size,
    classes=taxa,
    output_dir=output_dir,
    workers=workers
)

# write predicted data together with input table
df['cnn_taxon'] = pred
df.drop('img_path', axis=1).to_csv('results/CNN-detailed-predictions.tsv.gz', sep='\t', index=False)

# TODO, save the full model with the best weights
