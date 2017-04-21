import tensorflow as tf
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
import numpy as np

def train_model(train,test,vocab_size,sequence_chunk_size,n_epoch=2,n_units=128,dropout=0.6,learning_rate=0.0001):

    trainX = train['sub_seqs']
    trainY = train['sub_label']
    testX =  test['sub_seqs']
    testY =  test['sub_label']

    # Sequence padding
    trainX = pad_sequences(trainX, maxlen=sequence_chunk_size, value=0.,padding='post')
    testX = pad_sequences(testX, maxlen=sequence_chunk_size, value=0.,padding='post')

    # Converting labels to binary vectors
    trainY = to_categorical(trainY, nb_classes=vocab_size)
    testY = to_categorical(testY, nb_classes=vocab_size)

    # Network building
    net = tflearn.input_data([None, sequence_chunk_size])
    net = tflearn.embedding(net, input_dim=vocab_size, output_dim=128,trainable=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.dropout(net, 0.5)
    net = tflearn.fully_connected(net, vocab_size, activation='softmax',weights_init=tflearn.initializations.xavier())
    net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, clip_gradients=0., tensorboard_verbose=2)
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
              batch_size=32,n_epoch=n_epoch)

    return model