def convnet_model(input_shape, training_labels, unique_labs, dropout_rate=dropout_rate):
    print('CNN input shape', input_shape)

    unique_labs = len(np.unique(training_labels))

    model = Sequential()

    #hidden layer 1
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid')) # padding='valid'

    #hidden layer 2 with Pooling
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid'))

    model.add(BatchNormalization())

    #hidden layer 3 with Pooling
    model.add(Conv3D(filters=8, kernel_size=(3, 3, 1), input_shape=input_shape, activation='relu', padding='same'))
    model.add(Dropout(dropout_rate))
    model.add(MaxPool3D(pool_size=(2, 2, 2), padding='valid'))

    model.add(Flatten())
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(unique_labs, activation='softmax'))

    return model