import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization

class AlexNet(Sequential):
    def __init__(self, input_shape, num_classes,metrics):
        super().__init__()

        self.add(Conv2D(96, kernel_size=(11,11), strides= 4,
                        padding= 'valid', activation= 'relu',
                        input_shape= input_shape,
                        kernel_initializer= 'he_normal'))
        self.add(MaxPool2D(pool_size=(2,2), strides= (2,2),
                              padding= 'valid'))
        self.add(BatchNormalization())

        
        self.add(Conv2D(256, kernel_size=(5,5), strides= 1,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(MaxPool2D(pool_size=(2,2), strides= (2,2),
                              padding= 'valid')) 
        self.add(BatchNormalization())

        
        self.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(BatchNormalization())

        
        self.add(Conv2D(384, kernel_size=(3,3), strides= 1,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(BatchNormalization())

        
        self.add(Conv2D(256, kernel_size=(3,3), strides= 1,
                        padding= 'valid', activation= 'relu',
                        kernel_initializer= 'he_normal'))
        self.add(MaxPool2D(pool_size=(2,2), strides= (2,2),
                              padding= 'valid'))
        self.add(BatchNormalization())

        
        self.add(Flatten())
        self.add(Dense(4096, activation= 'relu'))
        self.add(Dropout(0.4))
        self.add(BatchNormalization())
        self.add(Dense(4096, activation= 'relu'))
        self.add(Dropout(0.4))
        self.add(BatchNormalization())
        self.add(Dense(1000, activation= 'relu'))
        self.add(Dropout(0.4))
        self.add(BatchNormalization())
        self.add(Dense(num_classes, activation= 'softmax'))

        self.compile(optimizer= tf.keras.optimizers.Adam(0.001),
                    loss='categorical_crossentropy',
                    metrics=metrics)
