import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.layers import Lambda, Input, Flatten, Dense
from keras.preprocessing import image
from keras.optimizers import RMSprop
from keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Get number of classes from the number of directories in train folder
dirs = glob('dataset/train/*')
print('Classes: ' + str(len(dirs)))
x=len(dirs)

#set path of training example
train_path = 'dataset/train'
val_path = 'dataset/validation'

#create model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    #after 6 layers we use flatten to create single vector along with activation function
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),

  #since it's a multi-class hence we'll use softmax activation function.
    tf.keras.layers.Dense(x, activation='softmax')
])

#compile model
model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',metrics=['accuracy'])


train_datagen = ImageDataGenerator(
    rescale=1./255.,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    val_path,
    target_size = (224,224),
    batch_size = 32,
    class_mode = 'categorical')



#train
history = model.fit(
      train_generator,
      steps_per_epoch=10,
      epochs=10,
      validation_steps=10,
      validation_data=validation_generator)


#save model
model.save('mymodel.h5')













