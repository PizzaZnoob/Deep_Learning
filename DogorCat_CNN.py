# Importing libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data preprocessing
# Preprocessing the Training set

train_datagen = ImageDataGenerator(
    rescale=1. / 255,  # Feature scaling
    shear_range=0.2,  # Image augmentation to prevent overfitting
    zoom_range=0.2,
    horizontal_flip=True)

train_set = train_datagen.flow_from_directory(
    'Convolutional Neural Networks (CNN)/dataset/training_set',
    target_size=(64, 64),  # Image size
    batch_size=32,  # Images on each training batch
    class_mode='binary')  # Outcome categorical or binary

# Preprocessing the test set

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_set = test_datagen.flow_from_directory(
    'Convolutional Neural Networks (CNN)/dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Building the CNN
# Initializing the CNN

cnn = tf.keras.models.Sequential()

# Step 1 -> First convolutional layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 pooling

cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer with max pool layer

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 Flattening

cnn.add(tf.keras.layers.Flatten())

# Step 4 - Full Connection / adding a fully connected layer ann

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 - Output Layer

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Training the CNN
# Compiling the CNN

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set

cnn.fit(x=train_set, validation_data=test_set, epochs=25)

# Using the trained CNN
# Making a single prediction

import numpy as np
from tensorflow.keras.preprocessing import image

test_image = image.load_img('Convolutional Neural Networks (CNN)/dataset/single_prediction/cat_or_dog_1.jpg',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)  # Converting the pil image to array format
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image/255.0)
test_set.class_indices

if result[0][0] > 0.5:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)