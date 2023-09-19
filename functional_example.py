import tensorflow as tf
import tensorflow.keras as keras
from keras.utils import to_categorical

fashion_mnist = tf.keras.datasets.fashion_mnist

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

train_images = train_images/255
test_images = test_images/255

input = keras.Input(shape = (28,28,1), name="input layer")
y = keras.layers.Conv2D(128, (3,3), activation='relu')(input)
y = keras.layers.MaxPooling2D((2,2))(y)
y = keras.layers.Conv2D(64, (3,3), activation='relu')(y)
y = keras.layers.MaxPooling2D((2,2))(y)
y = keras.layers.Flatten()(y)
y = keras.layers.Dense(128, activation='relu')(y)
output = keras.layers.Dense(10, activation='softmax')(y)
model = keras.Model(inputs=input, outputs=output)

model.fit(train_images,train_labels,batch_size=16, epochs=10, validation_data=(test_images, test_labels))
model.evaluate(test_images, test_labels)