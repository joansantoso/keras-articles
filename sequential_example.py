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

model = keras.Sequential()
model.add(keras.Input(shape=(28,28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(train_images,train_labels,batch_size=16, epochs=10, validation_data=(test_images, test_labels))
model.evaluate(test_images, test_labels)