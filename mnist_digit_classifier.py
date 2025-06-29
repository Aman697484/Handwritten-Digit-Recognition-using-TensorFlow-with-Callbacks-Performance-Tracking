import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data() #load data

x_train,x_test = x_train/255.0,x_test/255.0 #normalize data

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape = (28,28)),
        tf.keras.layers.Dense(128,activation ='relu'),
        tf.keras.layers.Dense(10,activation = 'softmax')  #build the model
])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])                       #compile the model

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=3)
checkpoint = ModelCheckpoint("best_model.h5", save_best_only=True)

history = model.fit(x_train, y_train,
          epochs=20, # Train for the full 20 epochs
          validation_split=0.2,
          callbacks=[early_stop, checkpoint])


test_loss,test_acc = model.evaluate(x_test,y_test)
print("Accuracy\n:",test_acc)                   #evaluation

predictions = model.predict(x_test)

import numpy as np
# Changed index from 2 to 0 to display a different image
print("Predicted label:",np.argmax(predictions[0]))
print("Actual label:",y_test[0])

# Changed index from 2 to 0 to display a different image
plt.imshow(x_test[0],cmap = 'grey')
plt.title(f"Predicted label: {np.argmax(predictions[0])},Actual label: {y_test[0]}")
plt.axis('off')
plt.show() # Corrected

# Accuracy plot
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

# Loss plot
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

# The model saving and loading parts are removed as we are training the original model directly
# model.save("my_mnist_model.h5")
# from tensorflow.keras.models import load_model
# model = load_model("my_mnist_model.h5")
# loss, acc = model.evaluate(x_test, y_test)
# print("Restored model accuracy:", acc)
# model.compile(optimizer = 'adam',
#               loss = 'sparse_categorical_crossentropy',
#               metrics = ['accuracy'])
