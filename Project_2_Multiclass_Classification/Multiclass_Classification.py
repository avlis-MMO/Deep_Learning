import tensorflow as tf
import pandas as pd
import numpy as np
import keras.api._v2.keras as keras
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random, itertools
from sklearn.metrics import confusion_matrix

# Prittify cm
def prittify_CM(cm):
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    n_classes = cm.shape[0]

    # Lets prittify
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Create classes
    classes = False
    if classes:
        labels = classes
    else:
        labels = np.arange(cm.shape[0])

    # Label the axes
    ax.set(title='Confusion Matrix', xlabel = 'Predictd Label', ylabel = 'True Label',
        xticks = np.arange(n_classes), yticks= np.arange(n_classes), xticklabels = labels,
        yticklabels = labels)
    
    # Set x-axis to bottom
    ax.xaxis.set_label_position("bottom")
    ax.xaxis.tick_bottom()

    # Adjust label size
    ax.yaxis.label.set_size(20)
    ax.xaxis.label.set_size(20)
    ax.title.set_size(20)

    # Set threshold for different colors
    threshold = (cm.max() + cm.min())/2

    # Plot the text on each cell
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f" {cm[i, j]} \n ({cm_norm[i,j]*100:.1f}%)", 
                horizontalalignment="center", verticalalignment = "center", color="white" if cm[i,j] > threshold else "black",
                size = 9)
    plt.show()

# 1. Import data
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# 2. Check data
print(f"Training smaple:{train_data[0]}\n")
print(f"Training smaple:{train_labels[0]}\n")
print(train_data.max(), train_data.min())
# Check shape adn sample
print(train_data[0].shape)
plt.imshow(train_data[0])
plt.show()

# Create a small list to index our training labels
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Plot multiple random image of dataset
plt.figure(figsize=(7,7))
for i in range(4):
    ax = plt.subplot(2,2, i +1)
    rand_index = random.choice(range(len(train_data)))
    plt.imshow(train_data[rand_index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[rand_index]])
    plt.axis(False)
plt.show()

# 3. Building model
# Set seed
tf.random.set_seed(42)

# Normalize data
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# Create model
model = tf.keras.Sequential([
                            tf.keras.layers.Flatten(input_shape = (28,28)),
                            tf.keras.layers.Dense(100, activation='tanh'),
                            tf.keras.layers.Dense(75, activation='tanh'),
                            tf.keras.layers.Dense(50, activation='tanh'),
                            tf.keras.layers.Dense(10, activation = 'softmax')
])

# Compile model
model.compile(loss = "sparse_categorical_crossentropy",
             optimizer = tf.keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

# Find best learning rate
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 **(epoch/20))


# Fit model
norm_history = model.fit(train_data_norm, train_labels, epochs = 50, validation_data=(test_data_norm, test_labels))

# Plot the lr decay curve
#lrs = 1e-3  * (10 **(tf.range(40)/20))
#plt.semilogx(lrs, norm_history.history["loss"]) # After finding best lr change it on the model
#plt.xlabel("learning rate")
#plt.ylabel("loss")
#plt.show()

# Get the probabilities of each label
y_prob = model.predict(test_data_norm)
print(y_prob)
#Conver prob to label
pred_label = tf.argmax(y_prob, axis=1)
print(pred_label)
cm = confusion_matrix(test_labels, pred_label)
prittify_CM(cm)     
