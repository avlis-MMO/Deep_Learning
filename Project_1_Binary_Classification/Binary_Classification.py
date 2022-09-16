from types import new_class
import tensorflow as tf
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

def plot_decision_boundary(model, X, y):

    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    y_min, y_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, x_max, 100))
    
    #Create X value 
    x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together

    # Make preds
    y_pred = model.predict(x_in)

    # Check for multi-class
    if  len(y_pred[0]) > 1:
        print("doing multiclass classification")
        # We have to rehsape our preds
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("Doing binary classification")
        y_pred = np.round(y_pred.reshape(xx.shape))
    
    # Plot the decision boundary
    plt.contourf(xx, yy, y_pred, alpha = 0.7, cmap =plt.cm.RdYlBu)
    plt.scatter(X[:,0], X[:,1], c=y, s=25, cmap =plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
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
        plt.text(j, i, f"{cm[i, j]} ({cm_norm[i,j]*100:.1f}%)", 
                horizontalalignment="center", color="white" if cm[i,j] > threshold else "black",
                size = 15)
    plt.show()
# Make 1000 examples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

circles = pd.DataFrame({'X0':X[:,0], 'X1': X[:,1], 'labels':y})

# Visualize data
plt.scatter(x=circles['X0'], y = circles['X1'], c=y, cmap =plt.cm.RdYlBu)
plt.show()

# Check the shapes
print(X.shape, y.shape)

# How many samples
print(len(X), len(y))

# Split the data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
# Create the model
tf.random.set_seed(42)
model_1 = tf.keras.Sequential([
                    tf.keras.layers.Dense(4, input_shape = (2,), activation = 'tanh'),
                    tf.keras.layers.Dense(4, activation = 'tanh'),
                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                            ])

# Compile model
model_1.compile(loss = "binary_crossentropy",
                optimizer = "Adam",
                metrics=['accuracy'])

tf.random.set_seed(42)
model_2 = tf.keras.Sequential([
                    tf.keras.layers.Dense(4, input_shape = (2,), activation = 'tanh'),
                    tf.keras.layers.Dense(4, activation = 'tanh'),
                    tf.keras.layers.Dense(1, activation = 'sigmoid')
                            ])

# Compile model
model_2.compile(loss = "binary_crossentropy",
                optimizer = tf.keras.optimizers.Adam(0.05),
                metrics=['accuracy'])

## Finding the best lr
# Create learning callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10 **(epoch/20))

# Fit model
history = model_1.fit(X_train, y_train, epochs=100, callbacks = [lr_scheduler], verbose = 0)
model_2.fit(X_train, y_train, epochs=50, verbose = 0)
# Visualize model preds
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_2, X_train, y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_2, X_test, y_test)
plt.show()

# Evaluate model
model_2.evaluate(X_test,y_test)

# Check confusion matrix
y_pred = model_2.predict(X_test)
cm = confusion_matrix(y_test, tf.round(y_pred))
prittify_CM(cm)

# Convert the history into a dataframe
pd.DataFrame(history.history).plot()
plt.title("Model curves")
plt.show()
lrs = 1e-4 * 10**(tf.range(100)/20)
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("learning rate") # After getting best learning lets create new model
plt.show()
