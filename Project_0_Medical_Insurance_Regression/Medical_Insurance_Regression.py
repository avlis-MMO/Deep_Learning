import pandas as pd
import tensorflow as tf
import keras.api._v2.keras as keras
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

directory_of_python_script = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(directory_of_python_script,'insurance.csv'))

df.head()

# No Null values
df.info()

# Transforms the columns to values between 0 and 1, and
# one hot eocnde categorical columns
ct = make_column_transformer(
                    (MinMaxScaler(), ['age', 'bmi', 'children']), # Turn all values between 0 and 1
                    (OneHotEncoder(handle_unknown="ignore"), ['sex','smoker','region']),
)


# Get the labels and features
X = df.drop('charges', axis=1)
y = df['charges']

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Fit and transform the column transformer to our training data and
# Transform only the test set to avoid leakeage
X_train_normal = ct.fit_transform(X_train)
X_test_normal = ct.transform(X_test)

# Create model
tf.random.set_seed(42)
insurance_model = tf.keras.Sequential([
                            tf.keras.layers.Dense(100, input_shape=(11,), activation = 'relu'),
                            tf.keras.layers.Dense(200, activation = 'relu'),
                            tf.keras.layers.Dense(100, activation = 'relu'),
                            tf.keras.layers.Dense(1)
                            ])

# Compile model
insurance_model.compile(loss= tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
              metrics = ['mae'])

# Fit model
history = insurance_model.fit(X_train_normal, y_train, epochs = 100, verbose = 0)

# Evaluate the results
insurance_model.evaluate(X_test_normal, y_test)

# Plot history
pd.DataFrame(history.history).plot()
plt.ylabel('Loss')
plt.xlabel("Epochs")
plt.show()
