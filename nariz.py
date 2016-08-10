import tensorflow as tf
import numpy as np

# Data sets
TRAINING = "train.csv"
TEST = "test.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv(filename=TRAINING, target_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv(filename=TEST, target_dtype=np.int)

x_train, x_test, y_train, y_test = training_set.data, test_set.data, \
  training_set.target, test_set.target

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=3)

# Fit model.
classifier.fit(x=x_train, y=y_train, steps=200)

# Evaluate accuracy.
accuracy_score = classifier.evaluate(x=x_test, y=y_test)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))

# Classify two new fruits.
new_samples = np.array(
    [[180, 260, 121, 172, 82], [199, 176, 102, 263, 115]], dtype=float)
y = classifier.predict(new_samples)
print ('Predictions: {}'.format(str(y)))
