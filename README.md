# Classification of Handwritten Digits using Neural Networks

## Overview

This project involves building and evaluating a neural network model for handwritten digit recognition using the MNIST dataset. The goal is to classify images of handwritten digits (0-9) into their corresponding digit classes.

A neural network which is a supervised machine learning algorithm, which learns from experience in terms of hierarchy of representations. Each level of the hierarchy corresponds to a layer of the model. 

## Dataset

The MNIST dataset is a collection of handwritten digits, commonly used for training various image processing systems. The dataset contains:
- **Training Set:** 60,000 images of handwritten digits.
- **Test Set:** 10,000 images of handwritten digits.

## Code

### Importing necessary libraries

We first load all the required libraries and set seed as 1999 to ensure reproducibility.

```python
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import ConfusionMatrixDisplay


seed = 1999
np.random.seed(seed)
tf.random.set_seed(seed)
```

### Data Loading and Pre-processing

```python
# Loading the dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```

Now this would be the raw images in the dataset, and in order for our model to be efficient, it is good practice to normalise our data. Therefore,we preprocess the data to help improve the convergence speed of the training process and the overall performance of the model.
We also use One-hot encoding to convert the class labels.

```python
# normalize to range 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

#train_labels = train_labels
#test_labels = test_labels

# Hot encoding
train_labels_h = to_categorical(train_labels)
test_labels_h = to_categorical(test_labels)

print(train_images.shape)
print(train_labels_h.shape)
print(test_images.shape)
print(test_labels_h.shape)
```

We can check how the images in the MNIST dataset look, and how the one-hot encoding has changes the class labels:

```python
import matplotlib.pyplot as plt
plt.imshow(train_images[0], cmap = 'gray')

print(train_labels[0])
print(train_labels_h[0])
```

```python
train_lab = to_categorical(train_labels_h[0])
print(train_lab)
```

### Model Building and Model Compilation

Our neural network will have one Flatten layer and two dense layers. The flattening layer converts our 2D array of 28 by 28 pixel images to a 1D vector of length 784, so it can be passed onto the Dense layers.
The first dense layer is a fully connected layer with 128 neurons that applies the ReLU activation function, learning patterns from the flattened input of 784 length vector.
The second dense layer is the output layer  with 10 neurons (one for each class, since our classification task has 10 classes) using the softmax function to output probabilities that sum to 1, representing the likelihood of each class.

```python
# Function to build and compile the model
def build_model(optimizer):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train and evaluate the model
def train_and_evaluate(optimizer_name, optimizer):
    # Reset TensorFlow session
    tf.keras.backend.clear_session()
    
    model = build_model(optimizer)
    model.fit(train_images, train_labels_h, validation_data=(test_images, test_labels_h), epochs=25, batch_size=32, verbose=0)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images, test_labels_h, verbose=0)
    test_predicted = model.predict(test_images, verbose=0)
    test_predicted_classes = np.argmax(test_predicted, axis=1)
    
    precision = precision_score(test_labels, test_predicted_classes, average='macro')
    recall = recall_score(test_labels, test_predicted_classes, average='macro')
    f1 = f1_score(test_labels, test_predicted_classes, average='macro')
    accuracy = accuracy_score(test_labels, test_predicted_classes)
    
    return {
        'Optimizer': optimizer_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1,
        'Predictions': test_predicted_classes  # To be used for confusion matrix plotting
    }

# Define optimizers
optimizers = {
    'Adam': tf.keras.optimizers.Adam(learning_rate=0.001),
    'SGD': tf.keras.optimizers.SGD(learning_rate=0.001),
    'RMSprop': tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Adagrad': tf.keras.optimizers.Adagrad(learning_rate=0.001)
}

# Evaluate all optimizers
results = []
predictions = {}
for name, opt in optimizers.items():
    result = train_and_evaluate(name, opt)
    results.append({
        'Optimizer': name,
        'Accuracy': result['Accuracy'],
        'Precision': result['Precision'],
        'Recall': result['Recall'],
        'F1_score': result['F1_score']
    })
    predictions[name] = result['Predictions']

results_df = pd.DataFrame(results)
```

The loss function we use here is the *categorical cross-entropy function*. Categorical cross-entropy is a critical loss function for training neural networks in multi-class classification problems. It quantifies how far off the model's predictions are from the actual labels by comparing the predicted probability distribution to the true label's one-hot encoded vector. The goal during training is to minimize this loss, thereby improving the model's accuracy.

The evaluation metric we are using on the model is accuracy. Although we are calculating other metrics like precision, recall, F1-score etc., the model will be mainly evaluated based on the accuracy metric, and the other metrics will be able to provide more nuanced insights into the model's performance.

The model is trained for 25 epochs with a batch size of 32.

Four optimizers are used here: Adam, SGD, Adagrad, RMSProp.

Although true postiives and false negatives are important metrics in classification, we will not be calculating them here, as ours is a multi-class classification. This makes it harder to define what, say, a true negative is, as opposed to binary classification. Metrics such as recall and precision, which use true positives, can still be calculated, by finding TP,TN,FP,FN for each class, and then taking the average. This is another reason why accuracy is primarily chosen as an evaluation metric here, because of the simplicity of the definition (correctly classified images/total images).

The results are stored in a dataframe.

```python
# Print model summary
model.summary()
```

This just provides us with a brief overview of our model, with information on the layer, output shape and number of paramters in each layer.

Now we can print the results here:

```python
print(results_df)
```

Let us determine the best performing and the worst performing optimizer based on accruacy as the metric:

```python
# Determine the best and worst optimizer based on accuracy
best_optimizer = results_df.loc[results_df['Accuracy'].idxmax()]
worst_optimizer = results_df.loc[results_df['Accuracy'].idxmin()]

# Print the best and worst optimizer details
print("\nBest Optimizer:")
print(best_optimizer)

print("\nWorst Optimizer:")
print(worst_optimizer)
```

We are filtering our results of the four optimizers, to only include the best and worst performing optimizer. We then produce a bar graph that shows the precision, accuracy, recall, and F1-score for Adam and SGD optimizers (best and worst).

```python
# Filter results to include only the best and worst optimizers
filtered_results_df = results_df[results_df['Optimizer'].isin([best_optimizer['Optimizer'], worst_optimizer['Optimizer']])]

# Prepare data for plotting
metrics_df = pd.melt(filtered_results_df, id_vars='Optimizer', var_name='metric', value_name='value')

# Define custom color palette
custom_palette = {
    best_optimizer['Optimizer']: '#BBF90F',  # Parrot Green
    worst_optimizer['Optimizer']: '#7BC8F6'  # Dusty Blue
}

# Plotting the results
plt.figure(figsize=(12, 7))
sns.barplot(x='metric', y='value', hue='Optimizer', data=metrics_df, palette=custom_palette)
plt.title('Comparison of Metrics for Best and Worst Optimizers')
plt.ylabel('Metric Value')
plt.xlabel('Metric')
plt.tight_layout()
plt.show()
```

One reason Adam might have outperformed SGD optimizer is that Adam automatically adjust the learning rate for each parameter during training, reducing the need for manual tuning. SGD uses a fixed global learning rate, which can be challenging to tune. The same learning rate is applied to all parameters, which may not be optimal for parameters that have different gradient magnitudes.

### Visualizing the Data

Let us display the first 25 images from the training set:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])
plt.show()
```

### Confusion Matrices

The confusion matrix provides a detailed view of how well the model's predictions match the actual labels, allowing us to see which digits are most frequently confused.
Let us show confusion matrices for the best and worst optimizers:

```python
# Plot confusion matrix for the best optimizer
best_optimizer_preds = predictions[best_optimizer['Optimizer']]
cm_best = confusion_matrix(test_labels, best_optimizer_preds)
disp_best = ConfusionMatrixDisplay(confusion_matrix=cm_best, display_labels=np.arange(10))
plt.figure(figsize=(10, 8))
disp_best.plot(cmap='Greens')
plt.title(f'Confusion Matrix for {best_optimizer["Optimizer"]}')
plt.show()
```

```python
# Plot confusion matrix for the worst optimizer
worst_optimizer_preds = predictions[worst_optimizer['Optimizer']]
cm_worst = confusion_matrix(test_labels, worst_optimizer_preds)
disp_worst = ConfusionMatrixDisplay(confusion_matrix=cm_worst, display_labels=np.arange(10))
plt.figure(figsize=(10, 8))
disp_worst.plot(cmap='Greens')
plt.title(f'Confusion Matrix for {worst_optimizer["Optimizer"]}')
plt.show()
```

### Misclassification

Since our accuracy is not 100%, we know that images have been misclassified in our model. Let us see how the misclassified images look for the best and worst optimizers:

```python
# Function to plot misclassifications
def plot_misclassifications(images, true_labels, predicted_labels, num_examples=10):
    misclassified_indices = np.where(true_labels != predicted_labels)[0]
    if len(misclassified_indices) == 0:
        print("No misclassifications found.")
        return
    
    # Randomly sample misclassified indices
    sample_indices = np.random.choice(misclassified_indices, num_examples, replace=False)
    
    plt.figure(figsize=(12, 12))
    for i, index in enumerate(sample_indices):
        plt.subplot(5, 5, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[index], cmap=plt.cm.binary)
        plt.xlabel(f'True: {true_labels[index]}, Pred: {predicted_labels[index]}')
    plt.tight_layout()
    plt.show()
```

```python
# Plot misclassifications for the best optimizer
plot_misclassifications(test_images, test_labels, best_optimizer_preds)
```

```python
# Plot misclassifications for the worst optimizer
plot_misclassifications(test_images, test_labels, worst_optimizer_preds)
```
