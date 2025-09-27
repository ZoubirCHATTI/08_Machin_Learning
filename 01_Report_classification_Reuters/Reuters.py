#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import reuters
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

#Loading data from Keras
(train_data, train_labels),(test_data, test_labels)=reuters.load_data(num_words=10000)

#Visual inspection of data
counts=np.bincount(train_labels)
plt.bar(range(len(counts)), counts/np.sum(counts))
plt.xlabel('class index')
plt.xticks(np.arange(1, 47, 2 ))
plt.ylabel('frequency')
plt.yticks(np.arange(0, 0.45, 0.05))
plt.title('Frequency of presence of each class')
plt.grid(True)
plt.show()

#Extracting words indices (given the word, we get the index)
word_index=reuters.get_word_index()
print(word_index['good'])

#Getting word from the index (given the index, we get the word associated)
reverse_word_index=dict([(value, key) for (key, value) in word_index.items()])

#A function that decodes every index in the data to get the review
def decode_review(index, sequence):
    return " ".join([reverse_word_index.get(i-3, "?") for i in sequence[index]])

#Testing the function on review 0 and 1 of train data
review_0=decode_review(0, train_data)
review_1=decode_review(1, train_data)

print(review_0)
print(review_1)

#Transforming data (input) to a binary matrix
def vectorize_sequence(sequence, dimensions):
    results=np.zeros((len(sequence), dimensions))
    for i, seq in enumerate(sequence):
        for j in seq:
            results[i, j]=1
    return results

#Transforming the output to a binary matrix
def to_one_hot(sequence, dimensions):
    results = np.zeros((len(sequence), dimensions))
    for i, seq in enumerate(sequence):
        results[i, seq] = 1
    return results

#Vectorising inputs
X_train=vectorize_sequence(train_data, 10000)
X_test=vectorize_sequence(test_data, 10000)

#vectorising outputs
Y_train=to_one_hot(train_labels, 46)
Y_test=to_one_hot(test_labels, 46)

#Creation of a DNN model of training
model=Sequential()
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(46, activation='softmax'))

#Compiling the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Condition of stop (if validation loss increases 4 consicutive epochs, the training stops)
early_stop=EarlyStopping(monitor='val_loss', patience=4)

#Running the training
history=model.fit(X_train, Y_train, validation_split=0.2, callbacks=[early_stop], epochs=20)

#Evaluation on the testing dataset
loss, accuracy=model.evaluate(X_test, Y_test)
print(f"loss={loss}\naccuracy={accuracy}")

history_dict=history.history

training_loss=history_dict['loss']
training_accuracy=history_dict['accuracy']

validation_loss=history_dict['val_loss']
validation_accuracy=history_dict['val_accuracy']

epochs=range(1, len(training_loss)+1)

#Plotting loss and accuracy against epochs
plt.subplot(1, 2, 1)
plt.plot(epochs, training_loss, label='Training Loss', color='r', marker='o')
plt.plot(epochs, validation_loss, label='Validation Loss', color='g', marker='+')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(epochs, training_accuracy, label='Training Accuracy', color='b', marker='o')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy', color='g', marker='+')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#Comparing with the baseline
import copy

#Comparing model robustess with random prediciton
test_labels_copy=copy.copy(test_labels)
np.random.shuffle(test_labels_copy)

compare=np.array(test_labels_copy)==np.array(test_labels)
print(f"random accuracy={compare.mean()}\nModel accuracy={accuracy} ")





