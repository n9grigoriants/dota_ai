import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.layers import Embedding


with open("matrices.pkl", "rb") as f:
    X = pickle.load(f)

with open("items_lists.pkl", "rb") as f:
    y = pickle.load(f)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_test, y_test)


model = Sequential([
    Input(shape=(10, 6)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(96, activation='relu'),
    Dense(64, activation='relu'),
    Dense(172, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('\nTest accuracy:', test_acc)