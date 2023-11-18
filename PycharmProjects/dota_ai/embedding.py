import numpy as np
import pickle
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Embedding, Concatenate, Reshape
from tensorflow.keras import Model


with open("matrices.pkl", "rb") as f:
    X = pickle.load(f)

with open("items_lists.pkl", "rb") as f:
    y = pickle.load(f)

with open("heroes.json", "r") as f:
    heroes_data = json.load(f)


hero_to_index = {hero: idx for idx, hero in enumerate(heroes_data.keys())}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


X_train = np.array(X_train)
X_test = np.array(X_test)


X_train_first_param = X_train[:, :, 0]
X_test_first_param = X_test[:, :, 0]


X_train_rest_params = X_train[:, :, 1:]
X_test_rest_params = X_test[:, :, 1:]


X_train_indices = np.round(X_train_rest_params * 1000).astype(int)
X_test_indices = np.round(X_test_rest_params * 1000).astype(int)

y_train = np.array(y_train)
y_test = np.array(y_test)


input_first_param = Input(shape=(10, 1))
input_rest_params = Input(shape=(10, 5))


embedding_layer = Embedding(input_dim=1001, output_dim=5, name="hero_embedding")(input_rest_params)

reshaped_embedding = Reshape((10, 5 * 5))(embedding_layer)

concatenated = Concatenate(axis=-1)([input_first_param, reshaped_embedding])

flattened = Flatten()(concatenated)
dense1 = Dense(128, activation='relu')(flattened)
dense2 = Dense(96, activation='relu')(dense1)
dense3 = Dense(64, activation='relu')(dense2)
output = Dense(172, activation='softmax')(dense3)

model = Model(inputs=[input_first_param, input_rest_params], outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit([X_train_first_param, X_train_indices], y_train, epochs=10, validation_data=([X_test_first_param, X_test_indices], y_test))

test_loss, test_acc = model.evaluate([X_test_first_param, X_test_indices], y_test)
print('\nTest accuracy:', test_acc)


embedding_weights = model.get_layer("hero_embedding").get_weights()[0]

# Применение функции sigmoid
sigmoid_weights = 1 / (1 + np.exp(-embedding_weights))

# Обновление heroes.json
for hero, index in hero_to_index.items():
    heroes_data[hero] = sigmoid_weights[index].tolist()

with open("heroes.json", "w") as f:
    json.dump(heroes_data, f, indent=4)