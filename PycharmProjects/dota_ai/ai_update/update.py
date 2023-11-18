import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

# Загрузите новые данные
with open("new_matrices.pkl", "rb") as f:
    X_new = pickle.load(f)

with open("new_items_lists.pkl", "rb") as f:
    y_new = pickle.load(f)

X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.1)

# разрубание массива на 1 и 2 элементы (принадлежность к команде и ID героя) для train
X_train_1_new = np.array([[vec[0] for vec in sequence] for sequence in X_train_new])
X_train_2_new = np.array([[vec[1] for vec in sequence] for sequence in X_train_new])

# разрубание массива на 1 и 2 элементы (принадлежность к команде и ID героя) для test
X_test_1_new = np.array([[vec[0] for vec in sequence] for sequence in X_test_new])
X_test_2_new = np.array([[vec[1] for vec in sequence] for sequence in X_test_new])

# Преобразование y_train и y_test в numpy массивы
y_train_new = np.array(y_train_new)
y_test_new = np.array(y_test_new)


# Загрузите вашу предварительно обученную модель
model = load_model('model_2.h5')

# Продолжите обучение модели с новыми данными
history = model.fit([X_train_1_new, X_train_2_new], y_train_new, epochs=10, validation_data=([X_test_1_new, X_test_2_new], y_test_new))

# Оцените модель с новыми данными, если необходимо
test_loss_new, test_precision_new = model.evaluate([X_test_1_new, X_test_2_new], y_test_new)
print('\nNew Test precision:', test_precision_new)

# Сохраните модель после дополнительного обучения, если необходимо
model.save('model_2.h5')