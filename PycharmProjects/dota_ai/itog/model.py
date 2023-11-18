import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Input, Embedding, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.metrics import Recall


# загрузка данных
with open("../matrices.pkl", "rb") as f:
    X = pickle.load(f)

with open("../items_lists.pkl", "rb") as f:
    y = pickle.load(f)


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = Precision(name='precision')
        self.recall = Recall(name='recall')
        self.f1_score = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        p = self.precision.result()
        r = self.recall.result()
        self.f1_score.assign(2 * ((p * r) / (p + r + tf.keras.backend.epsilon())))

    def result(self):
        return self.f1_score

    def reset_states(self):
        # сбрасываем состояние метрик precision и recall
        self.precision.reset_states()
        self.recall.reset_states()
        self.f1_score.assign(0.0)


class PrecisionThresholded(tf.keras.metrics.Precision):
    def __init__(self, threshold=0.9, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        thresholded_preds = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
        return super().update_state(y_true, thresholded_preds, sample_weight)


class AccuracyThresholded(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, threshold=0.8, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


# разделение X и y на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# разрубание массива на 1 и 2 элементы (принадлежность к команде и ID героя) для train
X_train_1 = np.array([[vec[0] for vec in sequence] for sequence in X_train])
X_train_2 = np.array([[vec[1] for vec in sequence] for sequence in X_train])

# разрубание массива на 1 и 2 элементы (принадлежность к команде и ID героя) для test
X_test_1 = np.array([[vec[0] for vec in sequence] for sequence in X_test])
X_test_2 = np.array([[vec[1] for vec in sequence] for sequence in X_test])


y_train = np.array(y_train)
y_test = np.array(y_test)


# находим максимальное значение в X_train_2, чтобы определить размерность входных данных для слоя Embedding
max_index = np.max(X_train_2)

# определение входных слоев модели
input_1 = Input(shape=(10,))
input_2 = Input(shape=(10,))

# создание слоя вложения для преобразования идентификаторов из input_2 в векторы
embedding_layer = Embedding(input_dim=max_index + 1, output_dim=64)(input_2)
embedding_flat = Flatten()(embedding_layer)

# объединение выхода из input_1 и слоя вложения
merged = Concatenate()([input_1, embedding_flat])

optimizer = Adam(learning_rate=0.001)

# определение ПОЛНОСВЯЗНЫХ слоев модели
dense1 = Dense(1024, activation='relu')(merged)
dropout1 = Dropout(0.3)(dense1)  # добавляем dropout после активации и регуляризации
dense2 = Dense(512, activation='relu')(dropout1)
dropout2 = Dropout(0.3)(dense2)
dense3 = Dense(512, activation='relu')(dropout2)
dropout3 = Dropout(0.3)(dense3)
dense4 = Dense(256, activation='relu')(dropout3)
dropout4 = Dropout(0.3)(dense4)
output = Dense(183, activation='sigmoid')(dropout4)

# создание модели с двумя входами и одним выходом
model = Model(inputs=[input_1, input_2], outputs=output)


# компиляция модели
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=[Precision(name='precision'),
                       PrecisionThresholded(name='precision_0.8'),
                       'accuracy',
                       AccuracyThresholded(name='accuracy_0.8')])

# выводим архитектуру модели
model.summary()



# обучение модели
history = model.fit([X_train_1, X_train_2], y_train, epochs=20, validation_data=([X_test_1, X_test_2], y_test), batch_size=128)

# оценка модели
test_loss, test_precision, test_precision_0_8, test_accuracy, test_accuracy_0_8 = model.evaluate([X_test_1, X_test_2], y_test)
print('\nTest precision score:', test_precision)
print('Test precision with threshold 0.8 score:', test_precision_0_8)
print('Test accuracy score:', test_accuracy)
print('Test accuracy with threshold 0.8 score:', test_accuracy_0_8)

# сохранение модели
model.save('model_3.h5')