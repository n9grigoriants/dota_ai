from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np


class PrecisionThresholded(tf.keras.metrics.Precision):
    def __init__(self, threshold=0.8, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def update_state(self, y_true, y_pred, sample_weight=None):
        thresholded_preds = tf.cast(tf.greater_equal(y_pred, self.threshold), tf.float32)
        return super().update_state(y_true, thresholded_preds, sample_weight)


class AccuracyThresholded(tf.keras.metrics.BinaryAccuracy):
    def __init__(self, threshold=0.8, **kwargs):
        super().__init__(threshold=threshold, **kwargs)


custom_objects = {
    'PrecisionThresholded': PrecisionThresholded,
    'AccuracyThresholded': AccuracyThresholded
}


loaded_model = load_model('model_3.h5', custom_objects=custom_objects)
# Подготовка одного образца данных
X_single = [[0, 81], [1, 64], [1, 16], [1, 96], [1, 30], [-1, 13], [-1, 51], [-1, 37], [-1, 80], [-1, 10]]  # Ваш одиночный образец данных

X_single_1 = np.array([vec[0] for vec in X_single]).reshape(1, -1)  # Добавляем дополнительное измерение для пакета
X_single_2 = np.array([vec[1] for vec in X_single]).reshape(1, -1)  # Добавляем дополнительное измерение для пакета

# Получение предсказаний
y_pred = loaded_model.predict([X_single_1, X_single_2])

# y_pred будет содержать вероятности для каждого класса.
# Если вы хотите получить метку класса, используйте argmax:
y_pred_label = np.argmax(y_pred, axis=1)

sorted_indices = np.argsort(y_pred[0])

# Взятие последних 5 индексов для топ-5 классов
top_10_indices = sorted_indices[-10:][::-1]  # [::-1] для инвертирования порядка, чтобы начать с наибольшего
anti_top_10_indices = sorted_indices[:10][::1]

print("Top 10 predicted classes:", top_10_indices)
print("Top 10 probabilities:", y_pred[0][top_10_indices])
print("Anti-Top 10 predicted classes:", anti_top_10_indices)
print("Anti-Top 10 probabilities:", y_pred[0][anti_top_10_indices])
