# Импорт необходимых библиотек
import numpy as np  # Работа с массивамиульны и матрицами
import random  # Для перемешивания данных
from sklearn import datasets  # Наборы данных для тестирования, в данном случае Iris
import matplotlib.pyplot as plt  # Построение графиков

# Загрузка датасета Iris
iris = datasets.load_iris()

# Подготовка датасета: каждая запись содержит данные о цветке (признаки) и его класс
dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

# Константы для нейронной сети
INPUT_DIM = 4  # Размер входного слоя (количество признаков у цветка)
OUT_DIM = 3    # Размер выходного слоя (количество классов)
H_DIM = 10     # Размер скрытого слоя (количество нейронов в этом слое)
loss_arr = []  # Массив для хранения значений ошибки на каждой итерации

# Инициализация весов и смещений для входного и скрытого слоя
W1 = np.random.rand(INPUT_DIM, H_DIM)  # Веса между входным и скрытым слоем
b1 = np.random.rand(1, H_DIM)          # Смещения для скрытого слоя

# Инициализация весов и смещений для скрытого и выходного слоя
W2 = np.random.rand(H_DIM, OUT_DIM)    # Веса между скрытым и выходным слоем
b2 = np.random.rand(1, OUT_DIM)        # Смещения для выходного слоя

# Нормализация весов для лучшей сходимости
W1 = (W1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
b1 = (b1 - 0.5) * 2 * np.sqrt(1 / INPUT_DIM)
W2 = (W2 - 0.5) * 2 * np.sqrt(1 / H_DIM)
b2 = (b2 - 0.5) * 2 * np.sqrt(1 / H_DIM)

# Гиперпараметры обучения
NUM_EPOCHS = 400  # Количество эпох обучения
ALPHA = 0.001     # Скорость обучения
BATCH_SIZE = 50   # Размер мини-выборки

# Активационная функция ReLU: заменяет все отрицательные значения на 0
def relu(t):
    return np.maximum(t, 0)

# Функция Softmax: преобразует выходы в вероятности
def softmax_batch(t):
    out = np.exp(t)  # Экспоненцируем все значения
    return out / np.sum(out, axis=1, keepdims=True)  # Нормализуем, чтобы сумма равнялась 1

# Функция вычисления потерь для одного объекта
def sparse_cross_entropy(z, y):
    return -np.log(z[0, y])

# Функция вычисления потерь для батча
def sparse_cross_entropy_batch(z, y):
    return -np.log(np.array([z[j, y[j]] for j in range(len(y))]))

# Преобразование меток классов в формат "one-hot encoding"
def to_full_batch(y, num_classes):
    y_full = np.zeros((len(y), num_classes))
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full

# Производная ReLU
def relu_deriv(t):
    return (t >= 0).astype(float)

# Цикл обучения нейронной сети
for ep in range(NUM_EPOCHS):
    random.shuffle(dataset)  # Перемешиваем данные перед каждой эпохой
    for i in range(len(dataset) // BATCH_SIZE):
        # Формируем батч данных
        batch_x, batch_y = zip(*dataset[i * BATCH_SIZE: i * BATCH_SIZE + BATCH_SIZE])
        x = np.concatenate(batch_x, axis=0)  # Входные данные
        y = np.array(batch_y)  # Метки классов

        # Прямое распространение (forward pass)
        t1 = x @ W1 + b1       # Линейная комбинация входов и весов
        h1 = relu(t1)          # Применяем активацию ReLU
        t2 = h1 @ W2 + b2      # Линейная комбинация выходов скрытого слоя
        z = softmax_batch(t2)  # Преобразуем выходы в вероятности
        E = np.sum(sparse_cross_entropy_batch(z, y))  # Вычисляем ошибку

        # Обратное распространение (backpropagation)
        y_full = to_full_batch(y, OUT_DIM)  # Преобразуем метки в "one-hot encoding"
        dE_dt2 = z - y_full                 # Градиент ошибки на выходном слое
        dE_dW2 = h1.T @ dE_dt2             # Градиент весов между скрытым и выходным слоем
        dE_db2 = np.sum(dE_dt2, axis=0, keepdims=True)  # Градиент смещений на выходном слое
        dE_dh1 = dE_dt2 @ W2.T             # Градиент активаций скрытого слоя
        dE_dt1 = dE_dh1 * relu_deriv(t1)   # Градиент скрытого слоя (с учетом ReLU)
        dE_dW1 = x.T @ dE_dt1              # Градиент весов между входным и скрытым слоем
        dE_db1 = np.sum(dE_dt1, axis=0, keepdims=True)  # Градиент смещений на скрытом слое

        # Обновление весов и смещений
        W1 -= ALPHA * dE_dW1
        b1 -= ALPHA * dE_db1
        W2 -= ALPHA * dE_dW2
        b2 -= ALPHA * dE_db2

        # Сохраняем значение ошибки
        loss_arr.append(E)

# Функция для предсказания класса
def predict(x):
    t1 = x @ W1 + b1       # Линейная комбинация входов и весов
    h1 = relu(t1)          # Применяем ReLU
    t2 = h1 @ W2 + b2      # Линейная комбинация выходов скрытого слоя
    z = softmax_batch(t2)  # Преобразуем выходы в вероятности
    return z

# Функция вычисления точности на всём датасете
def calc_accuracy():
    correct = 0
    for x, y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)  # Предсказанный класс
        if y_pred == y:
            correct += 1
    acc = correct / len(dataset)  # Доля правильных ответов
    return acc

# Вывод точности модели
accuracy = calc_accuracy()
print("Accuracy - ", accuracy)

# Классификация цветка на основе заданных признаков
def classify_flower(features):
    features = np.array(features).reshape(1, -1)  # Преобразуем вектора в нужный формат
    probabilities = predict(features)            # Получаем вероятности классов
    predicted_class = np.argmax(probabilities)   # Определяем класс с максимальной вероятностью
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    probabilities_percent = (probabilities[0] * 100).round(2)  # Проценты
    return class_names[predicted_class], probabilities_percent

# Примеры использования
setosa_features = [5.0, 3.4, 1.5, 0.2]
versicolor_features = [6.0, 2.8, 4.5, 1.3]
virginica_features = [6.5, 3.0, 5.5, 2.0]

# Классификация для примера
predicted_class, probabilities = classify_flower(setosa_features)

# Вывод результатов классификации
print(f"Признаки: {setosa_features}")
print(f"Предсказанный класс: {predicted_class}")
print(f"Вероятности классов: {probabilities}")

# Построение графика ошибки
plt.plot(loss_arr)
plt.title("График ошибки")
plt.xlabel("Итерация")
plt.ylabel("Ошибка")
plt.show()