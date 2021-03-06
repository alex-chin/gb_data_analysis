from abc import ABC

import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class FuncDist(ABC):
    def calc(self, x1, x2):
        pass


class GbKnn:
    eta = 0
    qu = 0
    k_neighbours = 0

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    func_dist = None

    def __init__(self, X, y, func_dist: FuncDist, k_neighbours=1, test_size=0.25, eta=1, qu=0.5):
        self.params(k_neighbours=k_neighbours, eta=eta, qu=qu)
        # разделение выборки встроено в класс
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y,
                                                                                                test_size=test_size,
                                                                                                random_state=1)
        self.func_dist = func_dist

    def params(self, k_neighbours=1, eta=1, qu=0.5):
        self.k_neighbours = k_neighbours
        self.eta = eta
        self.qu = qu

    def predict(self):
        answers = []
        for x in self.X_test:
            test_distances = []

            for i in range(len(self.X_train)):
                # расчет расстояния от классифицируемого объекта до
                # объекта обучающей выборки
                distance = self.func_dist.calc(x, self.X_train[i])

                # Записываем в список значение расстояния и ответа на объекте обучающей выборки
                test_distances.append((distance, self.y_train[i]))

            # создаем словарь со всеми возможными классами
            classes = {class_item: 0 for class_item in set(self.y_train)}

            test_distances = self.ordered(test_distances)
            # print(test_distances)

            # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
            for d in test_distances[0:self.k_neighbours]:
                classes[d[1]] += 1

            # Записываем в список ответов наиболее часто встречающийся класс
            answers.append(sorted(classes, key=classes.get)[-1])
        return answers

    def accuracy(self, pred, y):
        return (sum(pred == y) / len(y))

    def quality(self, pred):
        return self.accuracy(pred, self.y_test)

    def ordered(self, distance):
        return sorted(distance)


class EvMetDist(FuncDist):
    def calc(self, x1, x2):
        distance = 0
        for i in range(len(x1)):
            distance += np.square(x1[i] - x2[i])

        return np.sqrt(distance)


class GbKnnWNum(GbKnn):

    def ordered(self, distance):
        w_distance = []
        for i, (d, y) in enumerate(sorted(distance)):
            w_distance.append((d * (1 / (i + 1)), y))
        return sorted(w_distance, reverse=True)


class GbKnnWDist(GbKnn):

    def ordered(self, distance):
        w_distance = []
        for i, (d, y) in enumerate(distance):
            w_distance.append((self.qu ** d, y))
        return sorted(w_distance, reverse=True)


X, y = load_iris(return_X_y=True)
# Для наглядности возьмем только первые два признака (всего в датасете их 4)
X = X[:, :2]

gbknn1 = GbKnn(X, y, EvMetDist(), test_size=0.2)
gbknn2 = GbKnnWNum(X, y, EvMetDist(), test_size=0.2)
gbknn3 = GbKnnWDist(X, y, EvMetDist(), test_size=0.2, qu=0.3)
preds1 = []
preds_wnum = []
preds_wdist = []

for k in range(1, 10):
    gbknn1.params(k_neighbours=k)
    gbknn2.params(k_neighbours=k)
    gbknn3.params(k_neighbours=k)
    preds1.append(gbknn1.quality(gbknn1.predict()))
    preds_wnum.append(gbknn2.quality(gbknn2.predict()))
    preds_wdist.append(gbknn3.quality(gbknn3.predict()))

print(preds1)
print(preds_wnum)
print(preds_wdist)