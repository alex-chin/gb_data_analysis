from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection
import numpy as np

from abc import ABC, abstractmethod


# Абстрактный класс функции потерь
class FuncLoss(ABC):
    @abstractmethod
    def loss(self, real, prediction):
        pass

    @abstractmethod
    def grad(self, x, y):
        pass


# класс реализации градиентного бустинга
class GBGradBoost:
    n_trees = 0
    max_depth = 0
    eta = 0
    coefs = []

    X_train = None
    X_test = None
    y_train = None
    y_test = None
    trees_list = []

    train_errors = []
    test_errors = []

    func_loss = None

    # обучающая выборка с классами и функция потерь - обязательные параметры
    def __init__(self, X, y, func_loss: FuncLoss, n_trees=10, max_depth=3, test_size=0.25, eta=1):

        self.params(n_trees=n_trees, max_depth=max_depth, eta=eta)
        # разделение выборки встроено в класс
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y,
                                                                                                test_size=test_size,
                                                                                                random_state=42)
        self.func_loss = func_loss

    # установка гипер-параметров
    def params(self, n_trees=10, max_depth=3, eta=1):
        self.coefs = [1] * n_trees
        self.trees_list = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.eta = eta

    def gb_predict(self, X):
        # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
        # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании
        # прибавляются с шагом eta
        return np.array(
            [sum([self.eta * coef * alg.predict([x])[0] for alg, coef in zip(self.trees_list, self.coefs)]) for x in X])

    def calc_error(self):
        # расчет ошибки на обучающей и тестовой выборке
        self.train_errors.append(self.func_loss.loss(self.y_train, self.gb_predict(self.X_train)))
        self.test_errors.append(self.func_loss.loss(self.y_test, self.gb_predict(self.X_test)))

    def gb_fit(self):
        # Деревья будем записывать в список
        self.trees_list = []

        # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
        self.train_errors = []
        self.test_errors = []

        for i in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)

            # инициализируем бустинг начальным алгоритмом, возвращающим ноль,
            # поэтому первый алгоритм просто обучаем на выборке и добавляем в список
            if len(self.trees_list) == 0:
                # обучаем первое дерево на обучающей выборке
                tree.fit(self.X_train, self.y_train)

                self.calc_error()
            else:
                # Получим ответы на текущей композиции
                target = self.gb_predict(self.X_train)

                # алгоритмы начиная со второго обучаем на сдвиг
                tree.fit(self.X_train, self.func_loss.grad(self.y_train, target))

                self.calc_error()

            self.trees_list.append(tree)

    def evaluate_alg(self):
        train_prediction = self.gb_predict(self.X_train)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_depth} \
        с шагом {self.eta} на тренировочной выборке: {self.func_loss.loss(self.y_train, train_prediction)}')

        test_prediction = self.gb_predict(self.X_test)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_depth} \
        с шагом {self.eta} на тестовой выборке: {self.func_loss.loss(self.y_test, test_prediction)}')


class FlossMSqured(FuncLoss):
    # mean_squared_error
    def loss(self, real, prediction):
        return (sum((real - prediction) ** 2)) / len(real)

    def grad(self, y, z):
        return (y - z)


from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

gbgboost = GBGradBoost(X, y, FlossMSqured(), max_depth=3, n_trees=10)
gbgboost.gb_fit()
gbgboost.evaluate_alg()
print(gbgboost.train_errors)

# [29866.82175226586, 2964.872977770384, 2618.4394752548965, 2285.1549132329656, 2064.1530414723293, 1914.161272900323,
# 1694.899590521892, 1556.2360696132414, 1486.7167587738113, 1334.4849216740972]

# gbgboost.params(max_depth=4, n_trees=15)
# gbgboost.gb_fit()
# gbgboost.evaluate_alg()
