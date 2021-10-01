from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection
import numpy as np


# %%

class GBGradBoost:
    n_trees = 0
    max_depth = 0
    eta = 0
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    trees_list = []

    def __init__(self, X, y, n_trees=10, max_depth=3, test_size=0.25, eta=1):
        self.params(n_trees=n_trees, max_depth=max_depth, eta=eta)
        # разделение выборки встроено в класс
        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(X, y,
                                                                                                test_size=test_size)

    def params(self, n_trees=10, max_depth=3, eta=1):
        self.trees_list = []
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.eta = eta

    def gb_predict(self, X, coef_list):
        # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
        # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании прибавляются с шагом eta
        return np.array(
            [sum([self.eta * coef * alg.predict([x])[0] for alg, coef in zip(self.trees_list, coef_list)]) for x in X])

    def mean_squared_error(self, y_real, prediction):
        return (sum((y_real - prediction) ** 2)) / len(y_real)

    def bias(self, y, z):
        return (y - z)

    # функция обучения градиентного бустинга.

    def gb_fit(self, coefs):
        # Деревья будем записывать в список
        self.trees_list = []

        # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
        train_errors = []
        test_errors = []

        for i in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)

            # инициализируем бустинг начальным алгоритмом, возвращающим ноль,
            # поэтому первый алгоритм просто обучаем на выборке и добавляем в список
            if len(self.trees_list) == 0:
                # обучаем первое дерево на обучающей выборке
                tree.fit(self.X_train, self.y_train)

                train_errors.append(self.mean_squared_error(self.y_train, self.gb_predict(self.X_train, coefs)))
                test_errors.append(self.mean_squared_error(self.y_test, self.gb_predict(self.X_test, coefs)))
            else:
                # Получим ответы на текущей композиции
                target = self.gb_predict(self.X_train, coefs)

                # алгоритмы начиная со второго обучаем на сдвиг
                tree.fit(self.X_train, self.bias(self.y_train, target))

                train_errors.append(self.mean_squared_error(self.y_train, self.gb_predict(self.X_train, coefs)))
                test_errors.append(self.mean_squared_error(self.y_test, self.gb_predict(self.X_test, coefs)))

            self.trees_list.append(tree)

        return train_errors, test_errors

    def evaluate_alg(self, coefs):
        train_prediction = self.gb_predict(self.X_train, coefs)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_depth} \
        с шагом {self.eta} на тренировочной выборке: {self.mean_squared_error(self.y_train, train_prediction)}')

        test_prediction = self.gb_predict(self.X_test, coefs)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_depth} \
        с шагом {self.eta} на тестовой выборке: {self.mean_squared_error(self.y_test, test_prediction)}')


from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
#
# X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
# # Число деревьев в ансамбле
# для простоты примем коэффициенты равными 1
coefs = [1] * 10
# Максимальная глубина деревьев
# Шаг

gbgboost = GBGradBoost(X, y, max_depth=3, n_trees=10)
train_errors, test_errors = gbgboost.gb_fit(coefs)
gbgboost.evaluate_alg(coefs)
