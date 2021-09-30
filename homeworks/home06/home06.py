from sklearn.tree import DecisionTreeRegressor

from sklearn import model_selection
import numpy as np


# %%

class GBGradBoost:
    n_trees = 0
    max_depth = 0

    def __init__(self, n_trees=10, max_depth=3):
        self.params(n_trees, max_depth)

    def params(self, n_trees=10, max_depth=3):
        self.n_trees = n_trees
        self.max_depth = max_depth

    def gb_predict(self, X, trees_list, coef_list, eta):
        # Реализуемый алгоритм градиентного бустинга будет инициализироваться нулевыми значениями,
        # поэтому все деревья из списка trees_list уже являются дополнительными и при предсказании прибавляются с шагом eta
        return np.array(
            [sum([eta * coef * alg.predict([x])[0] for alg, coef in zip(trees_list, coef_list)]) for x in X])

    def mean_squared_error(self, y_real, prediction):
        return (sum((y_real - prediction) ** 2)) / len(y_real)

    def bias(self, y, z):
        return (y - z)

    # функция обучения градиентного бустинга.

    def gb_fit(self, X_train, X_test, y_train, y_test, coefs, eta):
        # Деревья будем записывать в список
        trees = []

        # Будем записывать ошибки на обучающей и тестовой выборке на каждой итерации в список
        train_errors = []
        test_errors = []

        for i in range(self.n_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=42)

            # инициализируем бустинг начальным алгоритмом, возвращающим ноль,
            # поэтому первый алгоритм просто обучаем на выборке и добавляем в список
            if len(trees) == 0:
                # обучаем первое дерево на обучающей выборке
                tree.fit(X_train, y_train)

                train_errors.append(self.mean_squared_error(y_train, self.gb_predict(X_train, trees, coefs, eta)))
                test_errors.append(self.mean_squared_error(y_test, self.gb_predict(X_test, trees, coefs, eta)))
            else:
                # Получим ответы на текущей композиции
                target = self.gb_predict(X_train, trees, coefs, eta)

                # алгоритмы начиная со второго обучаем на сдвиг
                tree.fit(X_train, self.bias(y_train, target))

                train_errors.append(self.mean_squared_error(y_train, self.gb_predict(X_train, trees, coefs, eta)))
                test_errors.append(self.mean_squared_error(y_test, self.gb_predict(X_test, trees, coefs, eta)))

            trees.append(tree)

        return trees, train_errors, test_errors

    def evaluate_alg(self, X_train, X_test, y_train, y_test, trees, coefs, eta):
        train_prediction = self.gb_predict(X_train, trees, coefs, eta)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_depth} \
        с шагом {eta} на тренировочной выборке: {self.mean_squared_error(y_train, train_prediction)}')

        test_prediction = self.gb_predict(X_test, trees, coefs, eta)

        print(f'Ошибка алгоритма из {self.n_trees} деревьев глубиной {self.max_depth} \
        с шагом {eta} на тестовой выборке: {self.mean_squared_error(y_test, test_prediction)}')


from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25)
# Число деревьев в ансамбле
n_trees = 10
# для простоты примем коэффициенты равными 1
coefs = [1] * n_trees
# Максимальная глубина деревьев
max_depth = 3
# Шаг
eta = 1

gbgboost = GBGradBoost(max_depth=3, n_trees=10)
trees, train_errors, test_errors = gbgboost.gb_fit(X_train, X_test, y_train, y_test, coefs, eta)
gbgboost.evaluate_alg(X_train, X_test, y_train, y_test, trees, coefs, eta)
