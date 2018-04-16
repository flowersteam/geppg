import numpy as np

class KNNRegressor():
    """ A wrap around sklearn knn regressor.
    """

    def __init__(self, n_neighbors=5):

        try:
            from sklearn.neighbors import KNeighborsRegressor
            globals()['KNeighborsRegressor'] = KNeighborsRegressor
        except:
            raise ImportError("You need sklearn.neighbors to use class {}".format(self.__class__.__name__))

        self._model = KNeighborsRegressor(n_neighbors=n_neighbors, metric='euclidean', algorithm='ball_tree', weights='distance')
        self._prediction = None
        self._X = None
        self._Y = None

    def update(self, X, Y):
        """
        X: representation
        Y: policy
        """
        if self._X is None:
            self._X = np.copy(X).reshape(1,-1)
            self._Y = np.copy(Y).reshape(1,-1)
        else:
            self._X = np.concatenate([self._X, X.reshape(1, -1)], axis=0)
            self._Y = np.concatenate([self._Y, Y.reshape(1, -1)], axis=0)

        self._model.fit(self._X, self._Y)

    def predict(self, input_X):
        """
        input_X: goal (representation)
        """
        in_knn = np.copy(input_X)
        return self._model.predict(in_knn.reshape(1, -1))


    def terminate(self):
        pass

    @property
    def prediction(self):
        return self._prediction