import numpy as np


class NaiveBayes:
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.priors = [len(y_train[y_train==c]) / len(y_train) for c in self.classes]
        self.priors_log = np.log(self.priors)

        self.means = [X_train[y_train == c].mean() for c in self.classes]
        self.stds = [X_train[y_train == c].std() for c in self.classes]

    def compute_likelihood(self, row, class_index):

        likelihood = 0.0 # float

        for feature in row.index:
            mean = self.means[class_index][feature]
            std = self.stds[class_index][feature]
            if np.isnan(std) or (std < 1e-9):
                std = 1e-9 # deny divide to zero (at line 21 divide std) 

            likelihood = likelihood + (
                            -np.log(np.sqrt(2 * np.pi) * std)
                            - ((row[feature] - mean) ** 2) / (2 * std**2)
            )

        return likelihood

    def predict(self, X):
        y_pred = []

        for _, row in X.iterrows():
            poster = []
            for i in range(len(self.classes)):
                likelihood = self.compute_likelihood(row, i)
                poster.append(likelihood + self.priors_log[i])

            y_pred.append(self.classes[np.argmax(poster)])

        return np.array(y_pred)


