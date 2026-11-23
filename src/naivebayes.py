# src.naivebayes

import numpy as np

class NaiveBayes:
    """
    Gaussian Naive Bayes classifier
    """

    def fit(self, X_train, y_train):
        """
        Fit the Naive Bayes model by computing statistics per class
        Args:
            X_train: matrix of features (N, D)
            y_train: vector of true labels (N,)

        Returns:
            None
        """
        # Get unique class labels
        self.classes = np.unique(y_train)
        # Compute prior probability P(c) for each class
        self.priors = [len(y_train[y_train==c]) / len(y_train) for c in self.classes]
        # Compute log of priors to avoid underflow
        self.priors_log = np.log(self.priors)

        # Compute mean of each feature for each class
        self.means = [X_train[y_train == c].mean() for c in self.classes]
        # Compute standard deviation of each feature for each class
        self.stds = [X_train[y_train == c].std() for c in self.classes]

    def compute_likelihood(self, row, class_index):
        """
        Compute log-likelihood of a row under a Gaussian distribution
        Args:
            row: one sample (Series of feature values)
            class_index: index of the class in self.classes

        Returns:
            float: log-likelihood (sum over features)
        """
        # Initialize likelihood accumulator
        likelihood = 0.0 # float

        # Loop over each feature of the row
        for feature in row.index:
            # Get mean of this feature for the class
            mean = self.means[class_index][feature]
            # Get standard deviation of this feature for the class
            std = self.stds[class_index][feature]
            # Avoiding divide to zero
            if np.isnan(std) or (std < 1e-9):
                std = 1e-9 # deny divide to zero (at line 21 divide std) 

            # Add log of Gaussian density of this feature to the likelihood
            likelihood = likelihood + (
                            -np.log(np.sqrt(2 * np.pi) * std)
                            - ((row[feature] - mean) ** 2) / (2 * std**2)
            )

        return likelihood

    def predict(self, X):
        """
        Predict class labels for samples
        Args:
            X: matrix of features (N, D)

        Returns:
            ndarray: predicted class labels, shape (N,)
        """
        # Initialize list of predictions
        y_pred = []

        # Loop over each sample
        for _, row in X.iterrows():
            # Initialize list of posterior scores
            poster = []
            # Loop over each class
            for i in range(len(self.classes)):
                # Compute log-likelihood of the row under this class
                likelihood = self.compute_likelihood(row, i)
                # Compute posterior: log-likelihood + log-prior
                poster.append(likelihood + self.priors_log[i])

            # Choose the class with the highest posterior
            y_pred.append(self.classes[np.argmax(poster)])

        return np.array(y_pred)
