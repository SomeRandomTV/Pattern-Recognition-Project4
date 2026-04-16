import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal




def load_data(path_to_data: Path):

    try:
        data_path = pd.read_excel(path_to_data)
        data_path.columns = ["F0","F1", "F2", "F3", "F4", "CLASS"]
        _X = np.array(data_path[["F0", "F1", "F2", "F3", "F4"]])
        _Y = np.array(data_path["CLASS"])
        return _X, _Y

    except FileNotFoundError as e:
        raise (e)


class BayesClassifier:

    def __init__(self):

        self.c1_m = np.zeros(5)
        self.c2_m = np.ones(5)

        self.priors = np.array([0.5, 0.5])

        self.true_covs = [
            np.array([
                [0.8,  0.2,  0.1,  0.05, 0.01],
                [0.2,  0.7,  0.1,  0.03, 0.02],
                [0.1,  0.1,  0.8,  0.02, 0.01],
                [0.05, 0.03, 0.02, 0.9,  0.01],
                [0.01, 0.02, 0.01, 0.01, 0.8 ]
            ]),
            np.array([
                [0.9,  0.1,  0.05, 0.02, 0.01],
                [0.1,  0.8,  0.1,  0.02, 0.02],
                [0.05, 0.1,  0.7,  0.02, 0.01],
                [0.02, 0.02, 0.02, 0.6,  0.02],
                [0.01, 0.02, 0.01, 0.02, 0.7 ]
            ])
        ]

        self.means = []
        self.covs = []

    @staticmethod
    def plot_fhistogram(X: np.ndarray, y: np.ndarray, title: str):
        n_features = X.shape[1]

        fig, axes = plt.subplots(1, n_features, figsize=(20,6))

        for i, ax in enumerate(axes):
            ax.hist(X[y == 1, i], bins=60, alpha=0.6, label="Class 1", color="steelblue")
            ax.hist(X[y == 2, i], bins=60, alpha=0.6, label="Class 2", color="tomato")
            ax.set_title(f"Feature {i}")
            ax.legend()

        plt.suptitle(f"{title}Distribution by Feature")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_phistogram(y_true: np.ndarray, y_pred: np.ndarray, error_rate: np.float16, title: str):

        fig, axes = plt.subplots(figsize=(15,6))
        axes.hist(y_pred[y_true == 1], bins=5, alpha=0.6, label="Actual Class 1", color="steelblue")
        axes.hist(y_pred[y_true == 2], bins=5, alpha=0.6, label="Actual Class 2", color="tomato")

        _stats = f'Test Error: {error_rate:.4f}'
        axes.text(0.05, 0.95, _stats, transform=axes.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        axes.legend()

        plt.suptitle(f"Predictions vs Truth ({title})")
        plt.show()

    def fit(self, X: np.ndarray, y: np.ndarray, mode="nb"):

        if mode == "nb":
            m1 = np.mean(X[y == 1], axis=0)
            m2 = np.mean(X[y == 2], axis=0)
            v1 = np.var(X[y == 1], axis=0)
            v2 = np.var(X[y == 2], axis=0)

            self.means = [m1, m2]
            self.covs = [np.diag(v1), np.diag(v2)]

        elif mode == "mle":
            m1 = np.mean(X[y == 1], axis=0)
            m2 = np.mean(X[y == 2], axis=0)
            c1 = np.cov(X[y == 1].T)
            c2 = np.cov(X[y == 2].T)

            self.means = [m1, m2]
            self.covs = [c1, c2]

        elif mode == "true":
            self.means = [self.c1_m, self.c2_m]
            self.covs = self.true_covs


    def predict(self, X: np.ndarray):

        dist1 = multivariate_normal(mean=self.means[0], cov=self.covs[0])
        dist2 = multivariate_normal(mean=self.means[1], cov=self.covs[1])

        log_p1 = dist1.logpdf(X) + np.log(self.priors[0])
        log_p2 = dist2.logpdf(X) + np.log(self.priors[1])

        scores = np.stack([log_p1, log_p2], axis=1)
        predictions = np.argmax(scores, axis=1) + 1

        return predictions

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray):

        errors = np.sum(y_true != y_pred)
        error_rate = errors / len(y_true)

        print(f"Total Samples: {len(y_true)}")
        print(f"Misclassified: {errors}")
        print(f"Test Error: {error_rate:.4f}")

        return error_rate


def main():

    train_data = Path(sys.argv[1])
    test_data = Path(sys.argv[2])

    if test_data is None or train_data is None:
        print("ERROR: Expected 2 pos arguments but at least one is missing")
        print("USAGE: this_file.py <path/to/train_data> <path/to/test_data>")
        exit(1)

    tr_X, tr_y = load_data(train_data)
    tst_X, tst_y = load_data(test_data)

    bs = BayesClassifier()
    bs.plot_fhistogram(tr_X, tr_y, title=f"Train data set ({len(tr_X)}) samples\n")
    bs.plot_fhistogram(tst_X, tst_y, title=f"Test data set ({len(tst_y)}) samples\n")

    for  mode in ["nb", "mle", "true"]:
        print(f"======== {mode}  ========\n")
        bs.fit(tr_X, tr_y, mode)
        p = bs.predict(tst_X)
        e_rate = bs.evaluate(tst_y, p)
        bs.plot_phistogram(tst_y, p, e_rate, title=mode)
        print("\n========================\n")



if __name__ == "__main__":
    main()



