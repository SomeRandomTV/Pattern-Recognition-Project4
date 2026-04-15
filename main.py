from os import stat
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt



def load_data(path_to_data: Path):

    try:
        data_path = pd.read_excel(path_to_data)
        data_path.columns = ["F0","F1", "F2", "F3", "F4", "CLASS"]
        _X = np.array(data_path[["F0", "F1", "F2", "F3", "F4"]])
        _Y = np.array(data_path["CLASS"])
        return (data_path, _X, _Y)

    except FileNotFoundError as e:
        raise (e)


class BayesClassifier:

    def __init__(self):

        self.c1_m = np.zeros([5,1])
        self.c2_m = np.ones([5,1])

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

        self.mle_means = None
        self.mle_cov = None
        self.naive_means = None
        self.naive_cov = None


    @staticmethod
    def data_stats(data):

        print("========== Data Stats =========")
        print(" ----- Head ------")
        print(data.head())
        print(" ----- Stats -----")
        print(data.describe())


    @staticmethod
    def plot_histogram(X: np.array, y: np.array):
        n_features = X.shape[1]

        fig, axes = plt.subplots(1, n_features, figsize=(20,6))
        

        for i, ax in enumerate(axes):
            ax.hist(X[y == 1, i], bins=30, alpha=0.6, label="Class 1", color="steelblue")
            ax.hist(X[y == 2, i], bins=30, alpha=0.6, label="Class 2", color="tomato")
            ax.set_title(f"Feature {i+1}")
            ax.legend()

        plt.suptitle("Distribution by Feature")
        plt.tight_layout()
        plt.show()






def main():

    _path_to_data = Path(sys.argv[1])
    _input_data, X, y = load_data(_path_to_data)
    print(np.unique(y))
    bs = BayesClassifier()
    bs.data_stats(_input_data)
    bs.plot_histogram(X, y)



if __name__ == "__main__":
    main()



