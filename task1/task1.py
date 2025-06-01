# Predicting the topic of a document from term-frequency vectors. This is a multiclass classification task.
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help='Task config file')
    parser.add_argument('-train_feat', type=str, required=True, help='Training feature file')
    parser.add_argument('-train_target', type=str, required=True, help='Training target file')
    parser.add_argument('-dev_feat', type=str, required=True, help='Development feature file')
    parser.add_argument('-dev_target', type=str, required=True, help='Development target file')
    return parser.parse_args()

# Load config values
def load_config(path):
    with open(path, 'r') as f:
        token = f.read().split()
    n_train = int(token[1])
    n_dev = int(token[3])
    d = int(token[5])
    c = int(token[7])
    return n_train, n_dev, d, c

# Load sparse data as sparse matrix
def load_sparse(path, shape):
    rows, cols, vals = [], [], []
    with open(path, 'r') as file:
        for line in file:
            row, col, value = map(float, line.strip().split())
            rows.append(int(row))
            cols.append(int(col))
            vals.append(value)
    return coo_matrix((vals, (rows, cols)), shape=shape)

def main():
    args = parse_arguments()

    # Load config
    n_train, n_dev, d, c = load_config(args.config)
    print(f"Train docs: {n_train}, Dev docs: {n_dev}, Vocab size: {d}, Classes: {c}")

    # Load data
    X_train = load_sparse(args.train_feat, (n_train, d)).tocsr()
    y_train = np.loadtxt(args.train_target, dtype=int)

    X_dev = load_sparse(args.dev_feat, (n_dev, d)).tocsr()
    y_dev = np.loadtxt(args.dev_target, dtype=int)

    print("Data loaded.")

    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Train multiclass logistic regression
    clf = LogisticRegression(
        solver='saga',
        max_iter=10,
        n_jobs=-1,
        verbose=0
    )

    clf.fit(X_train, y_train)
    print("Model trained.")

    # Evaluate
    y_pred = clf.predict(X_dev)
    accuracy = accuracy_score(y_dev, y_pred)
    print(f"Development set accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
