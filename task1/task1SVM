# Name: Andy Ngo
# Date: 06/05/2025
# Description: Predicting the topic of a document from term-frequency vectors. This is a multiclass classification task.
# Support Vector Classification implementation

# how to call
# python task1SVM.py -config proj_data/task1_topics/task1.config -train_feat proj_data/task1_topics/train.sparseX -train_target proj_data/task1_topics/train.CT -dev_feat proj_data/task1_topics/dev.sparseX -dev_target proj_data/task1_topics/dev.CT


import argparse
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
from scipy.sparse import coo_matrix
import warnings
import wandb


# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, required=True, help="Path to config file")
    parser.add_argument('-train_feat', type=str, required=True, help="Path to training features file")
    parser.add_argument('-train_target', type=str, required=True, help="Path to training labels file")
    parser.add_argument('-dev_feat', type=str, required=True, help="Path to development features file")
    parser.add_argument('-dev_target', type=str, required=True, help="Path to development labels file")
    parser.add_argument('--max_iter', type=int, default=1000, help="Maximum number of iterations for LinearSVC")
    return parser.parse_args()


# Load config file
def load_config(path):
    with open(path, 'r') as f:
        token = f.read().split()
    n_train = int(token[1])
    n_dev = int(token[3])
    d = int(token[5])
    c = int(token[7])
    return n_train, n_dev, d, c


# Load sparse file
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
    n_train, n_dev, d, c = load_config(args.config)
    print(f"Train docs: {n_train}, Dev docs: {n_dev}, Vocab size: {d}, Classes: {c}")

    # Initialize Weights & Biases run
    wandb.init(
        project="multiclass-text-classification-svm",
        config={
            "max_iter": args.max_iter,
            "n_train": n_train,
            "n_dev": n_dev,
            "vocab_size": d,
            "num_classes": c
        }
    )
    config = wandb.config

    # Load datasets
    X_train = load_sparse(args.train_feat, (n_train, d)).tocsr()
    y_train = np.loadtxt(args.train_target, dtype=int)
    X_dev = load_sparse(args.dev_feat, (n_dev, d)).tocsr()
    y_dev = np.loadtxt(args.dev_target, dtype=int)
    print("Data loaded.")

    # Suppress convergence warnings
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    # Train LinearSVC model
    clf = LinearSVC(
        max_iter=config.max_iter,
        dual=False,
        verbose=0
    )
    clf.fit(X_train, y_train)
    print("Model trained.")

    # Evaluate on train and dev sets
    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_dev_pred = clf.predict(X_dev)
    dev_accuracy = accuracy_score(y_dev, y_dev_pred)

    # Log overall accuracy
    wandb.log({
        "train_accuracy": train_accuracy,
        "dev_accuracy": dev_accuracy
    })

    # Compute majority-class baseline for train set
    train_classes = [0] * c
    for i in range(n_train):
        train_classes[int(y_train[i])] += 1
    train_majority = max(train_classes)
    train_baseline = train_majority / n_train

    # Compute majority-class baseline for dev set
    dev_classes = [0] * c
    for i in range(n_dev):
        dev_classes[int(y_dev[i])] += 1
    dev_majority = max(dev_classes)
    dev_baseline = dev_majority / n_dev

    print("Train baseline: %f" % train_baseline)
    print("Dev baseline: %f" % dev_baseline)

    # Compute per-class precision
    dev_report = classification_report(y_dev, y_dev_pred, output_dict=True)
    class_precisions = {
        cls: dev_report[cls]['precision']
        for cls in map(str, range(c)) if cls in dev_report
    }

    # Log per-class precision table to Weights & Biases
    precision_table = wandb.Table(columns=["Class", "Precision"])
    for cls, precision in class_precisions.items():
        precision_table.add_data(cls, precision)

    wandb.log({
        "Per-Class Precision (Dev Set)": precision_table
    })

    # Log a sample of predictions to Weights & Biases
    pred_table = wandb.Table(columns=["Sample_ID", "True_Label", "Predicted_Label"])
    for i in range(min(100, len(y_dev))):
        pred_table.add_data(i, int(y_dev[i]), int(y_dev_pred[i]))

    wandb.log({"Sample Predictions (Dev Set)": pred_table})

    # End Weights & Biases run
    wandb.finish()


if __name__ == "__main__":
    main()
