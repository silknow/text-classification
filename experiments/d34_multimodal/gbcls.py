#!/usr/bin/env python

"""SilkNOW Gradient Boosting Classifier."""
import argparse
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV

# from sklearn.ensemble import GradientBoostingClassifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SilkNOW Gradient Boosting Classifier",
        usage="""sngbclassify <command> [<args>]

The available commands are:
   train        Train a text classification model
   evaluate     Evaluate a text classification model
   classify     Classify text samples
""",
    )
    subparsers = parser.add_subparsers(required=True, dest="command")

    #
    # TRAIN
    #
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--data-train", type=str, help="train CSV file", required=True
    )

    train_parser.add_argument(
        "--cols",
        help="CSV columns for data (x)",
        nargs="+",
        required=True,
    )

    train_parser.add_argument(
        "--target", type=str, help="CSV column for target (y)", required=True
    )

    train_parser.add_argument(
        "--model-save", type=str, help="save model path", required=True
    )

    #
    # Evaluate
    #
    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument(
        "--model-load", type=str, help="model path", required=True
    )
    eval_parser.add_argument(
        "--data-test", type=str, help="test CSV path", required=True
    )
    eval_parser.add_argument(
        "--cols",
        help="CSV columns for data (x)",
        nargs="+",
        required=True,
    )
    eval_parser.add_argument(
        "--target", type=str, help="CSV column target", required=True
    )

    #
    # Classify
    #
    classify_parser = subparsers.add_parser("classify")
    classify_parser.add_argument("--model-load", type=str, help="model path")
    classify_parser.add_argument(
        "--data-input", type=str, help="input CSV path", required=True
    )
    classify_parser.add_argument(
        "--data-output", type=str, help="output file path", required=True
    )
    classify_parser.add_argument(
        "--cols",
        help="CSV columns for data (x)",
        nargs="+",
        required=True,
    )

    classify_parser.add_argument(
        "--scores", action="store_true", help="output prediction scores"
    )

    args = parser.parse_args()
    return args


def train_model(
    train_file_path,
    data_cols,
    target_col,
    save_path,
    do_cv=False,
    eval_set_path=None,
):
    """Train a model."""
    all_cols = data_cols + [target_col]
    df = pd.read_csv(
        train_file_path,
        delimiter="\t",
        usecols=all_cols,
        keep_default_na=False,
        dtype={c: "str" for c in all_cols},
    )
    for col in all_cols:
        df[col] = df[col].astype(str)

    y = df[target_col].copy()

    data_encoders = {col: LabelEncoder() for col in data_cols}
    for col in data_cols:
        df[col] = data_encoders[col].fit_transform(df[col])

    x = df[data_cols].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    params = {
        "n_estimators": [10, 20, 30, 50, 70, 100],
    }

    clf = XGBClassifier(
        objective="multi:softmax",
        eval_metric="mlogloss",
        use_label_encoder=False,
    )
    if do_cv:
        clf = GridSearchCV(clf, params)
        clf.fit(x, y)
        print(clf.best_params_)
        clf = clf.best_estimator_
    else:
        clf.fit(x, y, verbose=True)

    model_dict = {
        "model": clf,
        "le": label_encoder,
        "data_encoders": data_encoders,
    }
    n_feats = len(clf.feature_importances_)
    n_cols = len(data_cols)
    print(f"data cols: {n_cols} -> feats: {n_feats}")
    for feature, importance in zip(data_cols, clf.feature_importances_):
        print(f"{feature}: {importance}")

    pickle.dump(model_dict, open(save_path, "wb"))
    return model_dict


def eval_model(load_path, test_file_path, data_cols, target_col):
    """Evaluate model."""
    all_cols = data_cols + [target_col]
    df = pd.read_csv(
        test_file_path,
        delimiter="\t",
        usecols=all_cols,
        keep_default_na=False,
        dtype={c: "str" for c in all_cols},
    )
    for col in all_cols:
        df[col] = df[col].astype(str)

    model_dict = pickle.load(open(load_path, "rb"))
    model = model_dict["model"]

    data_encoders = model_dict["data_encoders"]
    for col in data_cols:
        df[col] = data_encoders[col].transform(df[col])
    x = df[data_cols].copy()

    label_encoder = model_dict["le"]
    target_names = list(label_encoder.classes_)
    labels = list(range(len(target_names)))

    df[target_col] = label_encoder.transform(df[target_col])
    y = df[target_col].copy()

    p = model.predict(x)

    acc = accuracy_score(y, p)
    r = classification_report(y, p, labels=labels, target_names=target_names)
    print(r)
    print(f"accuracy:\t{acc}")

    pr, rc, f1, sp = precision_recall_fscore_support(
        y, p, beta=1.0, labels=labels, average="micro"
    )

    results = {
        "acc": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "support": sp,
    }
    return results


def classify_csv(
    load_path, test_file_path, result_file_path, data_cols, scores=False
):
    """Classify a CSV file, write to new file."""
    df = pd.read_csv(
        test_file_path,
        delimiter="\t",
        keep_default_na=False,
        dtype={c: "str" for c in data_cols},
    )
    for col in data_cols:
        df[col] = df[col].astype(str)

    x = df[data_cols]

    model_dict = pickle.load(open(load_path, "rb"))
    model = model_dict["model"]

    data_encoders = model_dict["data_encoders"]
    for col in data_cols:
        x[col] = data_encoders[col].transform(x[col])

    label_encoder = model_dict["le"]

    p = model.predict(x)
    p_labels = label_encoder.inverse_transform(p)

    df["predicted"] = p_labels

    df.to_csv(result_file_path, delimiter="\t", index=False)

    return p_labels


def main():
    """Read arguments, perform action."""
    args = parse_args()

    if args.command == "train":
        train_model(
            args.data_train,
            args.cols,
            args.target,
            args.model_save,
        )
    elif args.command == "evaluate":
        eval_model(args.model_load, args.data_test, args.cols, args.target)
    elif args.command == "classify":
        classify_csv(
            args.model_load,
            args.data_input,
            args.data_output,
            args.cols,
            args.scores,
        )


if __name__ == "__main__":
    main()
