import pandas as pd
import numpy as np
from beautifultable import BeautifulTable
from sklearn.linear_model import LogisticRegression

from utils import start_time, end_time
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, balanced_accuracy_score, \
    roc_auc_score
from xgboost import XGBClassifier

methods = {
    "missing_value": ("zero", "mean", "lr", "drop"),
    "normalization": ("minmax", "zscore"),
    "feature_selection": ("lvcf", "pca"),
    "machine_learning": ("dt", "svm", "ann", "lr", "rf", "xgb", "bagging"),
}

# methods = {
#     "missing_value": ("zero",),
#     "normalization": ("minmax",),
#     "feature_selection": ("lvcf",),
#     "machine_learning": ("dt",),
# }


def run(dataset, mv, nm, fc, ml):
    """
    Implement each method into it's own function declaration
    """

    # print("loading dataset ...")
    local_dataset = dataset
    local_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)

    """
    Handle Missing Values
    """
    # print("handling missing values ...")
    if mv == "zero":
        local_dataset.replace(to_replace=np.nan, value=0, inplace=True)
    elif mv == "mean":
        local_dataset.fillna(local_dataset.mean(), inplace=True)
    elif mv == "lr":
        local_dataset.interpolate(method='linear', limit_direction='forward', inplace=True)
    elif mv == "drop":
        local_dataset.dropna(inplace=True)

    # print("preparing dataset ...")
    label = local_dataset[["Label"]]
    clean = local_dataset.drop(["Label"], axis=1)

    """
    Data normalization
    """
    # print("normalizing dataset ...")
    if nm == "minmax":
        scaler = MinMaxScaler()
        local_dataset = pd.DataFrame(scaler.fit_transform(clean), columns=clean.columns)
    elif nm == "zscore":
        scaler = StandardScaler()
        local_dataset = pd.DataFrame(scaler.fit_transform(clean), columns=clean.columns)
    local_dataset["Label"] = label
    clean = local_dataset.drop(["Label"], axis=1)

    """
    Feature Selection
    """
    # print("optimizing features ...")
    if fc == "lvcf":
        lv = VarianceThreshold(threshold=0.03)
        lv.fit_transform(clean)
        filtered_cols = clean.drop(
            columns=clean.columns[lv.get_support()].array
        ).columns.array
        lvcf_dataset = clean.drop(columns=filtered_cols)
        lvcf_dataset["Label"] = local_dataset["Label"].astype('category').cat.codes
        cor = lvcf_dataset.corr()
        cor_target = abs(cor["Label"])
        relevant_features = cor_target[cor_target > 0.2]
        all_feature = set(local_dataset.columns)
        relevant_features_idx = set(relevant_features.index)
        removed_feature = list(all_feature - relevant_features_idx)
        local_dataset.drop(columns=removed_feature, inplace=True)
    elif fc == "pca":
        pca = PCA(0.95)
        pca_dataset = pd.DataFrame(pca.fit_transform(clean))
        pca_dataset["Label"] = local_dataset["Label"]
        local_dataset = pca_dataset

    local_dataset["Label"] = local_dataset["Label"].astype("category").cat.codes
    clean = local_dataset.drop(["Label"], axis=1)
    label = local_dataset["Label"]

    """
    Preparing dataset for machine learning
    """
    # print("preparing dataset for classification ...")
    x_train, x_test, y_train, y_test = train_test_split(
        clean, label, test_size=1 / 7.0, random_state=1
    )

    """
    Machine Learning
    """
    # print("run machine learning classification ...")
    clf = None
    start = start_time()
    if ml == "dt":
        clf = DecisionTreeClassifier(criterion="entropy")
        clf.fit(x_train, y_train)
    elif ml == "svm":
        clf = SVC()
        clf.fit(x_train, y_train)
    elif ml == "ann":
        clf = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
        clf.fit(x_train, y_train)
    elif ml == "lr":
        clf = LogisticRegression(random_state=0)
        clf.fit(x_train, y_train)
    elif ml == "rf":
        clf = RandomForestClassifier(random_state=0)
        clf.fit(x_train, y_train)
    elif ml == "xgb":
        clf = XGBClassifier()
        clf.fit(x_train, y_train)
    elif ml == "bagging":
        clf = BaggingClassifier()
        clf.fit(x_train, y_train)
    end = end_time()
    duration = end - start

    """
    Analytics
    """
    # print("gathering analytical data ...")
    scores = cross_val_score(estimator=clf, X=clean, y=label, cv=5, n_jobs=8)
    print("Mean: {:.3f} (std: {:.3f})".format(scores.mean(), scores.std()))

    y_pred = clf.predict(x_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print("Acc: ", acc)

    # Balanced Accuracy
    bacc = balanced_accuracy_score(y_test, y_pred)
    print("Balanced Acc: ", bacc)

    # Recall
    recall = recall_score(y_test, y_pred)
    print("Rec:", recall)

    # F1 Score
    f1 = f1_score(y_test, y_pred)
    print("F1:", f1)

    # Precision
    precision = precision_score(y_test, y_pred)
    print("Prec", precision)

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred)
    print("ROC Auc", roc_auc)

    print("Duration", duration)

    table.rows.append(["[{mv} {nm} {fc} {ml}]".format(
        mv=mv, nm=nm,
        fc=fc, ml=ml
    ), scores.mean(), acc, bacc, recall, f1, precision, roc_auc, duration])


if __name__ == '__main__':

    table = BeautifulTable(maxwidth=400)
    table.columns.header = ["", "Cross-Validation", "Accuracy", "Balanced Accuracy", "Recall", "F1", "Precision",
                            "ROC AUC", "Time"]

    ds = pd.read_csv("/media/kmdr7/Seagate/TA/DATASETS/Dataset.csv")
    ds.replace([np.inf, -np.inf], np.nan, inplace=True)
    itr = 1

    for missing_value in methods.get("missing_value"):
        for normalization in methods.get("normalization"):
            for feature_selection in methods.get("feature_selection"):
                for machine_learning in methods.get("machine_learning"):
                    print("Run number [{itr}]".format(itr=itr))
                    print(
                        "Combination [{mv} {nm} {fc} {ml}]"
                        .format(
                            mv=missing_value, nm=normalization,
                            fc=feature_selection, ml=machine_learning
                        )
                    )
                    run(ds, missing_value, normalization, feature_selection, machine_learning)
                    itr += 1
                    print("==============================================================")
    print()
    print(table)
