from sklearn.model_selection import StratifiedShuffleSplit
from model.StratifiedShuffleSplitModel import StratifiedShuffleSplitModel


def run(data, label_name):
    X = data.copy().drop([label_name], axis=1)  # REMOVED TARGET / LABEL
    Y = data.copy()[label_name]
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    split.get_n_splits(X, Y)

    for train_index, test_index in split.split(X, Y):
        model = StratifiedShuffleSplitModel(
            X.iloc[train_index], Y[train_index],  # xtrain, ytrain
            X.iloc[test_index], Y[test_index],  # xdev, ydev
            X.iloc[test_index], Y[test_index]  # xtest, ytest
        )

    print("-- SPLIT TEST MODEL ----------")
    print("X_train:", model.X_train.shape)
    print("y_train:", model.y_train.shape)
    print("X_test:", model.X_test.shape)
    print("y_test:", model.y_test.shape)
    print("X_dev:", model.X_dev.shape)
    print("y_dev:", model.y_dev.shape)
    print("\n\n\n")
    return model
