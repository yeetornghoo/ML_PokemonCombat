from sklearn.model_selection import StratifiedShuffleSplit
from model.StratifiedShuffleSplitModel import StratifiedShuffleSplitModel


def run(data, label_name, label_map):

    # data.to_csv('data_test_model.csv', index=False)
    X = data.copy().drop([label_name], axis=1)  # REMOVED TARGET / LABEL
    Y = data.copy()[label_name]
    Y = data.copy()[label_name].map(label_map).values
    split = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=42)
    split.get_n_splits(X, Y)

    for train_index, test_index in split.split(X, Y):
        model = StratifiedShuffleSplitModel(
            X.loc[train_index], Y[train_index],  # xtrain, ytrain
            X.loc[test_index], Y[test_index]  # xtest, ytest
        )
    """
    print("-- SPLIT TEST MODEL ----------")
    print("X_train:", model.X_train.shape)
    print("y_train:", model.y_train.shape)
    print("X_test:", model.X_test.shape)
    print("y_test:", model.y_test.shape)
    print("\n\n\n")
    """
    return model