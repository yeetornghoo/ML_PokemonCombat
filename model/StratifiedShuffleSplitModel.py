class StratifiedShuffleSplitModel:

    def __init__(self, xtrain, ytrain, xtest, ytest):
        self.X_train = xtrain
        self.y_train = ytrain
        self.X_test = xtest
        self.y_test = ytest