class StratifiedShuffleSplitModel:

    def __init__(self, xtrain, ytrain, xdev, ydev, xtest, ytest):
        self.X_train = xtrain
        self.y_train = ytrain
        self.X_dev = xdev
        self.y_dev = ydev
        self.X_test = xtest
        self.y_test = ytest