import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from controller import QualityReportController
from controller import DataCleaning
from controller import TensorFlowController
from tensorflow import keras
from controller import TestSetController
from model import StratifiedShuffleSplitModel

# LOAD TEST DATA


data = pd.read_csv('dataset/master_pokemons.csv', index_col=0)
#QualityReportController.run(data)
data = DataCleaning.run(data)


# -- TRAIN / TEST MODEL
label_name = "pokemon_type_1"
data_model = TestSetController.run(data, label_name)

## RUN ML
TensorFlowController.run(data_model)
#print(data.head())

