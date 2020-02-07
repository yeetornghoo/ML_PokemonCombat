import pandas as pd
from controller import DataCleaning
from controller import TestSetController
from controller import ClassificationController

# LOAD TEST DATA
data = pd.read_csv('dataset/master_pokemons.csv')
# QualityReportController.run(data)
data = DataCleaning.run(data)

# -- TRAIN / TEST MODEL
label_name = "pokemon_type_1"
label_map = {"Bug": 0, "Dark": 1, "Dragon": 2, "Electric": 3, "Fairy": 4,
             "Fighting": 5, "Fire": 6, "Flying": 7, "Ghost": 8, "Grass": 9,
             "Ground": 10, "Ice": 11, "Normal": 12, "Poison": 13, "Psychic": 14,
             "Rock": 15, "Steel": 16, "Water": 17}

data_model = TestSetController.run(data, label_name, label_map)
ClassificationController.run(data_model)


#TensorFlowController.run(data_model)