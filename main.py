from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import mlflow.sklearn
import pandas as pd
import numpy as np
import mlflow


train=pd.read_csv('train.csv')

train['position'] = train['id_seqpos'].apply(lambda x: int(x.split('_')[-1]))
train = train.drop(columns=['id'])

# Handling missing values
# Impute missing values in features related to Garage with "No Garage"
mappings = {
    'sequence':    {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'b1_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'b2_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'b3_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'b4_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'b5_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},

    'a1_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'a2_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'a3_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'a4_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},
    'a5_sequence': {'G': '1', 'C': '2', 'U': '4', 'A': '3','-1':""},

    'structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'b1_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'b2_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'b3_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'b4_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'b5_structure': {'(': '1', ')': '2', '.': '3','-1':""},

    'a1_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'a2_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'a3_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'a4_structure': {'(': '1', ')': '2', '.': '3','-1':""},
    'a5_structure': {'(': '1', ')': '2', '.': '3','-1':""},

    'predicted_loop_type':    {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'b1_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'b2_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'b3_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'b4_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'b5_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},

    'a1_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'a2_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'a3_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'a4_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""},
    'a5_predicted_loop_type': {'S': '1', 'E': '2', 'H': '3', 'I': '4','X': '5', 'M': '6', 'B': '7','-1':""}

}
def apply_mappings(df, mappings):
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

train = apply_mappings(train, mappings)

cat_col=[]
for i in train.columns:
    if train[i].dtypes != 'O':
        continue
    else:
        cat_col.append(i)


train['sequence_update'] =train['b5_sequence']+train['b4_sequence']+train['b3_sequence']+train['b2_sequence']+train['b1_sequence']+train['sequence']+train['a1_sequence']+train['a2_sequence']+train['a3_sequence']+train['a4_sequence'] +train['a5_sequence']
train['predicted_loop_type_update'] =train['b5_predicted_loop_type']+train['b4_predicted_loop_type']+train['b3_predicted_loop_type']+train['b2_predicted_loop_type']+train['b1_predicted_loop_type']+train['predicted_loop_type']+train['a1_predicted_loop_type']+train['a2_predicted_loop_type']+train['a3_predicted_loop_type']+train['a4_predicted_loop_type'] +train['a5_predicted_loop_type']
train['structure_update'] =train['b5_structure']+train['b4_structure']+train['b3_structure']+train['b2_structure']+train['b1_structure']+train['structure']+train['a1_structure']+train['a2_structure']+train['a3_structure']+train['a4_structure'] +train['a5_structure']


train['sequence_update']=train['sequence_update'].apply(lambda x : int(x))
train['predicted_loop_type_update']=train['sequence_update'].apply(lambda x : int(x))
train['structure_update']=train['structure_update'].apply(lambda x : int(x))

train = train.drop(columns=['b1_sequence', 'a1_sequence',
       'b1_structure', 'a1_structure', 'b1_predicted_loop_type',
       'a1_predicted_loop_type', 'b2_sequence', 'a2_sequence', 'b2_structure',
       'a2_structure', 'b2_predicted_loop_type', 'a2_predicted_loop_type',
       'b3_sequence', 'a3_sequence', 'b3_structure', 'a3_structure',
       'b3_predicted_loop_type', 'a3_predicted_loop_type', 'b4_sequence',
       'a4_sequence', 'b4_structure', 'a4_structure', 'b4_predicted_loop_type',
       'a4_predicted_loop_type', 'b5_sequence', 'a5_sequence', 'b5_structure',
       'a5_structure', 'b5_predicted_loop_type', 'a5_predicted_loop_type','sequence', 'structure', 'predicted_loop_type'])



