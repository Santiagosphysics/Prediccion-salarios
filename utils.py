import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

class pre:
    def creation_data(self, path_csv):
        df = pd.read_csv(path_csv)
        return df
    
    def declaration_var(self, data, target):
        
        target_1 = data[target]
        features = data.drop(target, axis=1)
        names_f = {name:set() for name in features}
        for i in features:
            index_drop = features[i][features[i].isnull() == True].index
            features.drop(index_drop, inplace = True)    
            target_1.drop(index_drop, inplace = True)
        
        for i in features:
            for j in features[i]:
                names_f[i].add(j)   
                    
        index_targ = target_1[target_1.isnull() == True].index
        target_1.drop(index_targ, inplace=True)   
        features.drop(index_targ, inplace=True)                    
        return target_1, features, names_f
    
    def one_hot(self, variable, features):
        # for category in variable:
        #     if category not in features.columns:
        #         features[category] = 0  # Agrega la categoría con valores 0 si no está presente
        encoder = OneHotEncoder()
        feature_encoded = encoder.fit_transform(features[variable])
        df_one_hot = pd.DataFrame(feature_encoded.toarray(), columns=encoder.get_feature_names_out(variable))
        features.drop(variable, axis=1, inplace=True)
        names_one = [name for name in df_one_hot]
        df_one_hot[names_one] = df_one_hot[names_one].astype('int8')
        features.reset_index(drop=True, inplace=True)
        df_one_hot.reset_index(drop=True, inplace=True)
        features = pd.concat([features, df_one_hot], axis=1)
        return featuresbarras 
    
    def replace(self, variable, var_1, var_2, features):
        new_values = [var_1 if value == var_1 or value == var_2 else value for value in features[variable]]
        features[variable] = new_values
        return features

