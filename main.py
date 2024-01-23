from utils import pre
from pipeline import pipe, column_trans, gen_pipe
from sklearn.compose import ColumnTransformer
import numpy as np 

data = pre().creation_data('Salary_Data.csv')
target, features, names_f = pre().declaration_var(data, 'Salary')

# #Cambiar el tipo de formato
features['Age'] = features['Age'].astype('int16')
features['Years of Experience'] = features['Years of Experience'].astype('int64')

# Unir las variables repetidas
features = pre().replace('Education Level', "Bachelor's", "Bachelor's Degree", features)
features = pre().replace('Education Level', "Master's", "Master's Degree", features)
features = pre().replace('Education Level', "PhD", "phD", features)

num_features = ['Age', 'Years of Experience']
cat_features = ['Gender', 'Education Level', 'Job Title']

#Convertir las variables catégoricas en OneHotEncoder
num_transformer, cat_transformer = pipe(cat_features, features)
preprocessor = column_trans(num_transformer, num_features, cat_transformer, cat_features)

from sklearn.model_selection import train_test_split 
X_train, X_test,y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

linear_model = gen_pipe(preprocessor)

reg_l = linear_model.fit(X_train, y_train)

print(reg_l)

from sklearn.metrics import r2_score, mean_squared_error
prediction_linear = linear_model.predict(X_test)
print(r2_score(y_test, prediction_linear))
print(mean_squared_error(y_test, prediction_linear))
print(np.mean(target))
print('rmse', np.sqrt(mean_squared_error(y_test, prediction_linear)))