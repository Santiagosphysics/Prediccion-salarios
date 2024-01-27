from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from utils import pre

processing = pre()

def pipe(categorical_features, features):
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('one_hot', OneHotEncoder(handle_unknown='ignore'))])
    return num_transformer, cat_transformer

def column_trans(num_transformer, num_features, cat_transformer, cat_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', num_transformer, num_features),
            ('categorical', cat_transformer, cat_features)
        ]
    )
    return preprocessor

def gen_pipe( preprocessor):
    models = []    
    pipeline_ml = Pipeline(
        steps = [('preprocessor_column', preprocessor),
                ('linear_model', LinearRegression())])
    models.append(pipeline_ml)
        
    pipeline_ridge = Pipeline(
        steps = [('preprocessor_column', preprocessor),
                ('model_ridge', Ridge(alpha=0.2))])
    models.append(pipeline_ridge)
    
    pipeline_rf = Pipeline(
        steps=[('preprocessor_column', preprocessor),
            ('random_forest', RandomForestRegressor(n_estimators=200))])
    models.append(pipeline_rf)
    
    return models