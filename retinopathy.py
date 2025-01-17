import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_knn_model(data):
    data['Years Since Diagnosis'] = 2024 - data['Diagnosis Year']
    
    X = data.drop(['Retinopathy Status', 'Retinopathy Probability', 'Diagnosis Year'], axis=1) 
    y = data['Retinopathy Probability'] 

    y = y.round().astype(int)

    categorical_features = ['Gender', 'Diabetes Type']
    numerical_features = list(set(X.columns) - set(categorical_features))

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)])

    knn_classifier = KNeighborsClassifier()

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', knn_classifier)])

    param_grid = {'classifier__n_neighbors': [3, 5, 7, 9, 11],
                  'classifier__weights': ['uniform', 'distance'],
                  'classifier__metric': ['euclidean', 'manhattan'],
                  'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  'classifier__leaf_size': [20, 30, 40, 50],
                  'classifier__p': [1, 2]}

    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X, y)

    return best_pipeline

def predict_with_knn(model, input_data):
    predicted_labels = model.predict(input_data)
    
    retinopathy_probabilities = [0, 0.5, 1.5, 2.5, 3.5, 4.5]
    predicted_probabilities = [retinopathy_probabilities[label] for label in predicted_labels]
    
    return predicted_probabilities

data = pd.read_csv('dataset/diabetes_retinopathy_dataset.csv')
knn_model = train_knn_model(data)

new_data = pd.DataFrame({
    'Gender': ['Male'],
    'Diabetes Type': ['Type 1'],
    'Systolic BP': [120],
    'Diastolic BP': [80],
    'HbA1c (mmol/mol)': [90],
    'Estimated Avg Glucose (mg/dL)': [150],
    'Diagnosis Year': [2014]
})
new_data['Years Since Diagnosis'] = 2024 - new_data['Diagnosis Year']

predictions = predict_with_knn(knn_model, new_data)
print("Predictions:", predictions)