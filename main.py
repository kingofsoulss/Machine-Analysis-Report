import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Simulated data collection
data = {
    'machine_id': np.arange(1, 101),
    'temperature': np.random.uniform(20, 100, 100),
    'pressure': np.random.uniform(1, 10, 100),
    'vibration': np.random.uniform(0, 5, 100),
    'type': np.random.choice(['Type A', 'Type B', 'Type C'], 100),
    'operational': np.random.choice([True, False], 100)
}

df = pd.DataFrame(data)

# Data preprocessing
numeric_features = ['temperature', 'pressure', 'vibration']
categorical_features = ['type']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Feature extraction and model training pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Splitting data into training and test sets
X = df.drop(columns=['machine_id', 'operational'])
y = df['operational']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
clf.fit(X_train, y_train)

# Predicting and evaluating the model
y_pred = clf.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)

# Generating textual report
def generate_report(df, report):
    total_machines = df.shape[0]
    operational_machines = df[df['operational']].shape[0]
    non_operational_machines = total_machines - operational_machines
    
    report_text = f"""
    Machine Analysis Report
    =======================
    Total Machines Analyzed: {total_machines}
    Operational Machines: {operational_machines}
    Non-Operational Machines: {non_operational_machines}
    
    Model Performance:
    ------------------
    Precision: {report['True']['precision']:.2f}
    Recall: {report['True']['recall']:.2f}
    F1-Score: {report['True']['f1-score']:.2f}
    
    Feature Importance:
    -------------------
    """
    
    feature_importance = clf.named_steps['classifier'].feature_importances_
    feature_names = np.concatenate((numeric_features, clf.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out()))
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    for feature, importance in sorted_features:
        report_text += f"{feature}: {importance:.4f}\n"
    
    return report_text

# Printing the report
print(generate_report(df, report)) 
