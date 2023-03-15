from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier

#
# Dataset fields
#
numeric_features = ["if"+str(i) for i in range(1,14)]
categorical_features = ["cf"+str(i) for i in range(1,27)] + ["day_number"]
used_categorical_features = ["cf6", "cf9", "cf13", "cf16", "cf17", "cf19", "cf25", "cf26"]

fields = ["id", "label"] + numeric_features + categorical_features
fields_types = dict(zip(numeric_features, [float] * len(numeric_features)))
fields_types.update(dict(zip(categorical_features, [str] * len(categorical_features))))
fields_types.update({'id': int, 'label': int})


#
# Model pipeline
#

# We create the preprocessing pipelines for both numeric and categorical data.
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
#    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, used_categorical_features)
    ]
)

class ColumnDropper:
    def __init__(self, cols):
        self.cols = cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.drop(columns=self.cols, inplace=True)

col_deleter = ColumnDropper(cols=list(set(categorical_features) - set(used_categorical_features)) + ['id'])

# Now we have a full prediction pipeline.
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('randomforest', RandomForestClassifier(verbose=2, n_estimators=35))
])