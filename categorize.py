from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X = [
    [100, 1.90, 'rugby'],
    [110, 2.30, 'basket'],
    [90, 1.89, 'volleyball'],
    [70, 1.70, 'soccer']
]
transformers = [
    ['category vectorizer', OneHotEncoder(), [2]]
]


ct = ColumnTransformer(transformers, remainder='passthrough')

ct.fit(X)
X = ct.transform(X)

print(X)