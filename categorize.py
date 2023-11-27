from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
X = [
    [100, 1.90, 'rugby'],
    [110, 2.30, 'basket'],
    [90, 1.89, 'volleyball'],
    [70, 1.70, 'soccer']
]

ct = ColumnTransformer()