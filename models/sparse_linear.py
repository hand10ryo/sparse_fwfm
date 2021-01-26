import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from copy import deepcopy


class Model:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        raise NotImplementedError

    def __call__(self, X_test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class Linear(Model):
    def __init__(self, df_X: pd.DataFrame, df_y: pd.Series):
        self.fields = df_X.columns
        self.fields_dir = {}

        field_start_idx = 0
        for i, field in enumerate(self.fields):
            ohe = OneHotEncoder(handle_unknown="ignore")
            X_field = ohe.fit_transform(df_X[[field]])
            cols = ohe.categories_[0]

            if i == 0:
                X = csr_matrix(X_field)
            else:
                X = hstack([X, X_field])

            self.fields_dir[field] = {
                "field_idx": i,
                "start_idx": field_start_idx,
                "end_idx": field_start_idx + len(cols),
                "cols": deepcopy(cols),
                "encoder": deepcopy(ohe),
            }
            field_start_idx += len(cols)

        self.X = csr_matrix(X)
        self.y = df_y.values

        self.train()

    def train(self):
        self.reg = LinearRegression()
        self.reg.fit(self.X, self.y)

    def predict(self, X: csr_matrix):
        return self.reg.predict(X)

    def convert_sparse(self, df_X: pd.DataFrame):
        for i, field in enumerate(self.fields):
            X_field = self.fields_dir[field]["encoder"].transform(
                df_X[[field]])
            if i == 0:
                X = csr_matrix(X_field)
            else:
                X = hstack([X, X_field])
        return csr_matrix(X)

    def __call__(self, df_X: pd.DataFrame):
        X = self.convert_sparse(df_X)
        return self.predict(X)
