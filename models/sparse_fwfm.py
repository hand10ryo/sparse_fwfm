import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import csr_matrix, hstack
from tqdm import tqdm
from opt_einsum import contract, contract_expression
from copy import deepcopy


class Model:
    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        raise NotImplementedError

    def __call__(self, X_test: pd.DataFrame) -> pd.Series:
        raise NotImplementedError


class FwFM(Model):
    """ Field weighted Factorization Machines implemented by scipy.sparse.csr_matrix
    See colaboratory notebook https://colab.research.google.com/drive/1AsOLL_7ON_Fl22rIJ3RngvLAxe5IOmBh?usp=sharing
    or original paper https://arxiv.org/pdf/1806.03514.pdf  for details.

    requirements:
        pandas,
        numpy,
        sklearn,
        scipy,
        tqdm,
        opt_einsum
    """

    def __init__(self, df_X: pd.DataFrame, df_y: pd.Series, sample_weight: np.ndarray,
                 dim: int = 8, lr: float = 1e-3, n_epoch: int = 10, n_batch: int = 256,
                 ignore_interactions: list = [], train: bool = True,
                 lam_w: float = 0, lam_v: float = 0, lam_r: float = 0):
        """
        Arguments:
            df_X [pd.DataFrame] : Explanatory variables
            df_y [pd.Series] : Objective variable
            dim [int] : a number of dimention of embeddings.
            lr [float] : learning rate
            n_epoch [int] : a number of epoch/
            n_batch [int] : a number of sample in mini-batch.
            ignore_interactions [list]: element is pair of fields ignored interaction
            train [bool] : wheter run train or not when initializing.
            lam_w [float] : weight of l2 norm for w.
            lam_v [float] : weight of l2 norm for v.
            lam_r [float] : weight of l2 norm for r.

        """
        self.sample_weight = sample_weight
        self.dim = dim
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.ignore_interactions = ignore_interactions
        self.lam_w = lam_w
        self.lam_v = lam_v
        self.lam_r = lam_r

        self._preprocess(df_X, df_y)
        if train:
            self.train()

    def _preprocess(self, df_X: pd.DataFrame, df_y: pd.Series):
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

        self.b = 0
        self.w = np.random.rand(X.shape[1]) / 10
        self.v = np.random.rand(X.shape[1], self.dim) / 10
        self.r = np.random.rand(len(self.fields), len(self.fields)) / 10
        self.r_mask = np.ones([len(self.fields), len(self.fields)])

        for i in range(len(self.fields)):
            for j in range(i, len(self.fields)):
                self.r_mask[i, j] = 0

        for interaction in self.ignore_interactions:
            field_i, field_j = tuple(interaction)
            field_i_idx = self.fields_dir[field_i]["field_idx"]
            field_j_idx = self.fields_dir[field_j]["field_idx"]
            self.r_mask[field_i_idx, field_j_idx] = 0
            self.r_mask[field_j_idx, field_i_idx] = 0

        self.r = self.r * self.r_mask

        self.m2f = np.zeros([X.shape[1], len(self.fields)])
        for i, field in enumerate(self.fields):
            self.m2f[np.arange(self.fields_dir[field]["start_idx"],
                               self.fields_dir[field]["end_idx"]), i] = 1

        self.X = csr_matrix(X)
        self.y = df_y.values

        self.contract_predict = None
        self.contract_der_v = None
        self.contract_der_r = None

    def train(self):
        n_iter = int(self.X.shape[0] / self.n_batch)
        indices = np.arange(self.X.shape[0])

        for ep in range(self.n_epoch):
            np.random.shuffle(indices)
            for i in tqdm(range(n_iter)):
                batch_indices = indices[
                    self.n_batch * i: self.n_batch * (i + 1)
                ]
                X_batch = self.X[batch_indices]
                y_batch = self.y[batch_indices]
                w_batch = self.sample_weight[batch_indices]

                y_hat = self.predict(X_batch)
                a = -2 * (y_batch - y_hat) * w_batch

                dL_db = (a * 1).mean()
                dL_dw = (self.der_w(X_batch, reduction=None).T * a) / a.shape
                dL_dv = (a * self.der_v(X_batch, reduction=None)).mean(axis=2)
                dL_dr = (a * self.der_r(X_batch, reduction=None)).mean(axis=2)
                self.update(dL_db, dL_dw, dL_dv, dL_dr)

        return self

    def der_w(self, X: csr_matrix, reduction="mean") -> csr_matrix:
        dw = X
        if reduction == "mean":
            dw = dw.mean(axis=0)

        return dw

    def der_v(self, X: csr_matrix, reduction="mean") -> np.ndarray:
        if self.contract_der_v is None:
            self.contract_der_v = contract_expression(
                "ni,if,fg,nj,jd,jg->idn",
                X.shape, self.m2f.shape, self.r.shape, X.shape, self.v.shape, self.m2f.shape
            )

        dv = self.contract_der_v(
            X.A, self.m2f, self.r, X.A, self.v, self.m2f
        )

        if reduction == "mean":
            dv = dv.mean(axis=2)
        return dv

    def der_r(self, X: csr_matrix, reduction="mean") -> np.ndarray:
        if self.contract_der_r is None:
            self.contract_der_r = contract_expression(
                "ni,id,if,fg,bj,jd,jg->fgn",
                X.shape, self.v.shape, self.m2f.shape, self.r_mask.shape,
                X.shape, self.v.shape, self.m2f.shape
            )

        dr = self.contract_der_r(
            X.A, self.v, self.m2f, self.r_mask,
            X.A, self.v, self.m2f
        )

        if reduction == "mean":
            dr = dr.mean(axis=2)

        return dr

    def constraint_r(self, r):
        return

    def update(self, dL_db, dL_dw, dL_dv, dL_dr):
        self.b -= dL_db * self.lr
        self.w -= dL_dw * self.lr
        self.v -= dL_dv * self.lr
        self.r -= (dL_dr * self.lr + self.lam_r * self.r)

    def predict(self, X: csr_matrix):
        if X.shape[0] != self.n_batch:
            y_hat = self.contract_predict = contract(
                "ni,id,if,fg,nj,jd,jg->n",
                X.A, self.v, self.m2f, self.r,
                X.A, self.v, self.m2f
            )

        else:
            if self.contract_predict is None:
                self.contract_predict = contract_expression(
                    "ni,id,if,fg,nj,jd,jg->n",
                    X.shape, self.v.shape, self.m2f.shape, self.r.shape,
                    X.shape, self.v.shape, self.m2f.shape
                )

            y_hat = self.contract_predict(
                X.A, self.v, self.m2f, self.r,
                X.A, self.v, self.m2f
            )
        return y_hat

    def convert_sparse(self, df_X: pd.DataFrame):
        for i, field in enumerate(self.fields):
            X_field = self.fields_dir[field]["encoder"].transform(
                df_X[[field]])
            if i == 0:
                X = csr_matrix(X_field)
            else:
                X = hstack([X, X_field])
        return csr_matrix(X)

    def __call__(self, df_X: pd.DataFrame, chunk_size=1024):
        X = self.convert_sparse(df_X)
        indices = np.arange(X.shape[0])
        n_splits = int(X.shape[0] / chunk_size)
        y_hat = np.array([])
        for chunk_indices in tqdm(np.array_split(indices, n_splits)):
            y_hat = np.r_[y_hat, self.predict(X[chunk_indices])]
        return y_hat
