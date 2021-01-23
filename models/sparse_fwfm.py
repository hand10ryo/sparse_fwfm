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

    def __init__(self, df_X: pd.DataFrame, df_y: pd.Series, sample_weight=None, optimizer: str = "SGD",
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
        if sample_weight is None:
            self.sample_weight = np.ones(df_X.shape[0])
        else:
            self.sample_weight = sample_weight

        self.dim = dim
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.ignore_interactions = ignore_interactions
        self.lam_w = lam_w
        self.lam_v = lam_v
        self.lam_r = lam_r
        self.optimizer = optimizer

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

        if self.optimizer == "Adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.t = 0
            self.bm = 0
            self.bv = 0
            self.wm = np.zeros(self.w.shape)
            self.wv = np.zeros(self.w.shape)
            self.vm = np.zeros(self.v.shape)
            self.vv = np.zeros(self.v.shape)
            self.rv = np.zeros(self.r.shape)
            self.rm = np.zeros(self.r.shape)

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
        X_A = X.A

        if self.contract_der_v is None:
            self.contract_der_v = contract_expression(
                "ni,if,fg,nj,jd,jg->idn",
                X.shape, self.m2f.shape, self.r.shape, X.shape, self.v.shape, self.m2f.shape
            )

        dv = self.contract_der_v(
            X_A, self.m2f, self.r, X_A, self.v, self.m2f
        )

        if reduction == "mean":
            dv = dv.mean(axis=2)
        return dv

    def der_r(self, X: csr_matrix, reduction="mean") -> np.ndarray:
        if self.contract_der_r is None:
            self.contract_der_r = contract_expression(
                "ni,id,if,fg,nj,jd,jg->fgn",
                X.shape, self.v.shape, self.m2f.shape, self.r_mask.shape,
                X.shape, self.v.shape, self.m2f.shape
            )

        X_A = X.A
        dr = self.contract_der_r(
            X_A, self.v, self.m2f, self.r_mask,
            X_A, self.v, self.m2f
        )

        if reduction == "mean":
            dr = dr.mean(axis=2)

        return dr

    def constraint_r(self, r):
        return

    def update(self, dL_db, dL_dw, dL_dv, dL_dr):
        if self.optimizer == "SGD":
            self.b -= dL_db * self.lr
            self.w -= dL_dw * self.lr
            self.v -= dL_dv * self.lr
            self.r -= (dL_dr * self.lr + self.lam_r * self.r)

        elif self.optimizer == "Adam":
            self.t += 1

            gb = dL_db
            self.bm = self.beta1 * self.bm + (1 - self.beta1) * gb
            self.bv = self.beta2 * self.bv + (1 - self.beta2) * gb ** 2
            bm_hat = self.bm / (1 - self.beta1 ** self.t)
            bv_hat = self.bv / (1 - self.beta2 ** self.t)
            self.b -= self.lr * bm_hat / (bv_hat ** 0.5 + 1e-10)

            gw = dL_dw + self.lam_w * self.w
            self.wm = self.beta1 * self.wm + (1 - self.beta1) * gw
            self.wv = self.beta2 * self.wv + (1 - self.beta2) * gw ** 2
            wm_hat = self.wm / (1 - self.beta1 ** self.t)
            wv_hat = self.wv / (1 - self.beta2 ** self.t)
            self.w -= self.lr * wm_hat / (wv_hat ** 0.5 + 1e-10)

            gv = dL_dv + self.lam_v * self.v
            self.vm = self.beta1 * self.vm + (1 - self.beta1) * gv
            self.vv = self.beta2 * self.vv + (1 - self.beta2) * gv ** 2
            vm_hat = self.vm / (1 - self.beta1 ** self.t)
            vv_hat = self.vv / (1 - self.beta2 ** self.t)
            self.v -= self.lr * vm_hat / (vv_hat ** 0.5 + 1e-10)

            gr = dL_dr + self.lam_r * self.r
            self.rm = self.beta1 * self.rm + (1 - self.beta1) * gr
            self.rv = self.beta2 * self.rv + (1 - self.beta2) * gr ** 2
            rm_hat = self.rm / (1 - self.beta1 ** self.t)
            rv_hat = self.rv / (1 - self.beta2 ** self.t)
            self.r -= self.lr * rm_hat / (rv_hat ** 0.5 + 1e-10)

        else:
            raise NotImplementedError

    def adam_update(self, param, param_m, param_v, grad, lam):
        g = grad + lam * param
        param_m = self.beta1 * param_m + (1 - self.beta1) * g
        param_v = self.beta2 * param_v + (1 - self.beta2) * g ** 2
        param_m_hat = param_m / (1 - self.beta1 ** self.t)
        param_v_hat = param_v / (1 - self.beta2 ** self.t)
        param -= self.lr * param_m_hat / (param_v_hat ** 0.5 + 1e-10)

    def predict(self, X: csr_matrix):
        X_A = X.A
        y_hat_1st = contract("ni,i->n", X_A, self.w)

        if X.shape[0] != self.n_batch:

            y_hat_2nd = contract(
                "ni,id,if,fg,nj,jd,jg->n",
                X_A, self.v, self.m2f, self.r,
                X_A, self.v, self.m2f
            )

        else:
            if self.contract_predict is None:
                self.contract_predict = contract_expression(
                    "ni,id,if,fg,nj,jd,jg->n",
                    X.shape, self.v.shape, self.m2f.shape, self.r.shape,
                    X.shape, self.v.shape, self.m2f.shape
                )

            y_hat_2nd = self.contract_predict(
                X_A, self.v, self.m2f, self.r,
                X_A, self.v, self.m2f
            )
        return self.b + y_hat_1st + y_hat_2nd

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
        chunk_size = self.n_batch
        X = self.convert_sparse(df_X)
        indices = np.arange(X.shape[0])
        n_splits = int(X.shape[0] / chunk_size)
        y_hat = np.array([])
        for chunk_indices in tqdm(np.array_split(indices, n_splits)):
            y_hat = np.r_[y_hat, self.predict(X[chunk_indices])]
        return y_hat


class ThirdOrderFwFM(Model):
    """ Field weighted Factorization Machines implemented by scipy.sparse.csr_matrix
    See
     - colaboratory notebook https://colab.research.google.com/drive/1AsOLL_7ON_Fl22rIJ3RngvLAxe5IOmBh?usp=sharing,
     - fwfm paper https://arxiv.org/pdf/1806.03514.pdf or
     - High order factorization machines https://papers.nips.cc/paper/2016/file/158fc2ddd52ec2cf54d3c161f2dd6517-Paper.pdf
    for details.

    requirements:
        pandas,
        numpy,
        sklearn,
        scipy,
        tqdm,
        opt_einsum
    """

    def __init__(self, df_X: pd.DataFrame, df_y: pd.Series, sample_weight=None, train: bool = True,
                 optimizer: str = "SGD", dim: int = 8, lr: float = 1e-3, n_epoch: int = 10, n_batch: int = 256,
                 ignore_2nd_interactions: list = [], ignore_3rd_interactions: list = [],
                 lam_w: float = 0, lam_v: float = 0, lam_r: float = 0, lam_u: float = 0):
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
        if sample_weight is None:
            self.sample_weight = np.ones(df_X.shape[0])
        else:
            self.sample_weight = sample_weight

        self.dim = dim
        self.lr = lr
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.ignore_2nd_interactions = ignore_2nd_interactions
        self.ignore_3rd_interactions = ignore_3rd_interactions
        self.lam_w = lam_w
        self.lam_v = lam_v
        self.lam_r = lam_r
        self.lam_u = lam_u
        self.optimizer = optimizer

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
        self.u = np.random.rand(
            len(self.fields), len(self.fields), len(self.fields)) / 10

        if self.optimizer == "Adam":
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.t = 0
            self.bm = 0
            self.bv = 0
            self.wm = np.zeros(self.w.shape)
            self.wv = np.zeros(self.w.shape)
            self.vm = np.zeros(self.v.shape)
            self.vv = np.zeros(self.v.shape)
            self.rv = np.zeros(self.r.shape)
            self.rm = np.zeros(self.r.shape)
            self.uv = np.zeros(self.u.shape)
            self.um = np.zeros(self.u.shape)

        self.r_mask = np.ones([len(self.fields), len(self.fields)])
        self.u_mask = np.ones(
            [len(self.fields), len(self.fields), len(self.fields)])

        for i in range(len(self.fields)):
            for j in range(i, len(self.fields)):
                self.r_mask[i, j] = 0

        for i in range(len(self.fields)):
            for j in range(i, len(self.fields)):
                for k in range(j, len(self.fields)):
                    self.u_mask[i, j, k] = 0

        for interaction in self.ignore_2nd_interactions:
            field_i, field_j = tuple(interaction)
            field_i_idx = self.fields_dir[field_i]["field_idx"]
            field_j_idx = self.fields_dir[field_j]["field_idx"]
            self.r_mask[field_i_idx, field_j_idx] = 0
            self.r_mask[field_j_idx, field_i_idx] = 0

        self.r = self.r * self.r_mask

        for interaction in self.ignore_3rd_interactions:
            field_i, field_j, field_k = tuple(interaction)
            field_i_idx = self.fields_dir[field_i]["field_idx"]
            field_j_idx = self.fields_dir[field_j]["field_idx"]
            field_k_idx = self.fields_dir[field_k]["field_idx"]
            self.u_mask[field_i_idx, field_j_idx, field_k_idx] = 0
            self.u_mask[field_i_idx, field_k_idx, field_j_idx] = 0
            self.u_mask[field_j_idx, field_i_idx, field_k_idx] = 0
            self.u_mask[field_j_idx, field_k_idx, field_i_idx] = 0
            self.u_mask[field_k_idx, field_i_idx, field_j_idx] = 0
            self.u_mask[field_k_idx, field_j_idx, field_i_idx] = 0

        self.u = self.u * self.u_mask

        self.m2f = np.zeros([X.shape[1], len(self.fields)])
        for i, field in enumerate(self.fields):
            self.m2f[np.arange(self.fields_dir[field]["start_idx"],
                               self.fields_dir[field]["end_idx"]), i] = 1

        self.X = csr_matrix(X)
        self.y = df_y.values

        self.contract_predict_1st = None
        self.contract_predict_2nd = None
        self.contract_predict_3rd = None
        self.contract_der_v_2nd = None
        self.contract_der_v_3rd = None
        self.contract_der_r = None
        self.contract_der_u = None

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
                dL_du = (a * self.der_u(X_batch, reduction=None)).mean(axis=3)
                self.update(dL_db, dL_dw, dL_dv, dL_dr, dL_du)

        return self

    def der_w(self, X: csr_matrix, reduction="mean") -> csr_matrix:
        dw = X
        if reduction == "mean":
            dw = dw.mean(axis=0)

        return dw

    def der_v(self, X: csr_matrix, reduction="mean") -> np.ndarray:
        X_A = X.A

        if self.contract_der_v_2nd is None:
            self.contract_der_v_2nd = contract_expression(
                "ni,if,fg,nj,jd,jg->idn",
                X.shape, self.m2f.shape, self.r.shape, X.shape, self.v.shape, self.m2f.shape
            )

        dv_2nd = self.contract_der_v_2nd(
            X_A, self.m2f, self.r, X.A, self.v, self.m2f
        )

        if self.contract_der_v_3rd is None:
            self.contract_der_v_3rd = contract_expression(
                "ni,if,fgh,nj,jd,jg,nk,kd,jh->idn",
                X.shape, self.m2f.shape, self.u.shape,
                X.shape, self.v.shape, self.m2f.shape,
                X.shape, self.v.shape, self.m2f.shape,
            )

        dv_3rd = self.contract_der_v_3rd(
            X_A, self.m2f, self.u,
            X_A, self.v, self.m2f,
            X_A, self.v, self.m2f
        )

        dv = dv_2nd + dv_3rd

        if reduction == "mean":
            dv = dv.mean(axis=2)
        return dv

    def der_r(self, X: csr_matrix, reduction="mean") -> np.ndarray:
        X_A = X.A

        if self.contract_der_r is None:
            self.contract_der_r = contract_expression(
                "ni,id,if,nj,jd,jg,fg->fgn",
                X.shape, self.v.shape, self.m2f.shape,
                X.shape, self.v.shape, self.m2f.shape,
                self.r_mask.shape
            )

        dr = self.contract_der_r(
            X_A, self.v, self.m2f,
            X_A, self.v, self.m2f,
            self.r_mask
        )

        if reduction == "mean":
            dr = dr.mean(axis=2)

        return dr

    def der_u(self, X: csr_matrix, reduction="mean") -> np.ndarray:
        X_A = X.A

        if self.contract_der_u is None:
            self.contract_der_u = contract_expression(
                "ni,id,if,bj,jd,jg,bj,jd,jg,fgh->fghn",
                X.shape, self.v.shape, self.m2f.shape,
                X.shape, self.v.shape, self.m2f.shape,
                X.shape, self.v.shape, self.m2f.shape,
                self.u_mask.shape,
            )

        du = self.contract_der_u(
            X_A, self.v, self.m2f,
            X_A, self.v, self.m2f,
            X_A, self.v, self.m2f,
            self.u_mask
        )

        if reduction == "mean":
            du = du.mean(axis=3)

        return du

    def update(self, dL_db, dL_dw, dL_dv, dL_dr, dL_du):
        if self.optimizer == "SGD":
            self.b -= dL_db * self.lr
            self.w -= (dL_dw * self.lr + self.lam_w * self.w)
            self.v -= (dL_dv * self.lr + self.lam_v * self.v)
            self.r -= (dL_dr * self.lr + self.lam_r * self.r)
            self.u -= (dL_du * self.lr + self.lam_u * self.u)

        elif self.optimizer == "Adam":
            self.t += 1
            gb = dL_db
            self.bm = self.beta1 * self.bm + (1 - self.beta1) * gb
            self.bv = self.beta2 * self.bv + (1 - self.beta2) * gb ** 2
            bm_hat = self.bm / (1 - self.beta1 ** self.t)
            bv_hat = self.bv / (1 - self.beta2 ** self.t)
            self.b -= self.lr * bm_hat / (bv_hat ** 0.5 + 1e-10)

            gw = dL_dw + self.lam_w * self.w
            self.wm = self.beta1 * self.wm + (1 - self.beta1) * gw
            self.wv = self.beta2 * self.wv + (1 - self.beta2) * gw ** 2
            wm_hat = self.wm / (1 - self.beta1 ** self.t)
            wv_hat = self.wv / (1 - self.beta2 ** self.t)
            self.w -= self.lr * wm_hat / (wv_hat ** 0.5 + 1e-10)

            gv = dL_dv + self.lam_v * self.v
            self.vm = self.beta1 * self.vm + (1 - self.beta1) * gv
            self.vv = self.beta2 * self.vv + (1 - self.beta2) * gv ** 2
            vm_hat = self.vm / (1 - self.beta1 ** self.t)
            vv_hat = self.vv / (1 - self.beta2 ** self.t)
            self.v -= self.lr * vm_hat / (vv_hat ** 0.5 + 1e-10)

            gr = dL_dr + self.lam_r * self.r
            self.rm = self.beta1 * self.rm + (1 - self.beta1) * gr
            self.rv = self.beta2 * self.rv + (1 - self.beta2) * gr ** 2
            rm_hat = self.rm / (1 - self.beta1 ** self.t)
            rv_hat = self.rv / (1 - self.beta2 ** self.t)
            self.r -= self.lr * rm_hat / (rv_hat ** 0.5 + 1e-10)

            gu = dL_du + self.lam_u * self.u
            self.um = self.beta1 * self.um + (1 - self.beta1) * gu
            self.uv = self.beta2 * self.uv + (1 - self.beta2) * gu ** 2
            um_hat = self.um / (1 - self.beta1 ** self.t)
            uv_hat = self.uv / (1 - self.beta2 ** self.t)
            self.u -= self.lr * um_hat / (uv_hat ** 0.5 + 1e-10)

    def predict(self, X: csr_matrix):
        X_A = X.A
        if X.shape[0] != self.n_batch:
            y_hat_1st = contract("ni,i->n", X_A, self.w)

            y_hat_2nd = contract(
                "ni,id,if,nj,jd,jg,fg->n",
                X_A, self.v, self.m2f,
                X_A, self.v, self.m2f,
                self.r
            )

            y_hat_3rd = contract(
                "ni,id,if,nj,jd,jg,nk,kd,kh,fgh->n",
                X_A, self.v, self.m2f,
                X_A, self.v, self.m2f,
                X_A, self.v, self.m2f,
                self.u
            )

        else:
            if self.contract_predict_1st is None:
                self.contract_predict_1st = contract_expression(
                    "ni,i->n", X.shape, self.w.shape
                )

            y_hat_1st = self.contract_predict_1st(X_A, self.w)

            if self.contract_predict_2nd is None:

                self.contract_predict_2nd = contract_expression(
                    "ni,id,if,nj,jd,jg,fg->n",
                    X.shape, self.v.shape, self.m2f.shape,
                    X.shape, self.v.shape, self.m2f.shape,
                    self.r.shape
                )

            y_hat_2nd = self.contract_predict_2nd(
                X_A, self.v, self.m2f,
                X_A, self.v, self.m2f,
                self.r
            )

            if self.contract_predict_3rd is None:
                self.contract_predict_3rd = contract_expression(
                    "ni,id,if,nj,jd,jg,nk,kd,kh,fgh->n",
                    X.shape, self.v.shape, self.m2f.shape,
                    X.shape, self.v.shape, self.m2f.shape,
                    X.shape, self.v.shape, self.m2f.shape,
                    self.u.shape
                )

            y_hat_3rd = self.contract_predict_3rd(
                X_A, self.v, self.m2f,
                X_A, self.v, self.m2f,
                X_A, self.v, self.m2f,
                self.u
            )

        return self.b + y_hat_1st + y_hat_2nd + y_hat_3rd

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
        chunk_size = self.n_batch
        X = self.convert_sparse(df_X)
        indices = np.arange(X.shape[0])
        n_splits = int(X.shape[0] / chunk_size)
        y_hat = np.array([])
        for chunk_indices in tqdm(np.array_split(indices, n_splits)):
            y_hat = np.r_[y_hat, self.predict(X[chunk_indices])]
        return y_hat
