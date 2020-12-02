import warnings
import pickle
import numpy as np
import numpy.ma as ma
from scipy import stats, sparse, optimize
from sklearn import linear_model
from sklearn.metrics import pairwise
import textwrap
from tabulate import tabulate
from typing import Dict, Tuple, Optional

import utils

Array = np.array
OptionalFloat = Optional[float]
ParamDict = Dict[str, OptionalFloat]


# __all__ = ['Locations', 'binomial_pvalues', 'CPC', 'CPL']


class Locations(object):
    """
    Locations in 2D space with relevance and community attributes.

    The model represents nodes of a network embedded in 2D space, and provides
    methods for calculating human mobility models. The relevance attribute
    captures for example the population at each location (or some other measure
    of intrinsic strenght). The community attribute is currently optional but
    the model will be extended to make full use of it.

    Attributes
    ----------
    TODO: Update below
    N : int
        Number of nodes.
    dmat : np.array
        The matrix of pariwise distances between locations.
    relevance : np.array
        A measure of the relevance or intrinsic strength of the locations.
    in_relevance : np.array
        A second measure of the relevance (to be used as the in-attraction
        whereas relevance is the out-attraction).
    k : int
        The number of communities.
    comm : np.array
        The community of the nodes.
    data: np.array or None
        NxN matrix that hold data observations.

    """

    def __init__(self, N: int,
                 location_mat: Array,
                 orig_relvec: Array,
                 dest_relvec: Array = None,
                 comm_vec: Array = None,
                 use_coords: bool = False) -> None:
        """
        Instantiate the Locations object.

        Parameters
        ----------
        N : int
            The number of locations.
        location_mat : array_like
            The NxN matrix of pairwise distances between locations.
            Alternatively, if `use_coords` is set to True this should be the Nx2
            array of coordinates.
        orig_relvec : array_like
            The primary relevance vector of each location (for example the
            population). If `dest_relvec` is also provided then this vector is
            taken to be the origin relevance.
        dest_relvec : array_like or None, optional
            Provided to distintiguish two sets of intrinsic measures of relevance
            for each location (default is None). If provided, this vector represents
            the destination relevance while `orig_relvec` is origin relevance.
        comm_vec : array_like or None, optional
            A vector containing the community (block) assignment of each location
            (default is None). The community methods will be extended in the
            future.
        use_coords : bool, optional
            If set to True, `location_mat` should correspond to the Nx2 array of
            coordinates and the class uses the pairwise euclidean distance
            between locations.

        """
        assert (location_mat.shape == (N, 2) and use_coords) \
            or location_mat.shape == (N, N), \
            'location_mat is not NxN (distances), or Nx2 (coords) with `use_coords=True`'
        assert len(orig_relvec) == N, 'orig_relvec should have length N'

        if use_coords:
            self.dmat = pairwise.euclidean_distances(location_mat)
        else:
            assert not np.any(location_mat < 0), \
                'distance_matrix should be non_negative'
            self.dmat = location_mat

        self.N = N
        self.orig_rel = np.array(orig_relvec)  # prefer arr to list or Series
        self._data = None
        # self._datasparseflag = False  # Currently not used

        assert np.all(self.orig_rel > 0), \
            'origin relevance of locations should be positive'

        if dest_relvec is None:
            self.dest_rel = self.orig_rel  # binding to avoid duplicate data
            self._tworelevancesflag = False
        else:
            assert len(dest_relvec) == N, \
                'dest_relvec should have length N'
            self.dest_rel = np.array(dest_relvec)  # prefer arr
            self._tworelevancesflag = True
            assert np.all(self.dest_rel > 0), \
                'destination relevance of locations should be positive'

        if comm_vec is None:
            self.k = 1
            self.comm = np.ones(N, dtype=int)
        else:
            # self.k = len(np.unique(comm_vec))  #TODO:check below works better
            self.k = len(set(comm_vec))
            self.comm = np.array(comm_vec)

        return None

    def __str__(self) -> str:
        if self._tworelevancesflag:
            s = f"""
                Locations object (orig. and dest. relevance provided):
                    N={self.N} and k={self.k}
                """
        else:
            s = f'Locations object: N={self.N} and k={self.k}'

        if self.data is not None:
            s = s + '\n -- Data has been set'

        return textwrap.dedent(s)

    @property
    def data(self):
        """Retrieve the data stored inside the object."""
        return self._data

    @data.setter
    def data(self, data_mat: Array):
        """Set a copy of the data inside the object."""
        assert (data_mat < 0).sum() == 0, f'data matrix should be non-negative'
        # Test above works for dense and sparse matrices
        assert data_mat.shape == (self.N, self.N), 'data_mat is not NxN'

        # NB: We remove the diagonal from the data matrix BEFORE calculating
        # outflow and inflow
        if sparse.issparse(data_mat):
            # self._datasparseflag = True
            self._data = utils.sparsemat_remove_diag(data_mat)  # creates a copy
        else:
            self._data = data_mat.copy()
            self._data[np.diag_indices(self.N)] = 0

        # Row and column sums for dense and sparse matrices
        outflow = np.asarray(self._data.sum(axis=1)).flatten()
        inflow = np.asarray(self._data.sum(axis=0)).flatten()

        self.data_out = outflow
        self.data_in = inflow

        return None

    def io_matrix(self, threshold: float = np.inf, **kwargs) -> Array:
        """
        Calculate intervening opportunities matrix summing the dest. relevance.

        Parameters
        ----------
        threshold : float, optional
            Maximum radius to count locations as intervening opportunities
            (the default is np.inf).

        Returns
        -------
        np.array
            NxN intervening opportunities matrix.

        """
        S = np.zeros((self.N, self.N), dtype=self.dest_rel.dtype.type)

        # We use masked arrays
        masked_dmat = ma.masked_greater_equal(self.dmat, threshold)
        idx_mat = ma.argsort(masked_dmat, axis=1)
        # sort each row; masked elements are ignored but sorted last

        masked_rel = ma.masked_array(self.dest_rel)

        for i, idx in enumerate(idx_mat):
            masked_rel.mask = masked_dmat[i].mask
            masked_rel[i] = ma.masked        # origin is not used
            masked_rel[idx[-1]] = ma.masked  # the furthest point is not used

            rolled_idx = np.roll(idx, 1)
            S[i, idx] = masked_rel[rolled_idx].cumsum()

        return S

    def radiation_matrix(self, threshold: float = np.inf,
                         finite_correction: bool = True,
                         **kwargs) -> Array:
        """
        Calculate the radiation matrix.

        Parameters
        ----------
        threshold : float, optional
            Maximum radius to count locations as intervening opportunities
            (the default is np.inf).
        finite_correction :
            Wether to normalise each row using the term $1/(1 - m_i / M)$.

        Returns
        -------
        np.array
            NxN unconstrained radiation matrix.

        """
        S = self.io_matrix(threshold=threshold)
        # m = self.orig_rel[:, np.newaxis]  # column vector
        m = self.dest_rel[:, np.newaxis]  # column vector
        n = self.dest_rel[:, np.newaxis]  # column vector

        num = m * n.T
        den = (m + S) * (m + n.T + S)
        with np.errstate(invalid='ignore'):
            # 0/0 inside arrays is invalid floating point, not divide error;
            # to cover the case m_i + S_ij = 0 (can occur when m_i is zero)
            out = num / den
            out[np.isnan(out)] = 0.0

        out[np.diag_indices_from(out)] = 0.0

        if finite_correction:
            M = np.sum(m)
            norm = (1 - m / M)  # column vector
            out = out / norm  # each row is scaled

        return out

    def gravity_matrix(self, γ: float,
                       α: float = 1.0, β: float = 1.0, **kwargs) -> Array:
        """
        Calculate the gravity matrix.

        Parameters
        ----------
        γ : float
            Decay exponent.
        α, β : float, optional
            Respetively the power of the origin and destination terms.

        Returns
        -------
        np.array
            NxN unconstrained gravity matrix.

        """
        m = self.orig_rel[:, np.newaxis]  # column vector
        n = self.dest_rel[:, np.newaxis]  # column vector

        with np.errstate(divide='ignore', invalid='raise'):
            # 0**(-γ) raises divide error, 0 * ∞ raises invalid error in multiply
            dmat_power = self.dmat ** (-γ)
            dmat_power[np.diag_indices(self.N)] = 0.0
            # note this doesn't fix problems outside diagonal, this is on purpose

            out = (m ** α) * (n.T ** β) * dmat_power

        return out

    def gravity_calibrate_nonlinear(self, constraint_type: str,
                                    maxiters=500,
                                    use_log=False,
                                    verbose: bool = False) -> Tuple[float, float]:
        """
        Calibrate the gravity model using nonlinear least squares.

        Parameters
        ----------
        constraint_type : str

        Returns
        -------
        float, float

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        assert constraint_type in ['unconstrained', 'production', 'attraction', 'doubly'], \
            f'invalid constraint {constraint_type}'

        # The observations
        i, j = self.data.nonzero()
        y = np.asarray(self.data[i, j]).flatten()

        if constraint_type == 'unconstrained':
            x0 = [2, 1, 1]
            bounds = ([-np.inf, 0, 0], [np.inf, np.inf, np.inf])

            def cost_fun(x):  # gamma, alpha, beta
                fmat = self.gravity_matrix(x[0], α=x[1], β=x[2])
                K = self.data.sum() / fmat.sum()
                T_model = K * fmat
                if use_log:
                    out = np.log(y) - np.log(T_model[i, j])
                else:
                    out = y - T_model[i, j]
                return out

        elif constraint_type == 'production':
            x0 = [1, 1]
            bounds = ([-np.inf, 0], [np.inf, np.inf])

            def cost_fun(x):  # gamma, beta
                fmat = self.gravity_matrix(x[0], α=0, β=x[1])
                pmat = self.probability_matrix(fmat, constraint_type)
                T_model = self.data_out[:, np.newaxis] * pmat
                if use_log:
                    out = np.log(y) - np.log(T_model[i, j])
                else:
                    out = y - T_model[i, j]
                return out

        elif constraint_type == 'attraction':
            x0 = [1, 1]
            bounds = ([-np.inf, 0], [np.inf, np.inf])

            def cost_fun(x):  # gamma, alpha
                fmat = self.gravity_matrix(x[0], α=x[1], β=0)
                pmat = self.probability_matrix(fmat, constraint_type)
                T_model = pmat * self.data_in[np.newaxis, :]
                if use_log:
                    out = np.log(y) - np.log(T_model[i, j])
                else:
                    out = y - T_model[i, j]
                return out

        elif constraint_type == 'doubly':
            x0 = [2]
            bounds = (-np.inf, np.inf)

            def cost_fun(x):  # gamma
                fmat = self.gravity_matrix(x[0], α=0, β=0)
                T_model = simple_ipf(fmat, self.data_out, self.data_in,
                                     maxiters=maxiters,
                                     verbose=verbose)
                if use_log:
                    out = np.log(y) - np.log(T_model[i, j])
                else:
                    out = y - T_model[i, j]
                return out

        res = optimize.least_squares(cost_fun, x0, bounds=bounds)
        st = res.status
        assert st > 0, f'optimization exit status is failure'

        st_dict = {
            -1: 'improper input parameters status returned from MINPACK.',
            0: 'the maximum number of function evaluations is exceeded.',
            1: 'gtol termination condition is satisfied.',
            2: 'ftol termination condition is satisfied.',
            3: 'xtol termination condition is satisfied.',
            4: 'Both ftol and xtol termination conditions are satisfied.'
        }

        if verbose:
            print('Status: ', st_dict[st])

        return res.x

    def gravity_calibrate_cpc(self, constraint_type: str,
                              maxiters=500,
                              verbose: bool = False) -> Tuple[float, float]:
        """
        Calibrate the gravity model using CPC minisation

        Parameters
        ----------
        constraint_type : str

        Returns
        -------
        float, float

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        assert constraint_type in ['unconstrained', 'production', 'attraction', 'doubly'], \
            f'invalid constraint {constraint_type}'

        if constraint_type == 'unconstrained':
            x0 = [2, 1, 1]
            bounds = [(0, None)] * 3

            def cost_fun(x):  # gamma, alpha, beta
                fmat = self.gravity_matrix(x[0], α=x[1], β=x[2])
                K = self.data.sum() / fmat.sum()
                T_model = K * fmat
                return 1 - CPC(T_model, self.data)

        elif constraint_type == 'production':
            x0 = [1, 1]
            bounds = [(0, None)] * 2

            def cost_fun(x):  # gamma, beta
                fmat = self.gravity_matrix(x[0], α=0, β=x[1])
                pmat = self.probability_matrix(fmat, constraint_type)
                T_model = self.data_out[:, np.newaxis] * pmat
                return 1 - CPC(T_model, self.data)

        elif constraint_type == 'attraction':
            x0 = [1, 1]
            bounds = [(0, None)] * 2

            def cost_fun(x):  # gamma, alpha
                fmat = self.gravity_matrix(x[0], α=x[1], β=0)
                pmat = self.probability_matrix(fmat, constraint_type)
                T_model = pmat * self.data_in[np.newaxis, :]
                return 1 - CPC(T_model, self.data)

        elif constraint_type == 'doubly':
            x0 = [2]
            bounds = [(0, None)]

            def cost_fun(x):  # gamma
                fmat = self.gravity_matrix(x[0], α=0, β=0)
                T_model = simple_ipf(fmat, self.data_out, self.data_in,
                                     maxiters=maxiters,
                                     verbose=verbose)
                return 1 - CPC(T_model, self.data)

        res = optimize.minimize(cost_fun, x0, bounds=bounds)
        # assert res.sucess, f'optimization exit status is failure'

        if verbose:
            print('Status: ', res.message)

        return res.x

    def gravity_calibrate_all(self, verbose: bool = False,
                              **kwargs) -> Tuple[float, float, float]:
        """
        Calibrate the all of the gravity parameters using linear least squares.

        Parameters
        ----------
        verbose: bool, optional
            The defalt is False

        Returns
        -------
        γ, α, β : float

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        N = self.N

        # The observations
        i, j = self.data.nonzero()
        y = np.asarray(self.data[i, j]).flatten()

        # The model data
        m = self.data_out[:, np.newaxis]  # col. vec.
        n = self.data_in[np.newaxis, :]   # row vec.

        m_repeated = np.repeat(m, N, axis=1)
        n_repeated = np.repeat(n, N, axis=0)

        X = np.vstack((m_repeated[i, j], n_repeated[i, j], self.dmat[i, j])).T
        logX = np.log(X)

        # if verbose:
        #     ρ, _ = stats.pearsonr(logX[:, 0], logX[:, 1])
        #     print(f'Correlation between log O and D columns, ρ : {ρ:.3f}\n')

        reg = linear_model.LinearRegression()
        reg.fit(logX, np.log(y))

        α, β, γ = reg.coef_
        γ = -γ
        # k = np.exp(reg.intercept_)  # in practice works worse than unconstrained

        if γ < 0.5:
            warnings.warn(f'γ = {γ:.3f}')

        return γ, α, β

    def probability_matrix(self, f_mat: Array, constraint_type: str) -> Array:
        """
        Normalise the affinity matrix f_ij to get a probability.

        Parameters
        ----------
        f_mat : array_like
            A matrix that constains the flows predicted by one of the models.
        constraint_type  : str
            The type of constraint to use. Only 'production' or 'attraction'
            make sense in this context.

        Returns
        -------
        np.array
            NxN normalised probability matrix with either rows or colums that
            sum to one.

        """
        assert constraint_type in ['production', 'attraction'], \
            f'invalid constraint {constraint_type}'

        assert not sparse.issparse(f_mat), \
            'sparse matrices are not supported'

        p_mat = f_mat.astype(float)  # to avoid problems with division

        if constraint_type == 'production':
            # We assume that f_mat is NOT of type  sparse so that sums are flat
            row_sum = f_mat.sum(axis=1)
            idx = (row_sum > 0)
            p_mat[idx] = p_mat[idx] / row_sum[idx, np.newaxis]

        elif constraint_type == 'attraction':
            col_sum = f_mat.sum(axis=0)
            idx = (col_sum > 0)
            p_mat[:, idx] = p_mat[:, idx] / col_sum[np.newaxis, idx]

        return p_mat

    def draw_multinomial(self, p_mat, constraint_type, seed=0):
        """Draw from the constrained model using multinomial distribution."""
        assert constraint_type in ['production', 'attraction'], \
            f'invalid constraint {constraint_type}'

        rng = np.random.RandomState(seed)

        out = sparse.lil_matrix((self.N, self.N), dtype=int)

        if constraint_type == 'production':
            for i in range(self.N):
                draw = rng.multinomial(self.data_out[i], p_mat[i])
                j, = draw.nonzero()
                out[i, j] = draw[j]

        elif constraint_type == 'attraction':
            for j in range(self.N):
                draw = rng.multinomial(self.data_in[j], p_mat[:, j])
                i, = draw.nonzero()
                out[i, j] = draw[i]

        return out.tocsr()

    def score_draws(self, pmat, constraint_type, nb_repeats=20):
        """Calculate goodness-of-fit measures for multiple draws."""
        out = np.zeros((nb_repeats, 3))

        for k in range(nb_repeats):
            T = self.draw_multinomial(pmat, constraint_type, seed=k)
            out[k] = CPC(T, self.data), CPL(T, self.data), RMSE(T, self.data)

        return out

    def constrained_model(self, f_mat: Array,
                          constraint_type: str,
                          maxiters=500,
                          verbose=False) -> Array:
        """
        Calculate the constrained flux from the affinity matrix f_ij.

        Parameters
        ----------
        f_mat : array_like
        constrained : {'unconstrained', 'production', 'attraction', 'doubly'}

        Returns
        -------
        np.array
            NxN constrained flow matrix.

        """
        if self.data is None:
            raise DataNotSet('the data for the constraints is needed')

        assert constraint_type in ['unconstrained', 'production', 'attraction',
                                   'doubly'], \
            f'invalid constraint {constraint_type}'

        if constraint_type == 'unconstrained':
            K = self.data.sum() / f_mat.sum()
            out = K * f_mat

        elif constraint_type == 'production':
            p_mat = self.probability_matrix(f_mat, constraint_type)
            out = self.data_out[:, np.newaxis] * p_mat

        elif constraint_type == 'attraction':
            p_mat = self.probability_matrix(f_mat, constraint_type)
            out = p_mat * self.data_in[np.newaxis, :]

        elif constraint_type == 'doubly':
            out = simple_ipf(f_mat, self.data_out, self.data_in,
                             maxiters=maxiters,
                             verbose=verbose)

        return out

    # def constrained_model(self, model: str,
    #                       constraint_type: str,
    #                       params: ParamDict,
    #                       rounded: bool = False,
    #                       sparse: bool = False) -> Array:
    #     """
    #     Calculate the prediction of a model and apply the constraint type.
    #
    #     Parameters
    #     ----------
    #     model : {'gravity', 'radiation', 'io'}
    #     params : dict
    #     rounded : bool, optional
    #         Whether to round the output of the model or not (default is False).
    #     sparse : bool, optional
    #         Whether to return the output model as a sparse matrix (default is
    #         False). If set to True, the output is rounded irrespective of the
    #         kwarg `rounded` which is ignored.
    #
    #     Returns
    #     -------
    #     np.array
    #
    #     """
    #     method_name = f'{model}_matrix'
    #     method = getattr(self, method_name, None)
    #     if method:
    #         try:
    #             f_mat = method(**params)
    #         except Exception as e:
    #             message = f'Failed to apply model {model} with parameter {params}'
    #             raise ValueError(message) from e
    #     else:
    #         raise NotImplementedError(f'model {model} not implemented')
    #
    #     try:
    #         ϕ = self.constrained_flux(f_mat, constraint_type)
    #     except Exception as e:
    #         message = f'Failed to apply constraint type {model}'
    #         raise ValueError(message) from e
    #
    #     if rounded or sparse:
    #         ϕ = np.round(ϕ).astype(int)
    #     if sparse:
    #         ϕ = sparse.csr_matrix(ϕ)
    #
    #     return ϕ

    # def CPC_from_model(self, model: str,
    #                    constraint_type: str,
    #                    params: ParamDict) -> float:
    #     """
    #     Use a particular model and constraint type to calculate the CPC fit.
    #
    #     Parameters
    #     ----------
    #     model : {'gravity', 'radiation', 'io'}
    #     params : dict
    #
    #     Returns
    #     -------
    #     float
    #
    #     """
    #     Data = self.data
    #     return CPC(Data, self.constrained_model(model, constraint_type, params))
    #

    def pvalues_approx(self, pmat: Array, constraint_type: str) -> Array:
        """
        Calculate the p-values using the normal approximation.

        Parameters
        ----------
        pmat : Array
        constraint_type : {'production', 'attraction'}

        Returns
        -------
        Array
            A Mx2 array where M is the number of nonzero-edges. The first column
            stores the p-values for the hypothesis "the observation is not
            signifcantly larger than the predicted mean" and the second column
            for the hypothesis "the observation is not signifcantly smaller than
            the predicted mean."

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        assert constraint_type in ['production', 'attraction'], \
            f'invalid constraint {constraint_type}'

        ii, jj = self.data.nonzero()

        # Entry-wise first and second moments (binomial model)
        Data = self.data
        if constraint_type == 'production':
            Exp = self.data_out[:, np.newaxis] * pmat
        elif constraint_type == 'attraction':
            Exp = pmat * self.data_in[np.newaxis, :]
        Std = np.sqrt(Exp * (1 - pmat))

        Z_score = (Data[ii, jj] - Exp[ii, jj]) / Std[ii, jj]
        Z_score = np.asarray(Z_score).flatten()

        plus = stats.norm.cdf(-Z_score)
        minus = stats.norm.cdf(Z_score)

        return np.vstack((plus, minus)).T

    def pvalues_exact(self, pmat: Array, constraint_type: str) -> Array:
        """
        Calculate the p-values using the exact binomial distributions.

        Parameters
        ----------
        pmat : Array
        constraint_type : {'production', 'attraction'}

        Returns
        -------
        Array
            A Mx2 array where M is the number of nonzero-edges. The first column
            stores the p-values for the hypothesis "the observation is not
            signifcantly larger than the predicted mean" and the second column
            for the hypothesis "the observation is not signifcantly smaller than
            the predicted mean."

        """
        if self.data is None:
            raise DataNotSet('the data for comparison is needed')

        assert constraint_type in ['production', 'attraction'], \
            f'invalid constraint {constraint_type}'

        ii, jj = self.data.nonzero()
        n = len(ii)
        out = np.zeros((n, 2))

        # The target N is either the row or the column sum
        Ns = self.data_out[ii] \
            if constraint_type == 'production' \
            else self.data_in[jj]

        for k in range(n):
            i, j = ii[k], jj[k]
            out[k] = _binomial_pvalues(Ns[k], pmat[i, j], self.data[i, j])

        return out

    def significant_edges(self, pmat: Array,
                          constraint_type: str,
                          significance: float = 0.01,
                          exact: bool = False,
                          verbose: bool = False) -> Tuple[Array, Array]:
        """
        Calculate the significant edges according to a binomial test (or z-test).

        Parameters
        ----------
        pmat : Array
        constraint_type : {'production', 'attraction'}
        significance : float, optional
            By default 0.01.
        exact : bool, optional
            Whether to use the exact calculation as opposed the normal
            approximation (z-test). The default is False.
        verbose : bool, optional

        Returns
        -------
        edges

        """
        method_name = 'pvalues_' + ('exact' if exact else 'approx')
        method = getattr(self, method_name)
        pvals = method(pmat, constraint_type)

        significant = pvals < significance
        idx_plus = significant[:, 0]
        idx_minus = significant[:, 1]

        if verbose:
            n, nplus, nminus = len(pvals), np.sum(idx_plus), np.sum(idx_minus)
            nzero = n - nplus - nminus
            tab = [['Positive (observed larger)', nplus, f'{100*nplus/n:.2f}'],
                   ['Negative (model larger)', nminus, f'{100*nminus/n:.2f}'],
                   ['Not-significant:', nzero, f'{100*nzero/n:.2f}'],
                   ['Total', n, '100.00']]
            print(tabulate(tab, headers=['', 'Nb', '%']))

        return idx_plus, idx_minus


class DataLocations(Locations):

    def __init__(self, location_mat, data_mat):
        N, _ = data_mat.shape
        assert (data_mat < 0).sum() == 0, f'data matrix should be non-negative'
        assert data_mat.shape == (N, N), 'data_mat is not NxN'

        # NB: We remove the diagonal from the data matrix BEFORE calculating
        # outflow and inflow
        if sparse.issparse(data_mat):
            input_mat = utils.sparsemat_remove_diag(data_mat)  # creates a copy
        else:
            input_mat = data_mat.copy()
            input_mat[np.diag_indices(N)] = 0

        # Row and column sums for dense and sparse matrices
        outflow = np.asarray(input_mat.sum(axis=1)).flatten()
        inflow = np.asarray(input_mat.sum(axis=0)).flatten()

        coords = True if location_mat.shape == (N, 2) else False
        super().__init__(N, location_mat, outflow,
                         inflow, use_coords=coords)
        self.data = data_mat

        return None


# Outside the classes
def _binomial_pvalues(N: int, p: float, x: int) -> Tuple[float, float]:
    """
    Calculate a p-value according to the binomial distribution.

    Parameters
    ----------
    N, p : int, float
        The parameters of the binomial distribution.
    x : int
        The observation.

    Returns
    -------
    float, float

    """
    # N, x = int(N), int(x)
    B = stats.binom(N, p)

    # It would be useful to assume x > 0; that case is generally covered
    # when I call using T_data.nonzero(). The other limiting case is when
    # x == N
    if x == N:
        out = p**N, 1.0

    # Test below seems is slower (2m23 vs 1m35 for UK commuting data)
    # elif x < N // 2:
    #     probas = B.pmf(range(x + 1)).cumsum()
    #     out = 1 - round(probas[x - 1], 6), round(probas[x], 6)
    # else:
    #     probas = B.pmf(range(N, x - 1, -1)).cumsum()
    #     out = round(probas[-1], 6), 1 - round(probas[-2], 6)

    else:
        probas = B.pmf(range(x + 1)).cumsum()
        out = 1 - round(probas[x - 1], 6), round(probas[x], 6)
        # due to numerical errors I can get cumsum larger than 1 and this is why
        # we need to use rounding

    return out


def CPC(F1: Array, F2: Array, rel_tol: float = 1e-3) -> float:
    """
    Calculate the common part of commuters between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    x, y = F1.sum(), F2.sum()
    assert abs((x - y) / x) < rel_tol, 'arrays do not have same sum (up to rel. tol.)'
    # assert np.isclose(F1.sum(), F2.sum()), 'arrays should have same sum'

    denom_mat = F1 + F2
    num_mat = denom_mat - np.abs(F1 - F2)  # this implements 2*min(F1, F2)

    return np.sum(num_mat) / np.sum(denom_mat)


def CPL(F1: Array, F2: Array, rel_tol: float = 1e-3) -> float:
    """
    Calculate the common part of links between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    x, y = F1.sum(), F2.sum()
    assert abs((x - y) / x) < rel_tol, 'arrays do not have same sum (up to rel. tol.)'
    # assert np.isclose(F1.sum(), F2.sum()), 'arrays should have same sum'

    bool_F1 = F1 > 0
    bool_F2 = F2 > 0

    # sparse implements np.matrix and * is mat-mat multplication, not elementwise
    if sparse.issparse(F1):
        prod = bool_F1.multiply(bool_F2)
    elif sparse.issparse(F2):
        prod = bool_F2.multiply(bool_F1)
    else:
        prod = bool_F1 * bool_F2

    return 2 * np.sum(prod) / (np.sum(bool_F1) + np.sum(bool_F2))


def RMSE(F1: Array, F2: Array, rel_tol: float = 1e-3) -> float:
    """
    Calculate root-mean-square error between two models.

    Parameters
    ----------
    F1, F2 : array_like

    Returns
    -------
    float

    """
    x, y = F1.sum(), F2.sum()
    assert abs((x - y) / x) < rel_tol, 'arrays do not have same sum (up to rel. tol.)'
    # assert np.isclose(F1.sum(), F2.sum()), 'arrays should have same sum'

    N = np.prod(F1.shape)

    diff = F1 - F2
    power = diff.power(2) if sparse.issparse(diff) else np.power(diff, 2)

    return np.sqrt(power.sum() / N)


# def _iterative_proportional_fit(f_mat: Array,
#                                 target_outflow: Array,
#                                 target_inflow: Array,
#                                 rel_tol: float = 1e-3,
#                                 ϵ: float = 1e-6,
#                                 max_iters: int = 1000,
#                                 return_vecs: bool = False,
#                                 safe_mode: bool = False) -> Array:
#     """
#     Apply the IPFP in the case of the doubly constrained model.
#
#     Parameters
#     ----------
#     f_mat : array_like
#         The affinity matrix.
#     target_outflow, target_inflow : array_like
#         Respectively the column-sum and row-sum that the doubly constrained
#         model should have.
#     rel_tol: float, optional
#     ϵ : float, optional
#         The value to use as 'zero' for convergence aspects.
#     max_iters : int, optional
#     return_vecs : bool, optional
#         Return the Ai and Bj normalising vectors (default is False).
#     safe_mode : bool, optional
#         Whether to use assert statements or not (default is False).
#
#     Returns
#     -------
#     np.array
#         The doubly constrained model (or the best approximation to it).
#
#     """
#     N, M = f_mat.shape
#
#     if not isinstance(target_inflow, np.ndarray):
#         target_outflow = np.array(target_outflow)
#     if not isinstance(target_inflow, np.ndarray):
#         target_inflow = np.array(target_inflow)
#
#     if safe_mode:
#         assert N == M, 'matrix is should be square'
#         assert len(target_outflow) == N,\
#             'target_outflow should have same size as matrix'
#         assert len(target_inflow) == N,\
#             'target_inflow should have same size as matrix'
#         assert not np.any(target_outflow < 0),\
#             'target_outflow should be non-negative'
#         assert not np.any(target_inflow < 0),\
#             'target_inflow should be non-negative'
#
#     idx_rows = (target_outflow < ϵ)
#     idx_cols = (target_inflow < ϵ)
#
#     # Verify that the superset of zeros is in target
#     model_outflow = np.asarray(f_mat.sum(axis=1)).flatten()
#     idx_rows_subset = (model_outflow < ϵ)
#     assert np.all(idx_rows[idx_rows_subset]),\
#         'unsatisfiable constraints (zero rows in model and non-zero in target)'
#
#     model_inflow = np.asarray(f_mat.sum(axis=0)).flatten()
#     idx_cols_subset = (model_inflow < ϵ)
#     assert np.all(idx_cols[idx_cols_subset]),\
#         'unsatisfiable constraints (zero cols in model and non-zero in target)'
#
#     # Initialise variables
#     num_iter = 0
#     converged = False
#     outscale_vec_old = np.ones(N)  # vector of A_i (associated to O_i)
#
#     while not converged and num_iter < max_iters:
#         num_iter += 1
#
#         # NB: M @ x is matrix multiplication (no need for x to be a column vec.)
#
#         # Update B_j
#         inscale_vec = np.ones(N)  # vector of B_j (associated to D_j)
#         # Σ_i F_ij * (A_i O_i)
#         denom = f_mat.T @ (outscale_vec_old * target_outflow)
#         np.reciprocal(denom, out=inscale_vec, where=(denom > ϵ))
#         # The where clause takes care of dividing by zero, A_i = 1 in that case
#
#         # Update A_i
#         outscale_vec = np.ones(N)
#         denom = f_mat @ (inscale_vec * target_inflow)  # Σ_j F_ij * (B_j D_j)
#         np.reciprocal(denom, out=outscale_vec, where=(denom > ϵ))
#         # Idem; B_j = 1
#
#         AO_vec = outscale_vec * target_outflow
#         BD_vec = inscale_vec * target_inflow
#         F_rescaled = BD_vec * f_mat * AO_vec[:, np.newaxis]
#
#         # Current error
#         model_outflow = np.asarray(F_rescaled.sum(axis=1)).flatten()
#         model_inflow = np.asarray(F_rescaled.sum(axis=0)).flatten()
#
#         # Note we exclude zero rows or zero columns
#         colsum_error = np.linalg.norm(
#             model_outflow[~idx_rows] / target_outflow[~idx_rows] - 1.0, np.inf)
#         rowsum_error = np.linalg.norm(
#             model_inflow[~idx_cols] / target_inflow[~idx_cols] - 1.0, np.inf)
#         iter_error = np.linalg.norm(outscale_vec - outscale_vec_old)
#
#         # After error, update variable
#         outscale_vec_old = outscale_vec
#
#         if colsum_error < rel_tol and rowsum_error < rel_tol:
#             converged = True
#         elif iter_error < ϵ:
#             warnings.warn(
#                 f'Iterations converged ({num_iter} iters) but constraints cannot be satisfied')
#             # warnings.warn(f'{colsum_error:.3f}, {rowsum_error:.3f}')
#             converged = True
#
#     if not converged:
#         raise FailedToConverge(
#             f'reached maximum number of iterations ({max_iters})')
#
#     return (F_rescaled, outscale_vec, inscale_vec) if return_vecs else F_rescaled


# class FailedToConverge(Exception):
#     """Raised when an iterative method failed to converge."""
#
#     pass


class DataNotSet(Exception):
    """Raised when a method needs to access the data and it has not been set."""

    pass


def simple_ipf(mat: Array,
               target_rows: Array = None,
               target_cols: Array = None,
               return_vecs: bool = False,
               tol: float = 1e-3,
               maxiters: int = 100,
               verbose: bool = False) -> Array:
    N, _ = mat.shape
    b = np.ones(N)

    assert not (bool(target_rows is None) != bool(target_cols is None)), \
        'target_rows and target_cols should be provided or not provided together'
    # != is exclusive or for normalised boolean variables

    if target_rows is None:
        target_rows = np.ones(N)
        target_cols = np.ones(N)

    niter = 0

    while niter < maxiters:
        a = target_rows / (mat @ b)
        b = target_cols / (mat.T @ a)

        out = a[:, np.newaxis] * mat * b[np.newaxis, :]
        bool_row = np.allclose(out.sum(axis=1), target_rows, atol=tol)
        # rel_tol instead? this is the problem I address inside CPC, etc.
        bool_col = np.allclose(out.sum(axis=0), target_cols, atol=tol)

        niter += 1

        if bool_row and bool_col:
            break

    if verbose:
        print(f'Nb iters until convergence: {niter}')

    return (out, a, b) if return_vecs else out


def save_model(filename, locs=None, constraint_type=None,
               grav_params=None, balancing_factors=None):
    """
    Save a model storing optionally the locations object, the parameters or
    the balancing factors.

    """
    save_dict = {}

    if locs is not None:
        assert locs.data is not None
        save_dict['locs'] = locs

    if constraint_type is not None:
        assert constraint_type in ['unconstrained', 'production', \
                                   'attraction', 'doubly']
        save_dict['constraint_type'] = constraint_type

    if grav_params is not None:
        assert len(grav_params) == 3
        save_dict['grav_params'] = grav_params

    if balancing_factors is not None:
        assert len(balancing_factors) == 2
        save_dict['balancing_factors'] = balancing_factors

    if len(save_dict) > 0:
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
    else:
        print('Need to provide arguments')

    return None


def load_gravity_model(filename, locs=None, return_locs=False):
    """
    Load gravity model using the saved parameters and optionally the saved
    balancing factors.

    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        constraint_type = data.get('constraint_type')

        if locs is not None:
            pass
        elif 'locs' in data:
            locs = data['locs']
        else:
            raise AttributeError('`locs` not provided')

        fmat = locs.gravity_matrix(*data['grav_params'])

    if constraint_type is None:
        out = fmat

    elif constraint_type == 'unconstrained':
        K = locs.data.sum() / fmat.sum()
        out = K * fmat

    elif constraint_type == 'production':
        pmat = locs.probability_matrix(fmat, 'production')
        out = locs.data_out[:, np.newaxis] * pmat

    elif constraint_type == 'attraction':
        pmat = locs.probability_matrix(fmat, 'attraction')
        out = pmat * locs.data_in[np.newaxis, :]

    elif constraint_type == 'doubly' and 'balancing_factors' in data:
        a, b = data['balancing_factors']
        out = a[:, np.newaxis] * fmat * b[np.newaxis, :]

    elif constraint_type == 'doubly':
        out = simple_ipf(fmat, locs.data_out, locs.data_in)

    else:
        pass

    return (out, locs) if return_locs else out
