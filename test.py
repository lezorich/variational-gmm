import numpy as np
from scipy.special import digamma, loggamma, gammaln


def calculate_S_k(X, x_k_hat, responsibilities, N_k):
    """
    Return (K, D, D) tensor.
    Eq. 10.53 from Bishop
    """
    # (N, K, D) tensor
    normalizer = X[:, np.newaxis] - x_k_hat
    # (D, N, K) tensor
    normalizer = np.transpose(normalizer, axes=[2, 0, 1])

    # (K, D, N) tensor
    trans_normalizer = np.transpose(normalizer, axes=[2, 0, 1])

    # Multiply responsibilities for each (K, N) D vector. We still have
    # a (K, N, D) tensor.
    prod = responsibilities[np.newaxis, :] * normalizer
    prod = np.transpose(prod, axes=[2, 1, 0]) # (K, N, D) tensor

    # K dot products of dimensions (D, N) x (N, D) to get K (D, D) matrices
    # (K, D, D) tensor
    k_d_d_matrixes = np.einsum('lij, ljk -> lik',
                               trans_normalizer,
                               prod)

    return 1. / N_k[:, np.newaxis, np.newaxis] * k_d_d_matrixes

def calc_S_k(Z, X, xd, NK):
    (N, K) = np.shape(Z)
    (N1, D) = np.shape(X)
    assert N == N1

    S = [np.zeros((D, D)) for _ in range(K)]
    for n in range(N):
        for k in range(K):
            B0 = np.reshape(X[n, :] - xd[k, :], (D, 1))
            L = np.dot(B0, B0.T)
            assert np.shape(L) == np.shape(S[k]), np.shape(L)
            S[k] += Z[n, k] * L
    # safe divide
    for k in range(K):
        if NK[k] > 0:
            S[k] = S[k] / NK[k]
        else:
            print('Warning: NK[{}] is zero or less'.format(k))
    return S


def calcW(K, W0, xd, NK, m0, D, beta0, S):
    Winv = [None for _ in range(K)]
    for k in range(K):
        Winv[k] = np.linalg.inv(W0) + NK[k] * S[k]
        Q0 = np.reshape(xd[k, :] - m0, (D, 1))
        q = np.dot(Q0, Q0.T)
        Winv[k] += (beta0 * NK[k] / (beta0 + NK[k])) * q
        assert np.shape(q) == (D, D)
    W = []
    for k in range(K):
        try:
            W.append(np.linalg.inv(Winv[k]))
        except linalg.linalg.LinAlgError:
            print('Winv[{}]'.format(k), Winv[k])
            raise linalg.linalg.LinAlgError()
    return W

def _calculate_W_k(x_k_hat, m_0, K, D, N_k, beta_0, inv_W_0, S_k):
    """
    Return (K, D, D) tensor.
    Eq. 10.62 from Bishop
    """
    temp1 = x_k_hat - m_0
    # (K, D, D) tensor
    temp = np.einsum('ij, ik -> ijk', temp1, temp1)
    assert temp.shape == (K, D, D)

    temp2 = beta_0 * N_k / (beta_0 + N_k)
    # We have the inverted W_k
    inv_W_k = (inv_W_0[np.newaxis, :]
            + N_k[:, np.newaxis, np.newaxis] * S_k
            + temp2[:, np.newaxis, np.newaxis] * temp)
    # Invert inv_W_k
    return np.linalg.inv(inv_W_k)


def _calculate_E_mu_lambda(X, m_k, W_k, D, beta_k, nu_k):
    """
    Return (K, N) matrix.
    Eq. 10.64 from Bishop.
    """
    diff = X[:, np.newaxis] - m_k  # (N, K, D) tensor

    # (K, N, D) tensor
    diff = np.transpose(diff, axes=[1, 0, 2])

    # Take the dot product from each (N, D) vector with the (D, D) matrix.
    # The result is a (K, N, D) tensor
    first_dot = np.einsum('lij, ljk -> lik', diff, W_k)

    mul = np.sum(first_dot * diff, axis=2)  # (K, N) matrix.
    return (D / beta_k[:, np.newaxis]
            + nu_k[:, np.newaxis] * mul)

def Muopt(X, D, NK, betak, m, W, xd, vk, N, K):
    Mu = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            A = D / betak[k]  # shape (k,)
            B0 = np.reshape((X[n, :] - m[k, :]), (D, 1))
            B1 = np.dot(W[k], B0)
            l = np.dot(B0.T, B1)
            assert np.shape(l) == (1, 1), np.shape(l)
            Mu[n, k] = A + vk[k] * l  # shape (n,k)

    return Mu


def _calculate_E_log_det_lambda(D, nu_k, W_k):
    """
    Return K vector.
    Eq. 10.65 from Bishop.
    """
    # The original formula is nu_k + 1 - (np.arange(D) + 1), but we can
    # simplify it.
    temp0 = nu_k[:, np.newaxis] - np.arange(D)
    temp1 = digamma(temp0 / 2.)
    print('a', np.sum(temp1, axis=1))
    E_log_det_lambda = (np.sum(temp1, axis=1)
                        + D * np.log(2)
                        + np.log(np.linalg.det(W_k)))
    return E_log_det_lambda


def inv(matrix):
    return np.linalg.inv(matrix)


def mul(m1, m2):
    return np.dot(m1, m2)


def arr(list):
    return np.array(list)


def deter(matrix):
    return np.linalg.det(matrix)


def _E_ln_lambda_v(D, vk, Wk):
    print('b', sum(digamma((np.tile(vk, (D, 1)) - np.arange(1, D + 1)[:, None] + 1)/2)))
    return arr(sum(digamma((np.tile(vk, (D, 1)) - np.arange(1, D + 1)[:, None] + 1)/2)) + D * np.log(2) +  np.log(deter(Wk)))



def log_C(alpha):
    """Return the logarithm of the C(alpha) function (eq. B.23 of Bishop)."""
    return gammaln(np.sum(alpha)) - np.sum(gammaln(alpha))


def log_B(W, nu):
    """Return the logarithm of the B(W, nu) function (eq. B.79 of Bishop).

       Parameters:
       W -- D x D symmetric positive definite matrix.
       nu -- number of degrees of freedom of the distribution, restricted to
             nu > D - 1 to ensure that the Gamma function in the normalization
             factor is well defined.
    """
    D = W.shape[1]
    q1 = -.5 * nu * np.log(np.linalg.det(W))
    q2 = (nu * D / 2. * np.log(2.)
          + D * (D - 1) / 4. * np.log(np.pi)
          + np.sum(gammaln(.5 * (nu - np.arange(D)))))
    return q1 - q2


def wishart_entropy(W, nu):
    """Return the Wishart entropy (eq. B.82 of Bishop).

       Parameters:
       W -- D x D symmetric positive definite matrix.
       nu -- number of degrees of freedom of the distribution, restricted to
             nu > D - 1 to ensure that the Gamma function in the normalization
             factor is well defined.
    """
    D = W.shape[1]
    q1 = (np.sum(digamma(.5 * (nu - np.arange(D))))
          + D * np.log(2)
          + np.log(np.linalg.det(W)))
    entropy = -log_B(W, nu) - .5 * (nu - D - 1) * q1 + .5 * nu * D
    return entropy


np.random.seed(345433234)
N = 10
D = 2
X = np.random.random(size=(N, D))
K = 3
beta0 = 1e-20
W0 = np.eye(D)
m_0 = np.zeros(D)
nu_0 = D + 1.
inv_W_0 = np.linalg.inv(W0)
r = np.random.dirichlet(np.ones(K), size=N)
N_k = r.sum(axis=0)
x_k_hat = 1. / N_k[:, np.newaxis] * r.T.dot(X)
beta_k = beta0 + N_k
m_k = 1. / beta_k[:, np.newaxis] * (beta0 * m_0 + N_k[:, np.newaxis] * x_k_hat)
nu_k = nu_0 + N_k
S_k_1 = calculate_S_k(X, x_k_hat, r, N_k)
S_k_2 = calc_S_k(r, X, x_k_hat, N_k)
W_k_1 = _calculate_W_k(x_k_hat, m_0, K, D, N_k, beta0, inv_W_0, S_k_1)
W_k_2 = calcW(K, W0, x_k_hat, N_k, m_0, D, beta0, S_k_2)
# print('S_k_1: {}'.format(S_k_1))
# print('S_k_2: {}'.format(S_k_2))
# print('W_k_1: {}'.format(W_k_1))
# print('W_k_2: {}'.format(W_k_2))
# print(_calculate_E_mu_lambda(X, m_k, W_k_1, D, beta_k, nu_k))
# print(Muopt(X, D, N_k, beta_k, m_k, W_k_2, x_k_hat, nu_k, N, K))
print(_calculate_E_log_det_lambda(D, nu_k, W_k_1))
print(_E_ln_lambda_v(D, nu_k, W_k_1))
print(nu_k.shape)
print(N_k.shape)


ww = np.asarray([[4.00648864, -3.63846918, -3.00460901],
                 [-3.63846918,  4.31520301,  2.67550852],
                 [-3.00460901,  2.67550852,  2.94852793]])
nu_ww = 3

from scipy.stats import wishart

print('entropyyy')
print(wishart.entropy(df=nu_ww, scale=ww))
print(wishart_entropy(ww, nu_ww))
