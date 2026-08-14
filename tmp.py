import torch, time
from algebra import ad_matrix, get_eye_cached, dagger, random_hermitian

N = 20
B = 4
Xb = torch.randn(B, N, N, dtype=torch.complex64, device='cpu')
Xb = (Xb + Xb.conj().transpose(-1,-2)) / 2

# warmup
for _ in range(5):
    _ = ad_matrix(Xb)

t0 = time.perf_counter()
for _ in range(1000):
    r = ad_matrix(Xb)
t1 = time.perf_counter()
print(f'new: {(t1-t0)*1000:.2f} ms / 1000 calls')

r = ad_matrix(Xb)
print(torch.norm(r[0] - dagger(r[0])))

# old implementation inline
def ad_matrix_old(X):
    single = X.ndim == 2
    if single: X = X.unsqueeze(0)
    B2, N2, _ = X.shape
    I = get_eye_cached(N2, device=X.device, dtype=X.dtype)
    Xt = X.transpose(-1, -2)
    def kron_batched(A, B):
        k = torch.einsum('...ij,...kl->...ikjl', A, B)
        *batch, m, k2, n, l = k.shape
        return k.reshape(*batch, m*k2, n*l)
    result = kron_batched(I.view(1, N2, N2), X) - kron_batched(Xt, I.view(1, N2, N2))
    if single: result = result.squeeze(0)
    return result

for _ in range(5):
    _ = ad_matrix_old(Xb)

t0 = time.perf_counter()
for _ in range(1000):
    r = ad_matrix_old(Xb)
t1 = time.perf_counter()
print(f'old: {(t1-t0)*1000:.2f} ms / 1000 calls')

r = ad_matrix_old(Xb)
print(torch.norm(r[0] - dagger(r[0])))
print(torch.trace(r[0]))