import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import os

DATA_DIR = 'data/quintessence'
DIAG_DIR = 'data/diagnostics'
N_MIN = -np.log(5.0)
N_FIT = np.linspace(N_MIN, 0.0, 401)
BOUNDS = [(-1.2, -0.4), (-2.0, 0.5), (0.65, 0.75), (0.95, 1.05)]
REF_PATH = f'{DATA_DIR}/hilltop_k1_phi0.500/solution.npz'

def fit_rhs(N, y, w0, wa):
    Omf, Hf = y
    wf = w0 + wa*(1 - np.exp(N))
    return [3*wf*Omf*(Omf - 1), -1.5*Hf*(1 + wf*Omf)]

def H_fit(p):
    w0, wa, Om0, H0 = p
    sol = solve_ivp(fit_rhs, (0.0, N_MIN), [Om0, H0], args=(w0, wa),
                    method='DOP853', rtol=1e-10, atol=1e-12,
                    t_eval=N_FIT[::-1])
    return sol.y[1][::-1]

def err_max(p, H_Q):
    return np.max(np.abs((H_fit(p) - H_Q) / H_Q))

def err_rms(p, H_Q):
    return np.sqrt(np.mean(((H_fit(p) - H_Q) / H_Q)**2))

def D_M(H):
    z = np.exp(-N_FIT) - 1
    idx = np.argsort(z)
    z_s, H_s = z[idx], H[idx]
    DM = np.concatenate(([0.0],
                         np.cumsum(0.5*(1/H_s[:-1] + 1/H_s[1:])*np.diff(z_s))))
    return z_s, DM

def err_max_DM(p, H_Q):
    H_f = H_fit(p)
    z_s, DM_Q = D_M(H_Q)
    _, DM_f = D_M(H_f)
    mask = z_s > 1e-3
    return np.max(np.abs((DM_f[mask] - DM_Q[mask]) / DM_Q[mask]))

def fit(err_fn, H_Q):
    res = differential_evolution(err_fn, BOUNDS, args=(H_Q,),
                                  tol=1e-6, seed=42, maxiter=80, polish=False)
    res = minimize(err_fn, res.x, args=(H_Q,), method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000})
    return res.x, res.fun

def save(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)

d = np.load(REF_PATH)
H_ref = np.interp(N_FIT, d['N'], d['H_Q'])
w_ref = np.interp(N_FIT, d['N'], d['w_Q'])

print('=== Best fit (max H error) ===', flush=True)
p_best, E_best = fit(err_max, H_ref)
print(f'  w0={p_best[0]:.4f} wa={p_best[1]:.4f} Om0={p_best[2]:.4f} '
      f'H0={p_best[3]:.4f} E={E_best:.2e}', flush=True)

print('=== Degeneracy heatmap ===', flush=True)
w0g = np.linspace(p_best[0]-0.1, p_best[0]+0.1, 51)
wag = np.linspace(p_best[1]-0.5, p_best[1]+0.5, 51)
E_grid = np.zeros((len(w0g), len(wag)))
for i, w0 in enumerate(w0g):
    for j, wa in enumerate(wag):
        E_grid[i, j] = err_max([w0, wa, p_best[2], p_best[3]], H_ref)
    if i % 10 == 0: print(f'  row {i}/{len(w0g)}', flush=True)
save(dict(w0_grid=w0g, wa_grid=wag, E_grid=E_grid,
          w0_best=p_best[0], wa_best=p_best[1],
          Om0_best=p_best[2], H0_best=p_best[3], E_best=E_best),
     f'{DIAG_DIR}/heatmap.npz')

print('=== Metric sensitivity ===', flush=True)
out = {}
for name, fn in [('max_H', err_max), ('rms_H', err_rms), ('max_DM', err_max_DM)]:
    p, E = fit(fn, H_ref)
    out[f'{name}_params'] = p
    out[f'{name}_err_max'] = err_max(p, H_ref)
    out[f'{name}_err_rms'] = err_rms(p, H_ref)
    out[f'{name}_err_DM'] = err_max_DM(p, H_ref)
    print(f'  {name}: w0={p[0]:.4f} wa={p[1]:.4f} '
          f'E_max={out[f"{name}_err_max"]:.2e} E_rms={out[f"{name}_err_rms"]:.2e} '
          f'E_DM={out[f"{name}_err_DM"]:.2e}', flush=True)
save(out, f'{DIAG_DIR}/metric_sensitivity.npz')

print('Done.', flush=True)
