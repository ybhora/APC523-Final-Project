import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
import os, glob

DATA_DIR = 'data/quintessence'
FITS_DIR = 'data/fits'
N_MIN = -np.log(5.0)
N_FIT = np.linspace(N_MIN, 0.0, 401)
BOUNDS = [(-1.2, -0.4), (-2.0, 0.5), (0.65, 0.75), (0.95, 1.05)]

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

def fit(H_Q):
    res = differential_evolution(err_max, BOUNDS, args=(H_Q,),
                                  tol=1e-6, seed=42, maxiter=80, polish=False)
    res = minimize(err_max, res.x, args=(H_Q,), method='Nelder-Mead',
                   options={'xatol': 1e-7, 'fatol': 1e-10, 'maxiter': 5000})
    return res.x, res.fun

def load(path):
    d = np.load(path)
    H_Q = np.interp(N_FIT, d['N'], d['H_Q'])
    w_Q = np.interp(N_FIT, d['N'], d['w_Q'])
    return H_Q, w_Q, float(d['param'])

def save(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)

def scan(paths, label):
    print(f'=== {label} scan ===', flush=True)
    rows = []
    for p in paths:
        H_Q, w_Q, val = load(p)
        params, E = fit(H_Q)
        rows.append((val, *params, E, w_Q[-1]))
        print(f'  {label}={val:.4g}: w0={params[0]:.4f} wa={params[1]:.4f} '
              f'Om0={params[2]:.4f} H0={params[3]:.4f} E={E:.2e}', flush=True)
    rows.sort(key=lambda r: r[0])
    a = np.array(rows)
    return dict(param=a[:,0], w0=a[:,1], wa=a[:,2],
                Om0=a[:,3], H0=a[:,4], E=a[:,5], wQ_0=a[:,6])

exp_paths = glob.glob(f'{DATA_DIR}/exp_lam*/solution.npz')
save(scan(exp_paths, 'exp'), f'{FITS_DIR}/exp_scan.npz')

hil_paths = glob.glob(f'{DATA_DIR}/hilltop_k1_phi*/solution.npz')
save(scan(hil_paths, 'hilltop'), f'{FITS_DIR}/hilltop_scan.npz')

print('Done.', flush=True)
