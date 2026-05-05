import numpy as np
from scipy.integrate import solve_ivp
import os

OMEGA_Q_TARGET = 0.7
OMEGA_Q_INIT = 1e-6
N_MIN = -np.log(5.0)
N_GRID = np.linspace(N_MIN, 0.0, 1001)
DATA_DIR = 'data/quintessence'

def potential(kind, **p):
    if kind == 'exp':
        lam = p['lam']
        return (lambda phi: np.exp(lam*phi),
                lambda phi: lam*np.exp(lam*phi))
    if kind == 'hilltop':
        k2 = p['k2']
        return (lambda phi: 1.0 - 0.5*k2*phi**2,
                lambda phi: -k2*phi)

def rhs(s, y, V, dV):
    phi, u, OmQ = y
    Vp = V(phi)
    H2 = Vp / (3*OmQ - 0.5*u**2)
    KE = 0.5*H2*u**2
    wQ = (KE - Vp) / (KE + Vp)
    return [u,
            1.5*u*(wQ*OmQ - 1) - dV(phi)/H2,
            3*wQ*OmQ*(OmQ - 1)]

def hit_target(s, y, V, dV):
    return y[2] - OMEGA_Q_TARGET
hit_target.terminal = True
hit_target.direction = 1

def solve_adaptive(V, dV, phi_i, rtol=1e-10, atol=1e-12):
    y0 = [phi_i, 0.0, OMEGA_Q_INIT]
    sol = solve_ivp(rhs, (0.0, 30.0), y0, args=(V, dV), method='DOP853',
                    rtol=rtol, atol=atol, events=hit_target, dense_output=True)
    s_f = sol.t_events[0][0]
    phi, u, OmQ = sol.sol(N_GRID + s_f)
    Vp = V(phi)
    H2 = Vp / (3*OmQ - 0.5*u**2)
    KE = 0.5*H2*u**2
    wQ = (KE - Vp) / (KE + Vp)
    H = np.sqrt(H2)
    return dict(N=N_GRID, phi=phi, u=u, Omega_Q=OmQ, w_Q=wQ, H_Q=H/H[-1])

def save(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)

exp_lambdas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
hilltop_phis = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
K2_HILLTOP = 1.0

print('=== Exponential potential ===', flush=True)
for lam in exp_lambdas:
    V, dV = potential('exp', lam=lam)
    out = solve_adaptive(V, dV, phi_i=0.0)
    out['param'] = lam
    save(out, f'{DATA_DIR}/exp_lam{lam:.2f}/solution.npz')
    print(f'  lam={lam:.2f}: w_Q(0)={out["w_Q"][-1]:.4f}, Om_Q(0)={out["Omega_Q"][-1]:.4f}', flush=True)

print(f'=== Hilltop k^2={K2_HILLTOP} ===', flush=True)
for phi_i in hilltop_phis:
    V, dV = potential('hilltop', k2=K2_HILLTOP)
    out = solve_adaptive(V, dV, phi_i=phi_i)
    out['param'] = phi_i
    save(out, f'{DATA_DIR}/hilltop_k1_phi{phi_i:.3f}/solution.npz')
    print(f'  phi_i={phi_i:.3f}: w_Q(0)={out["w_Q"][-1]:.4f}, Om_Q(0)={out["Omega_Q"][-1]:.4f}', flush=True)

print('Done.', flush=True)
