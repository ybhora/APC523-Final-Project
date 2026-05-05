import numpy as np
from scipy.integrate import solve_ivp
import os, time

OMEGA_Q_TARGET = 0.7
OMEGA_Q_INIT = 1e-6
N_MIN = -np.log(5.0)
N_GRID = np.linspace(N_MIN, 0.0, 1001)
CONV_DIR = 'data/convergence'

K2 = 1.0
PHI_I = 0.5
V = lambda phi: 1.0 - 0.5*K2*phi**2
dV = lambda phi: -K2*phi

def rhs(s, y):
    phi, u, OmQ = y
    Vp = V(phi)
    H2 = Vp / (3*OmQ - 0.5*u**2)
    KE = 0.5*H2*u**2
    wQ = (KE - Vp) / (KE + Vp)
    return [u,
            1.5*u*(wQ*OmQ - 1) - dV(phi)/H2,
            3*wQ*OmQ*(OmQ - 1)]

def hit_target(s, y):
    return y[2] - OMEGA_Q_TARGET
hit_target.terminal = True
hit_target.direction = 1

def derived(phi, u, OmQ):
    Vp = V(phi)
    H2 = Vp / (3*OmQ - 0.5*u**2)
    KE = 0.5*H2*u**2
    return np.sqrt(H2), (KE - Vp) / (KE + Vp)

def solve_adaptive(rtol, atol, method='DOP853'):
    y0 = [PHI_I, 0.0, OMEGA_Q_INIT]
    t0 = time.time()
    sol = solve_ivp(rhs, (0.0, 30.0), y0, method=method,
                    rtol=rtol, atol=atol, events=hit_target, dense_output=True)
    runtime = time.time() - t0
    s_f = sol.t_events[0][0]
    phi, u, OmQ = sol.sol(N_GRID + s_f)
    H, wQ = derived(phi, u, OmQ)
    return dict(N=N_GRID, H_Q=H/H[-1], w_Q=wQ, Omega_Q=OmQ,
                nfev=sol.nfev, runtime=runtime)

def rk4_step(s, y, h):
    k1 = np.asarray(rhs(s, y))
    k2 = np.asarray(rhs(s + h/2, y + h*k1/2))
    k3 = np.asarray(rhs(s + h/2, y + h*k2/2))
    k4 = np.asarray(rhs(s + h, y + h*k3))
    return y + h*(k1 + 2*k2 + 2*k3 + k4)/6

def solve_rk4(h):
    y = np.array([PHI_I, 0.0, OMEGA_Q_INIT])
    ss = [0.0]; ys = [y.copy()]
    s = 0.0
    t0 = time.time()
    while s < 20.0 and y[2] < OMEGA_Q_TARGET:
        y = rk4_step(s, y, h)
        s += h
        ss.append(s); ys.append(y.copy())
    runtime = time.time() - t0
    ss = np.array(ss); ys = np.array(ys)
    i = np.searchsorted(ys[:, 2], OMEGA_Q_TARGET)
    s_f = ss[i-1] + (OMEGA_Q_TARGET - ys[i-1, 2])*(ss[i]-ss[i-1])/(ys[i, 2]-ys[i-1, 2])
    N_arr = ss - s_f
    phi, u, OmQ = ys.T
    H, wQ = derived(phi, u, OmQ)
    H = H / np.interp(0.0, N_arr, H)
    return dict(N=N_GRID,
                H_Q=np.interp(N_GRID, N_arr, H),
                w_Q=np.interp(N_GRID, N_arr, wQ),
                Omega_Q=np.interp(N_GRID, N_arr, OmQ),
                n_steps=len(ss), runtime=runtime)

def save(out, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **out)

print('=== Reference (DOP853, rtol=1e-12) ===', flush=True)
ref = solve_adaptive(rtol=1e-12, atol=1e-14)
save(ref, f'{CONV_DIR}/reference.npz')
print(f'  nfev={ref["nfev"]}, t={ref["runtime"]:.2f}s', flush=True)
H_ref = ref['H_Q']

print('=== Fixed-step RK4 ===', flush=True)
for h in [0.04, 0.02, 0.01, 0.005, 0.0025]:
    out = solve_rk4(h)
    out['h'] = h
    err = np.max(np.abs((out['H_Q'] - H_ref) / H_ref))
    out['err_H'] = err
    save(out, f'{CONV_DIR}/rk4_h{h:.4f}.npz')
    print(f'  h={h:.4f}: err={err:.2e}, n_steps={out["n_steps"]}, t={out["runtime"]:.3f}s', flush=True)

print('=== Adaptive (DOP853) ===', flush=True)
for rtol in [1e-6, 1e-8, 1e-10, 1e-12]:
    out = solve_adaptive(rtol=rtol, atol=rtol*1e-2)
    out['rtol'] = rtol
    err = np.max(np.abs((out['H_Q'] - H_ref) / H_ref))
    out['err_H'] = err
    save(out, f'{CONV_DIR}/adaptive_rtol{rtol:.0e}.npz')
    print(f'  rtol={rtol:.0e}: err={err:.2e}, nfev={out["nfev"]}, t={out["runtime"]:.3f}s', flush=True)

print('Done.', flush=True)
