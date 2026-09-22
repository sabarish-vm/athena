import numpy as np
import yt
import matplotlib.pyplot as plt
import glob
import os


folder = '.'

rB = 1e4
rho_infty = 4e-6
c_s_infty = 1e-2
tB = rB / c_s_infty

GN = 2.9e-14
mbh = 3.404e13

# Analytic Bondi accretion rate, for comparison against the simulated M(r)
mdot_analytic = np.pi * GN**2 * mbh**2 * rho_infty / c_s_infty**3

N_EDGE = 10   
files = glob.glob(f'{folder}/Bondi.out1.*.athdf')
files = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[-2]))
if not files:
    raise FileNotFoundError(f"No files matched {folder}/Bondi.out1.*.athdf")

times = np.array([float(yt.load(f).current_time) for f in files])
times_norm = times / tB


def load(f):
    """Load one snapshot, sorted by radius, in raw code units."""
    data = yt.load(f)
    ad = data.all_data()
    r = ad[('index', 'r')].d
    order = np.argsort(r)
    r = r[order]
    rho = ad[('athena_pp', 'rho')].d[order]
    vel = ad[('athena_pp', 'vel1')].d[order]
    return r, rho, vel


def M_of_r(r, rho, vel):
    return -4.0 * np.pi * r**2 * rho * vel   # vel<0 inflow -> M>0


r_last, rho_last, vel_last = load(files[-1])
M_last = M_of_r(r_last, rho_last, vel_last)

print("=== Step 1: outer M vs analytic mdot ===")
print(f"mdot (analytic)   = {mdot_analytic:.4e}")
print(f"M at outer edge   = {M_last[-1]:.4e}   "
      f"(ratio = {M_last[-1]/mdot_analytic:.3f})")

r_first, rho_first, vel_first = load(files[0])

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

ax1.plot(r_first[-N_EDGE:]/rB, rho_first[-N_EDGE:]/rho_infty, 'o-k',
          label=rf'$t={times_norm[0]:.3g}\ [t_B]$')
ax1.plot(r_last[-N_EDGE:]/rB, rho_last[-N_EDGE:]/rho_infty, 'o-', color='orange',
          label=rf'$t={times_norm[-1]:.3g}\ [t_B]$')
ax1.set_xlabel(r'$r\ [r_B]$'); ax1.set_ylabel(r'$\rho\ [\rho_\infty]$')
ax1.set_title('Density near outer edge'); ax1.legend()

ax2.plot(r_first[-N_EDGE:]/rB, np.abs(vel_first[-N_EDGE:])/c_s_infty, 'o-k')
ax2.plot(r_last[-N_EDGE:]/rB, np.abs(vel_last[-N_EDGE:])/c_s_infty, 'o-', color='orange')
ax2.set_xlabel(r'$r\ [r_B]$'); ax2.set_ylabel(r'$u_r\ [c_{s,\infty}]$')
ax2.set_title('Velocity near outer edge')

norm = plt.Normalize(vmin=times_norm.min(), vmax=times_norm.max())
cmap = plt.cm.copper

M_inner = []
for f, t in zip(files, times_norm):
    r, rho, vel = load(f)
    M = M_of_r(r, rho, vel)
    M_inner.append(np.mean(M[:N_EDGE]))
    ax3.plot(r[:N_EDGE]/rB, M[:N_EDGE], color=cmap(norm(t)))

ax3.axhline(mdot_analytic, color='red', ls='--', label=r'$\dot{M}$ analytic')
ax3.set_xscale('log'); ax3.set_yscale('log')
ax3.set_xlabel(r'$r\ [r_B]$'); ax3.set_ylabel(r'$\dot{M}\ [code]$')
ax3.set_title('Mass accretion near inner edge, all times')
ax3.legend()

plt.tight_layout()
outpath = os.path.join('New_plots/Modified_IC/bondi_check_new_bc_times2.png')
plt.savefig(outpath, bbox_inches='tight')
print(f"\nSaved plot: {outpath}")

print("\n=== Step 5: inner M/mdot vs time ===")
for t, m in zip(times_norm, M_inner):
    print(f"  t/tB={t:.4g}   M/mdot={m/mdot_analytic:.4f}")

ratio_last = M_inner[-1] / mdot_analytic
if ratio_last > 0.9:
    print("-> Inner region converged.")
elif len(M_inner) > 2 and M_inner[-1] > M_inner[-2]:
    print("-> Inner region still rising -> probably just needs longer tlim.")
else:
    print("-> Inner region stuck, not rising -> worth investigating "
          "(timestep near inner boundary, floors, outflow BC).")

plt.show()