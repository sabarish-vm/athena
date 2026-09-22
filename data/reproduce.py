import numpy as np
import yt
import matplotlib.pyplot as plt
import glob
import os

#rB = 1e4
#rho_infty = 4e-6
#c_sinfty = 1e-2

#G = 2.9e-14
G = 6.5e-28
#ur_infty   = -1e-4
#mbh        = 3.404e13  #1.0
rB = 4.5e8
rho_infty = 1.
c_sinfty = 3.3356409519815205e-05
ur_infty = 0.0
mbh = 7.719785599077413e+26
tB = rB / c_sinfty

folder = '.'

# Find all snapshots
files = glob.glob(f'{folder}/Bondi.out1.*.athdf')

# Sort snapshots by output number
files = sorted(
    files,
    key=lambda x: int(os.path.basename(x).split('.')[-2])
)

times = np.array([
    float(yt.load(f).current_time)
    for f in files
])

times_norm = times / tB

n_total = len(files)        
n_want = 20                 
idx = np.linspace(0, n_total - 1, n_want, dtype=int)
idx = np.unique(idx)       

files_plot = [files[i] for i in idx]
times_norm_plot = times_norm[idx]

# Numerical solution
r_grid = np.logspace(np.log10(10), np.log10(10000), 200)
mdot_n = np.pi * (G**2 * mbh**2) * rho_infty / (c_sinfty**3)
ur = np.sqrt(G * mbh / r_grid)
rho_r = mdot_n / (4 * np.pi * r_grid * r_grid * ur)

r_grid_norm = r_grid / rB
ur_norm = ur / c_sinfty
rho_r_norm = rho_r / rho_infty

norm = plt.Normalize(
    vmin=times_norm_plot.min(),
    vmax=times_norm_plot.max()
)

cmap = plt.cm.copper

fig, (ax1, ax2, ax3) = plt.subplots(
    1, 3,
    figsize=(12, 5)
)

for i, (f, t) in enumerate(zip(files_plot, times_norm_plot)):

    print(f'Loading {f}, t/tB = {t:.3e}')

    data = yt.load(f)
    ad = data.all_data()

    r = ad[('index', 'r')].d
    rho = ad[('athena_pp', 'rho')].d

    # Sort by radius
    order = np.argsort(r)
    r = r[order]
    rho = rho[order]

    # Normalize
    r_norm = r / rB
    rho_norm = rho / rho_infty

    vel = ad[('athena_pp', 'vel1')].d
    vel = vel[order]

    vel_norm = np.abs(vel) / c_sinfty

    # Color corresponding to time
    color = cmap(norm(t))

    if i == 0:
        label = rf'$t={t:.5f}\ [t_B]$'
    elif i == len(files_plot) - 1:
        label = rf'$t={t:.5f}\ [t_B]$'
    else:
        label = '_nolegend_'
    lw = 3 if i == 0 else 1
    color = cmap(norm(t))

    # Plot density
    ax1.plot(
        r_norm,
        rho_norm,
        color=color,
        alpha=0.8,
        lw=lw,
        label=label
    )

    # Plot velocity
    ax2.plot(
        r_norm,
        vel_norm,
        color=color,
        alpha=0.8,
        lw=lw,
        label=label
    )
    mdot = -4.0 * np.pi * r**2 * rho * vel
    
    ax3.plot(
        r_norm,
        mdot,
        color=color,
        lw=lw,
        label=label
        )

mdot_i = np.pi * 2.9e-14**2 * 3.4e+13**2 * rho_infty / c_sinfty**3
ax1.plot(r_grid_norm, rho_r_norm, '--',color='magenta', label='Numerical Solution')
ax2.plot(r_grid_norm, ur_norm, '--',color='magenta', label='Numerical Solution')
ax3.axhline(mdot_i, label='Numerical Solution')

# Density
ax1.set_xscale('log')
ax1.set_yscale('log')
#ax1.set_xlim(1e-2, 1e2)

# Velocity
ax2.set_xscale('log')
ax2.set_yscale('log')
#ax2.set_xlim(1e-2, 1e2)

# Mass accretion
ax3.set_xscale('log')
ax3.set_yscale('log')

ax1.set_xlabel(r'$r\ [r_B]$')
ax2.set_xlabel(r'$r\ [r_B]$')
ax3.set_xlabel(r'$r\ [r_B]$')

ax1.set_ylabel(r'$\rho\ [\rho_\infty$]')
ax2.set_ylabel(r'$u_r\ [c_{s,\infty}]$')
ax3.set_ylabel(r'$\dot{M}\ [code]$')

ax1.set_title('Density evolution')
ax2.set_title('Velocity evolution')
ax3.set_title('Mass accretion')

ax1.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()


plt.savefig(
    'Without_conduction/r1.5/sabr2.pdf',
    bbox_inches='tight'
)

plt.show()