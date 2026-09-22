import yt
import numpy as np
import matplotlib.pyplot as plt
import glob
import re

# Find Athena++ output files
files = glob.glob('Bondi.out1.*.athdf')

# Sort files by output number
files = sorted(
    files,
    key=lambda f: int(re.search(r'out1\.(\d+)', f).group(1))
)

# Bondi quantities
rB = 4.5e8
rho_infty = 1.0

# Bondi time in code units
tB_code = 4.5e8 / 3.3356409519815205e-05

# Get simulation times
times = np.array([
    float(yt.load(f).current_time)
    for f in files
])

times_B = times / tB_code

# Colour map
cmap = plt.cm.inferno
norm = plt.Normalize(times_B.min(), times_B.max())

# Plot
fig, ax = plt.subplots(figsize=(7, 5))

for file, t_B in zip(files, times_B):

    data = yt.load(file)
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
    print("Number of cells:", len(r))
    print("Number of unique radii:",len(np.unique(r)))
    ax.plot(
        r_norm,
        rho_norm,
        color=cmap(norm(t_B))
    )

# Colourbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

cbar = fig.colorbar(sm, ax=ax)
cbar.set_label(r'$t/t_B$')

# Mark the reference radius used in the IC
ax.axvline(
    1e-2,
    linestyle='--',
    linewidth=1,
    label=r'$r = 10^{-2}r_B$'
)

ax.set_xlabel(r'$r/r_B$')
ax.set_ylabel(r'$\rho/\rho_\infty$')

ax.set_xscale('log')
ax.set_yscale('log')

ax.set_title('Evolution of density profile')

ax.legend()

plt.tight_layout()
plt.show()