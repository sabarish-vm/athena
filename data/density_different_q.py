import numpy as np
import yt
import matplotlib.pyplot as plt

rB = 4.5e8
rho_infty = 1

run = {
    0.5: 'r^0.5_longer',
    1.2: 'r^1.2_longer',
    1.5: 'r^1.5_test',
    1.9: 'r^1.9_test',
}
norm = plt.Normalize(vmin=min(run.keys()), vmax=max(run.keys()))
cmap = plt.cm.magma
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5), sharey = True)

import glob
import os

for q, folder in run.items():
    files = f'{folder}/Bondi.out1.00000.athdf'

    all_files = glob.glob(f'{folder}/Bondi.out1.*.athdf')

    files_final = max(
        all_files,
        key=lambda x: int(os.path.basename(x).split('.')[-2])
    )


    #Initial time
    data = yt.load(files)
    ad = data.all_data()

    r = ad[('index', 'r')].d
    rho = ad[('athena_pp', 'rho')].d

    color = cmap(norm(q))

    # Sort by radius
    order = np.argsort(r)
    r = r[order]
    rho = rho[order]

    # Normalize
    r_norm = r / rB
    rho_norm = rho / rho_infty
    
    ax1.plot(
        r_norm,
        rho_norm,
        color=color,
        label = rf'q={q}'
    )

    #Finial time
    data_final = yt.load(files_final)
    ad_final = data_final.all_data()

    r_final = ad_final[('index', 'r')].d
    rho_final = ad_final[('athena_pp', 'rho')].d

    # Sort by radius
    order_final = np.argsort(r_final)
    r_final = r_final[order_final]
    rho_final = rho_final[order_final]

    # Normalize
    r_norm_final = r_final / rB
    rho_norm_final = rho_final / rho_infty
    
    ax2.plot(
        r_norm_final,
        rho_norm_final,
        color=color,
        label = rf'q={q}'
    )
    print(q,float(data_final.current_time))

ax1.set_xscale('log')
ax1.set_yscale('log')

ax2.set_xscale('log')
ax2.set_yscale('log')

ax2.set_xlabel(r'$r/r_B$')
ax1.set_xlabel(r'$r/r_B$')

ax1.set_ylabel(r'$\rho/\rho_\infty$')

ax1.set_title('Initial density profiles')
ax2.set_title('Final density profiles')
ax1.legend()
ax2.legend()

plt.tight_layout()
plt.savefig('density_longer_test.pdf')
