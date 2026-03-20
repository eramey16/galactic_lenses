import os
os.environ['SPS_HOME'] = "/global/homes/e/eramey16/fsps"
from docker import classify
from docker import db_util as util

import sqlalchemy as sa
from sqlalchemy import create_engine
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# # Get unlensed gals from classifier
classy = classify.Classifier()
# unlensed_gals = classy.query_galaxy(ra=150, dec=2, radius=5, limit=None)

# lrgs = util.clean_desi(unlensed_gals)

# g, r, z = lrgs.dered_mag_g, lrgs.dered_mag_r, lrgs.dered_mag_z
# w1, w2 = lrgs.dered_mag_w1, lrgs.dered_mag_w2

# # Cuts on color and magnitude for LRGs
# cut1 = (z - w1) > (0.8 * (r - z) - 0.6)
# cut2 = ((g - w1) > 2.9) | ((r - w1) > 1.8)
# cut3 = (((r-w1) > 1.8*(w1-17.14)) & ((r-w1) > (w1-16.33))) | ((r-w1) > 3.3)

# print(len(lrgs[cut1 & cut2 & cut3]), "out of", len(lrgs)), "gals"
# lrgs = lrgs[cut1 & cut2 & cut3]
# lrgs['ls_id'] = lrgs['ls_id'].astype(int)
# # path = os.path.expandvars("$SCRATCH/data/monocle/LRGs")
# # lrgs[['ls_id']].to_csv(f"{path}/lensed_LRGs.dat", index=False)
# z_unlensed = lrgs['z_phot_median']

def run():
    with classy.engine.connect() as conn:
        # gal_tbl = sa.Table("lensed_dr10", classy.meta, autoload_with=classy.engine)
        stmt = "select * from dr10_training"
        lensed_normal = pd.DataFrame(conn.execute(text(stmt)))

        # gal_tbl_2 = sa.Table("unlensed_dr10", classy.meta, autoload_with=classy.engine)
        stmt = "select * from unlensed_dr10"
        unlensed_normal = pd.DataFrame(conn.execute(text(stmt)))

        stmt = "select * from lrg_train"
        lrgs = pd.DataFrame(conn.execute(text(stmt)))
        lensed_lrg = lrgs[lrgs['lensed']==True].reset_index(drop=True)
        unlensed_lrg = lrgs[lrgs['lensed']!=True].reset_index(drop=True)
    set_types = {'lensed': lensed_normal, 'lensed LRG': lensed_lrg, 
                 'unlensed': unlensed_normal, 'unlensed LRG': unlensed_lrg}

    lensed_normal.to_csv("/pscratch/sd/e/eramey16/data/monocle/test_data/lensed_all.dat", index=False)
    print(len(lensed_normal))

    bands = ['g', 'r', 'i', 'z', 'w1', 'w2']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    styles = {
        'lensed':       {'color': 'royalblue',  'linestyle': '-',  'alpha': 0.8, 'zorder': 3},
        'lensed LRG':   {'color': 'royalblue',  'linestyle': '--', 'alpha': 0.8, 'zorder': 3},
        'unlensed':     {'color': 'tomato',      'linestyle': '-',  'alpha': 0.6, 'zorder': 2},
        'unlensed LRG': {'color': 'tomato',      'linestyle': '--', 'alpha': 0.6, 'zorder': 2},
    }

    for ax, band in zip(axes, bands):

        for label, df in set_types.items():
            data = df['dered_mag_'+band].replace([np.inf, -np.inf], np.nan).dropna()
            ax.hist(
                data,
                bins=50,
                density=True,           # Normalize to probability density
                histtype='step',        # Just the outline, no fill
                linewidth=1.5,
                label=label,
                **styles[label]
            )
        # ax.set_ylim([0, 500])
        ax.set_xlabel(f'{band.upper()}-band magnitude', fontsize=14)
        # ax.set_title(f'{band.upper()} band', fontsize=12)
    plt.suptitle("Magnitude Distribution of Samples", fontsize=20)
    
    axes[2].legend()
    for ax in [axes[0], axes[3]]:
        ax.set_ylabel('Density', fontsize=14)
    # plt.savefig("figs/thesis_mag_distribution.png", bbox_inches='tight')


if __name__=='__main__':
    # dr10_training has the lensed galaxies for some reason (I think?)
    run()




