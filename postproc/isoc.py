
import os
import glob
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import toml

# matplotlib
plt.style.use("~/cerfacs-black.mplstyle")

# logging
logging.basicConfig(
    level=logging.INFO,
    format='\n > %(asctime)s | %(name)s | %(levelname)s \n > %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

input_file = toml.load('input.toml')

# Otions
PLOT_INTG = 1
SHOW = 1


def find_isoc_files(my_path):
    files = sorted(glob.glob(os.path.join(my_path)))
    logger.info("Found %s intg" % len(files))
    return files


def read_integral(lst_files):
    prefix, _ = os.path.split(lst_files[0])
    logger.info(prefix)

    combined_df =  pd.read_csv(lst_files[0])
    for _idx, _file in enumerate(lst_files):
        if _idx:
            # print("\t > %s" % _file)
            _df = pd.read_csv(_file)
            combined_df = pd.merge_ordered(combined_df, _df, fill_method="ffill")

    # Drop duplicates
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    if "Time" in combined_df.columns.values:
        combined_df['Time_ms'] = 1e3 * combined_df['Time']

    # export to csv
    combined_df.to_csv("%s_merged.csv" % prefix, index=False, encoding='utf-8-sig')
    return combined_df





def postproc_integral():
    # Get surface integrals
    A_tot_files = sorted(glob.glob(input_file['isoc']['input_path_int_tot']))
    df_A_tot = read_integral(A_tot_files)
    print(df_A_tot)

    A_res_files = sorted(glob.glob(input_file['isoc']['input_path_int_res']))
    df_A_res = read_integral(A_res_files)
    print(df_A_res)

    A_res_grad_c = sorted(glob.glob(input_file['isoc']['input_path_int_norm_grac_c']))
    df_res_grad_c = read_integral(A_res_grad_c)
    print(df_res_grad_c)

    plt.figure()
    plt.plot(df_A_res.Time_ms, df_A_res.Area, label='Resolved')
    plt.plot(df_A_res.Time_ms, df_A_tot.efcy - df_A_res.Area, "--", label='SGS')
    plt.plot(df_A_tot.Time_ms, df_A_tot.efcy, "-.", label='Total')
    plt.plot(df_res_grad_c.Time_ms, -df_res_grad_c.norm_grad_c, 'o', alpha=0.5, label=r'$\int_V |\vec{\nabla} c|dV$')
    plt.legend()
    plt.xlabel("Time [ms]")
    plt.ylabel("Area [m$^2$]")
    plt.tight_layout()
    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(input_file["isoc"]["output_path"],
                                 'Fig_flame_surface.%s' % ext),
                    bbox_inches='tight',
                    pad_inches=0.01)
    plt.show()


if __name__ == "__main__":
    logger.info("Begin")

    if PLOT_INTG:
        postproc_integral()
