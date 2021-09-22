
import os
import glob
import logging

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import toml
import tqdm

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
PLOT_STATS = 1
PLOT_INTG = 0
SHOW = 1


def find_isoc_files(my_path):
    files = sorted(glob.glob(os.path.join(my_path)))
    logger.info("Found %s intg" % len(files))
    return files


def read_integral(lst_files):
    prefix, _ = os.path.split(lst_files[0])
    logger.info(prefix)

    combined_df = pd.read_csv(lst_files[0])
    for _idx, _file in enumerate(lst_files):
        if _idx:
            # print("\t > %s" % _file)
            _df = pd.read_csv(_file)
            combined_df = pd.merge_ordered(
                combined_df, _df, fill_method="ffill")

    # Drop duplicates
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    if "Time" in combined_df.columns.values:
        combined_df['Time_ms'] = 1e3 * combined_df['Time']

    # export to csv
    combined_df.to_csv("%s_merged.csv" %
                       prefix, index=False, encoding='utf-8-sig')
    return combined_df


def postproc_integral():
    # Get surface integrals
    A_tot_files = sorted(glob.glob(input_file['isoc']['input_path_int_tot']))
    df_A_tot = read_integral(A_tot_files)
    print(df_A_tot)

    A_res_files = sorted(glob.glob(input_file['isoc']['input_path_int_res']))
    df_A_res = read_integral(A_res_files)
    print(df_A_res)

    if input_file['isoc']['input_path_int_norm_grac_c']:
        A_res_grad_c = sorted(
            glob.glob(input_file['isoc']['input_path_int_norm_grac_c']))
        df_res_grad_c = read_integral(A_res_grad_c)
        print(df_res_grad_c)

    plt.figure()
    plt.plot(df_A_res.Time_ms, df_A_res.Area, label='Resolved')
    plt.plot(df_A_res.Time_ms, df_A_tot.efcy -
             df_A_res.Area, "--", label='SGS')
    plt.plot(df_A_tot.Time_ms, df_A_tot.efcy, "-.", label='Total')
    if input_file['isoc']['input_path_int_norm_grac_c']:
        plt.plot(df_res_grad_c.Time_ms, -df_res_grad_c.norm_grad_c,
                 'o', alpha=0.5, label=r'$\int_V |\vec{\nabla} c|dV$')
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


def read_stats(lst_files):
    prefix, _ = os.path.split(lst_files[0])
    logger.info(prefix)

    def get_df_min_max(_file):
        """Return min max dict only"""
        df_raw = pd.read_csv(_file)

        min_max = {}
        for col in df_raw.columns.values:
            min_max['%s_min' % col] = [df_raw[col].min()]
            min_max['%s_max' % col] = [df_raw[col].max()]

        return pd.DataFrame.from_dict(min_max)

    combined_df = get_df_min_max(lst_files[0])
    for _idx, _file in tqdm.tqdm(enumerate(lst_files)):
        if _idx:
            # print("\t > %s" % _file)
            _df = get_df_min_max(_file)
            combined_df = pd.merge_ordered(
                combined_df, _df, fill_method="ffill")

    # Drop duplicates
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    if "Time_min" in combined_df.columns.values:
        combined_df['Time_min_ms'] = 1e3 * combined_df['Time_min']

    # export to csv
    combined_df.to_csv("%s_merged.csv" %
                       prefix, index=False, encoding='utf-8-sig')
    return combined_df


def postproc_isoc_stats():
    logger.info("Begin postproc isoc stats")
    isoc_stats = sorted(glob.glob(input_file['isoc']['input_path_stats']))
    logger.info("Stat files: ", isoc_stats)

    df_st = read_stats(isoc_stats)

    logger.info(df_st)

    # Fig max pos vs t
    # Mean Pressure and combustion rate
    fig_pos, ax_pos = plt.subplots(1, 1, sharex=True,
                                   sharey=False)

    ax_pos.plot(df_st.Time_min_ms, df_st.X_max, label=r"$\max \left( x(c = c^*) \right)$")
    ax_pos.plot(df_st.Time_min_ms, df_st.Y_max, label=r"$\max \left( y(c = c^*) \right)$")
    ax_pos.plot(df_st.Time_min_ms, df_st.Z_max, label=r"$\max \left( z(c = c^*) \right)$")
    ax_pos.legend()

    ax_pos.set_xlabel('Time [ms]')
    ax_pos.set_xlim(left=0)
    ax_pos.set_ylabel('Maximum position [m]')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for ext in ['pdf', 'png']:
        fig_pos.savefig(os.path.join(input_file['isoc']['output_path'],
                                     'Fig_pos.%s' % ext),
                        bbox_inches='tight',
                        pad_inches=0.01)
    # Fig max pos vs t
    # Mean Pressure and combustion rate
    fig_speed, ax_speed = plt.subplots(1, 1, sharex=True,

                                   sharey=False)

    df_st['X_max_smooth'] = df_st['X_max'].rolling(window=51).mean()
    df_st['Y_max_smooth'] = df_st['Y_max'].rolling(window=51).mean()
    df_st['Z_max_smooth'] = df_st['Z_max'].rolling(window=51).mean()

    df_st['speed_X_max'] = np.gradient(df_st.X_max.rolling(window=51).mean(), df_st.Time_min.rolling(window=51).mean())
    df_st['speed_Y_max'] = np.gradient(df_st.Y_max.rolling(window=51).mean(), df_st.Time_min.rolling(window=51).mean())
    df_st['speed_Z_max'] = np.gradient(df_st.Z_max.rolling(window=51).mean(), df_st.Time_min.rolling(window=51).mean())

    ax_speed.plot(df_st.X_max, np.gradient(df_st.X_max, df_st.Time_min), color='C0', alpha=0.2)
    ax_speed.plot(df_st.Y_max, np.gradient(df_st.Y_max, df_st.Time_min), color='C1', alpha=0.2)
    ax_speed.plot(df_st.Z_max, np.gradient(df_st.Z_max, df_st.Time_min), color='C2', alpha=0.2)
    ax_speed.plot(df_st.X_max_smooth, df_st.speed_X_max, color='C0', label="x")
    ax_speed.plot(df_st.Y_max_smooth, df_st.speed_Y_max, color='C1', label="y")
    ax_speed.plot(df_st.Z_max_smooth, df_st.speed_Z_max, color='C2', label="z")
    ax_speed.legend()

    ax_speed.set_xlabel('Maximum position [m]')
    ax_speed.set_xlim(left=0)
    ax_speed.set_ylabel('Absolute flame speed [m/s]')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for ext in ['pdf', 'png']:
        fig_speed.savefig(os.path.join(input_file['isoc']['output_path'],
                                     'Fig_abs_flame_speed.%s' % ext),
                        bbox_inches='tight',
                        pad_inches=0.01)

    if SHOW:
        plt.show()
    pass


if __name__ == "__main__":
    logger.info("Begin")

    if PLOT_INTG:
        postproc_integral()

    if PLOT_STATS:
        postproc_isoc_stats()
