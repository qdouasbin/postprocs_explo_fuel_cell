
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

# REFERENCE VALUES
PRESSURE_REF = 1.3e5  # Pa
VENT_AREA = 0.197872  # m^2
PATH_TEMPORAL = input_file["Temporal"]["input_path"]

# Otions
PLOT_PROBES = 1
PLOT_MMM = 1
SHOW = 0
OUTPATH = input_file["Temporal"]["output_path"]


def find_probe_files():
    files = sorted(glob.glob(os.path.join(PATH_TEMPORAL, "avbp_local_*.h5")))
    logger.info("Found %s probes" % len(files))
    return files


def plot_probes(probes, var='overp_mbar'):
    logger.info("Plot %s" % var)

    nb_x = 3
    nb_y = 5
    nb_z = 4
    fig, axes = plt.subplots(nb_x, nb_y, sharex=True,
                             sharey=True, figsize=(10, 6))

    for _x in range(nb_x):
        _idx_x = _x + 1
        for _y in range(nb_y):
            _idx_y = _y + 1
            ax = axes[_x, _y]
            # print((_idx_x, _idx_y))
            for _z in range(nb_z):
                df = probes['x%s_y%s_z%s' % (_x + 1, _y+1, _z+1)]
                ax.plot(1e3 * df.atime, df[var], label='z%s' % _z)
            ax.set_title("x%s, y%s" % (_idx_x, _idx_y),
                         fontsize=8, y=1.05, pad=-14)
            if (_x + 1, _y+1) == (nb_x, nb_y):
                ax.legend()

    # Label axes
    for _ax in axes[_x, :]:
        _ax.set_xlabel("Time [ms]")
        _ax.set_xlim(left=0)

    for _ax in axes[:, 0]:
        if var == 'overp_mbar':
            _ax.set_ylabel("Overpressure [mbar]")
            _ax.set_ylim(bottom=0)
        if var == 'P':
            _ax.set_ylabel("Pressure [Pa]")
        if var == 'dP_dT':
            _ax.set_ylabel("Pressure time derivative [Pa/s]")
        if var == 'T':
            _ax.set_ylabel("Temperature [K]")
        if var == 'u_mag':
            _ax.set_ylabel("Velocity magnitude [m/s]")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPATH,
                                 'Fig_probes_%s.%s' % (var, ext)),
                    bbox_inches='tight',
                    pad_inches=0.01)

    if SHOW:
        plt.show()


def postproc_probes(var='overp_mbar'):
    probe_files = find_probe_files()
    # logger.info(probe_files)

    probes = dict()
    for _idx, _file_name in enumerate(probe_files):
        lst_file = _file_name.replace('.h5', '').split('_')
        _key = '%s_%s_%s' % (lst_file[-3], lst_file[-2], lst_file[-1])
        # print(_file_name, _key)
        df = pd.read_hdf(_file_name)
        df['atime_filt'] = df['atime'].rolling(window=101).mean()
        df['P_filt'] = df['P'].rolling(window=101).mean()
        df['dP_dt'] = np.gradient(df.P_filt, df.atime_filt)
        df['overp_mbar'] = 1e-2 * (df['P'] - PRESSURE_REF)
        # plt.figure()
        # plt.plot(df.atime, df.P)
        # plt.plot(df.atime_filt, df.P_filt)

        # plt.figure()
        # plt.plot(df.atime_filt, df.dP_dt)
        # plt.show()
        df['u_mag'] = np.sqrt(
            np.square(df.u) + np.square(df.v) + np.square(df.w))
        probes[_key] = df
        if not _idx:
            logger.info(df.columns.values)
        del(lst_file, _key, df)

    for var in ['dP_dt', 'u_mag', 'P', 'overp_mbar', 'T']:
        plot_probes(probes, var)


def postproc_mmm():
    # Fresh and burnt gases data for phi=1, H2/AIR, T=110 C, P=1.3 bar
    rho_fg = 0.849545
    Y_H2_fg = 0.0285102
    rho_bg = 0.124278

    # Get mmm
    mmm_file = sorted(glob.glob(os.path.join(PATH_TEMPORAL, "avbp_mmm.h5")))
    df_mmm = pd.read_hdf(mmm_file[-1])

    spec_file = sorted(glob.glob(os.path.join(
        PATH_TEMPORAL, 'avbp_xmc_spec.h5')))
    df_spec = pd.read_hdf(spec_file[-1])

    # Compute V_Comb
    logger.info("Plot Vcomb")
    dVfg_dt = df_mmm.Volume * \
        (-1.0 * df_spec.source_term_comb_H2) / (rho_fg * Y_H2_fg)
    dVcomb_dt = dVfg_dt * (rho_fg / rho_bg - 1.0)

    # Mean Pressure and combustion rate
    fig, axes = plt.subplots(2, 1, sharex=True,
                             sharey=False, figsize=(4, 4))

    ax_pres, ax_dv = axes

    ax_pres.plot(1e3 * df_mmm.atime, 1e-2 * (df_mmm.P_mean - PRESSURE_REF))

    ax_dv.plot(1e3 * df_mmm.atime, dVfg_dt, label=r"$\dot{V}_{fg}$")
    ax_dv.plot(1e3 * df_mmm.atime, dVcomb_dt, label=r"$\dot{V}_{comb}$")
    ax_dv.legend()

    ax_dv.set_xlabel('Time [ms]')
    ax_dv.set_xlim(left=0)
    ax_dv.set_ylabel('Volumetric rate [m$^3$/s]')
    ax_pres.set_ylabel('Mean overpressure [mbar]')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPATH,
                                 'Fig_overp_Vcomb.%s' % ext),
                    bbox_inches='tight',
                    pad_inches=0.01)

    # VENT
    # Get df_vent
    logger.info("Plot Vent")
    vent_file = sorted(glob.glob(os.path.join(PATH_TEMPORAL, "avbp_flux.h5")))
    df_vent = pd.read_hdf(vent_file[-1])
    probe_vent_file = sorted(glob.glob(os.path.join(
        PATH_TEMPORAL, "avbp_local_probe_encaps_x2_y3_z4.h5")))
    df_pb_vent = pd.read_hdf(probe_vent_file[-1])

    fig, ax1_vent = plt.subplots()  # ax1_vent is the Pa scale
    ax2_vent = ax1_vent.twinx()     # ax2_vent is the mbar scale

    def update_ax2(ax1_vent):
        def Pa2mbar(press):
            return 1e3 * 1e-5 * press
        p1, p2 = ax1_vent.get_ylim()
        ax2_vent.set_ylim(Pa2mbar(p1), Pa2mbar(p2))
        ax2_vent.figure.canvas.draw()

    # automatically update ylim of ax2_vent when ylim of ax1_vent changes.
    ax1_vent.callbacks.connect("ylim_changed", update_ax2)

    ax1_vent.plot(1e3 * df_mmm.atime, df_mmm.P_mean -
                  PRESSURE_REF, '--', label='mean over the domain')
    ax1_vent.plot(1e3 * df_pb_vent.atime, df_pb_vent.P - PRESSURE_REF,
                  '-.s', markevery=0.1, label='probe in front of the vent')
    ax1_vent.plot(1e3 * df_vent.atime, df_vent.PFmean_VENT /
                  VENT_AREA - PRESSURE_REF, label='mean over the vent')

    ax1_vent.legend()

    ax1_vent.set_xlabel("Time [ms]")
    ax1_vent.set_xlim(left=0)
    ax1_vent.set_ylabel("Overpressure [Pa]")
    ax2_vent.set_ylabel("Overpressure [mbar]")
    ax2_vent.grid(False)
    plt.tight_layout()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for ext in ['pdf', 'png']:
        fig.savefig(os.path.join(OUTPATH,
                                 'Fig_vent.%s' % ext),
                    bbox_inches='tight',
                    pad_inches=0.01)

    # HR
    logger.info("Plot HR")
    fig_hr, ax_hr = plt.subplots()  # ax1_vent is the Pa scale
    ax_hr.plot(1e3 * df_mmm.atime, df_mmm.HR_mean)
    ax_hr.set_xlabel("Time [ms]")
    ax_hr.set_xlim(left=0)
    ax_hr.set_ylabel("Heat Release [$J.s^{-1}$]")
    plt.tight_layout()

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for ext in ['pdf', 'png']:
        fig_hr.savefig(os.path.join(OUTPATH,
                                    'Fig_HR.%s' % ext),
                       bbox_inches='tight',
                       pad_inches=0.01)

    if SHOW:
        plt.show()


if __name__ == "__main__":
    logger.info("Begin")

    if PLOT_MMM:
        postproc_mmm()

    if PLOT_PROBES:
        postproc_probes()
