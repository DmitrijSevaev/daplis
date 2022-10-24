""" Module with scripts for calculating and plotting fits of the peaks in
the timestamp differences for LinoSPAD2 data.

This script utilizes an unpacking module used specifically for the LinoSPAD2
data output.

This file can also be imported as a module and contains the following
functions:

    * gauss_fit - function for fitting the peak in the timestamp differences
    with a gaussian function.

"""

import os
import glob
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
from functions import unpack as f_up


def fit_gauss(path, pix, timestamps: int = 512, show_fig: bool = False):
    '''
    Function for fitting the peak in the timestamp differences with a
    gaussian function.

    Parameters
    ----------
    path : str
        Path to the data files.
    pix : array-like
        DESCRIPTION.
    timestamps : int, optional
        Number of timestamps per acquisition cycle per pixel. The default
        is 512.
    show_fig : bool, optional
        Switch for showing the plot. The default is False.

    Returns
    -------
    None.

    '''

    def gauss(x, A, x0, sigma):
        return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

    os.chdir(path)

    DATA_FILES = glob.glob("*.dat*")

    for num, filename in enumerate(DATA_FILES):
        print("=========================================\n"
              "Fitting with gauss, Working on {}\n"
              "=========================================".format(filename))
        data = f_up.unpack_binary_flex(filename)

        data_1 = data[pix[0]]  # 1st pixel
        data_2 = data[pix[1]]  # 2nd pixel
        data_3 = data[pix[2]]  # 3d pixel
        data_4 = data[pix[3]]  # 4th pixel
        data_5 = data[pix[4]]  # 5th pixel

        pix_num = np.arange(pix[0], pix[-1]+1, 1)

        all_data = np.vstack((data_1, data_2, data_3, data_4, data_5))

        plt.rcParams.update({'font.size': 20})
        fig, axs = plt.subplots(4, 4, figsize=(24, 24))

        for q in range(5):
            for w in range(5):
                if w <= q:
                    continue

                data_pair = np.vstack((all_data[q], all_data[w]))

                minuend = len(data_pair)
                timestamps_total = len(data_pair[0])
                subtrahend = len(data_pair)
                timestamps = timestamps

                output = []

                for i in tqdm(range(minuend)):
                    acq = 0  # number of acq cycle
                    for j in range(timestamps_total):
                        if data_pair[i][j] == -1:
                            continue
                        if j % timestamps == 0:
                            acq = acq + 1  # next acq cycle
                        for k in range(subtrahend):
                            if k <= i:
                                continue  # to avoid repetition: 2-1, 53-45
                            for p in range(timestamps):
                                n = timestamps*(acq-1) + p
                                if data_pair[k][n] == -1:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] > 2.5e3:
                                    continue
                                elif data_pair[i][j] - data_pair[k][n] < -2.5e3:
                                    continue
                                else:
                                    output.append(data_pair[i][j]
                                                  - data_pair[k][n])

                if "Ne" and "540" in path:
                    chosen_color = "seagreen"
                elif "Ne" and "656" in path:
                    chosen_color = "orangered"
                elif "Ar" in path:
                    chosen_color = "mediumslateblue"
                else:
                    chosen_color = "salmon"

                if show_fig is True:
                    plt.ion()
                else:
                    plt.ioff()

                try:
                    bins = np.arange(np.min(output), np.max(output),
                                     17.857)
                except Exception:
                    continue

                plt.xlabel('\u0394t [ps]')
                plt.ylabel('Timestamps [-]')
                n, b, p = plt.hist(output, bins=bins, color=chosen_color)

                try:
                    n_max = np.argmax(n)
                    arg_max = (bins[n_max] + bins[n_max + 1]) / 2
                except Exception:
                    arg_max = None
                    pass

                sigma = 200

                par, covariance = curve_fit(gauss, b[:-1], n, p0=[max(n),
                                                                  arg_max,
                                                                  sigma])
                fit_plot = gauss(b, par[0], par[1], par[2])

                plt.figure(figsize=(16, 10))
                plt.xlabel('\u0394t [ps]')
                plt.ylabel('Timestamps [-]')
                plt.plot(b[:-1], n, 'o', color=chosen_color, label="data")
                plt.plot(b, fit_plot, '-', color="cadetblue", label="fit\n"
                         "\u03C3={}".format(par[-1]))
                plt.legend(loc='best')

                try:
                    os.chdir("results/gauss_fit")
                except Exception:
                    os.mkdir("results/gauss_fit")
                    os.chdir("results/gauss_fit")
                plt.savefig("{file}_pixels"
                            "{pix1}-{pix2}_fit.png".format(file=filename,
                                                           pix1=pix_num[q],
                                                           pix2=pix_num[w]))
                plt.pause(0.1)
                os.chdir("../..")