import numpy as np
import numpy.random as random
import statistics
import matplotlib.pyplot as plt
from matplotlib import colors

def monte_carlo(flux=5.6e-2, D=0.4):
    b = random.normal(-2.66, 0.06, 1000)
    m = random.normal(0.91, 0.06, 1000)
    flux_al = random.normal(flux, 0.1 * flux, 1000)
    L_bol_al = []

    lum = flux_al * (D ** 2)
    for i in range(1000):
        L_bol_al.append((np.log10(lum[i]) - b[i]) / m[i])
    L_bol_mean = statistics.mean(L_bol_al)
    L_bol_std = statistics.stdev(L_bol_al)
    return L_bol_al, L_bol_mean, L_bol_std


def plot_hist(flux, arr, mean, typ='exponent'):
    N, bins, patches = plt.hist(arr, bins=100)
    max_index = np.argmax(N)
    max_value = bins[max_index]

    fracs = N / N.max()
    norm = colors.Normalize(fracs.min(), fracs.max())

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='average')
    plt.axvline(max_value, color='b', linestyle='dashed', linewidth=1, label='most frequent')
    plt.legend()
    if typ == 'exponent':
        plt.xlabel('exponent of L_bol')
        plt.ylabel('frequency')
        print(f'most frequent result for {flux} mJy is: {max_value}')
    elif typ == 'mass':
        plt.xlabel('mass of the object')
        plt.ylabel('frequency')
        print(f'most frequent result for {flux} mJy is: {max_value} solar mass')
    return max_value

def find_flux(flux, lam_in, lam_out, exp=-0.1):
    f_in = 3.0e8/lam_in
    f_out = 3.0e8/lam_out
    return flux*(f_out/f_in)**(exp)