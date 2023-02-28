""" Script to estimate the slope of a passing wave
on a position-time plot
"""
# Github: hubernikus

import numpy as np
from numpy import linalg

import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn import mixture


class WaveDetector:
    def __init__(self, data_file, peak_threshold=230):
    # def __init__(self, data_file, peak_threshold=30):    
        self.correlation = np.genfromtxt(data_file, delimiter=',')

        self.n_pos = self.correlation.shape[0]
        self.n_time = self.correlation.shape[1]

        # Extract wave tip points with corresponding range of [0, 1] for position and time
        self.ind_siginificant = (self.correlation > peak_threshold)
        # self.ind_siginificant = (self.correlation < peak_threshold)

        # Step-size of 1 between position and time worked out to be optimal
        self.x_lim = [0, self.n_time]
        self.y_lim = [0, self.n_pos]
        
        time_vals, pos_vals = np.meshgrid(
            np.linspace(self.x_lim[0], self.x_lim[-1], self.n_time),
            np.linspace(self.y_lim[0], self.y_lim[-1], self.n_pos),
        )

        self.data = np.vstack((
            time_vals.flatten(),
            pos_vals.flatten(),
            # time_vals.flatten(),
        ))[:, self.ind_siginificant.flatten()]

        # self.data = self.data.T

        self.do_gmm_clustering()
        
    def do_gmm_clustering(self, n_gmm=5):
        self.gmm = mixture.GaussianMixture(
            n_components=n_gmm,
            covariance_type="full",
            max_iter=100
        ).fit(self.data.T)


    def do_slope_estimation(self, rel_weight_margin=0.1, color='#ADD8E6', plot_results=False):
        # TODO: Pre-processing to remove ellipses which are in the upper left or bottom
        
        max_weight = np.max(self.gmm.weights_)

        if plot_results:
            # Plot only relevant clusters
            fig, ax = plt.subplots()

        self.slopes = []
        self.weights = []
        
        for ii, (mean, covar) in enumerate(zip(self.gmm.means_, self.gmm.covariances_)):
            # Neglect clusteres with only few points
            if self.gmm.weights_[ii] < rel_weight_margin * max_weight:
                continue
            

            # Eigen values are in ascending order -> last eigenvector is important!
            eig_vals, eig_vecs = linalg.eigh(covar)

            eig_vals = 2.0 * np.sqrt(2.0) * np.sqrt(eig_vals)
            uu = eig_vecs[0] / linalg.norm(eig_vecs[0])

            if any(eig_vecs[-1, :] < 1e-3):
                continue
            self.weights.append(self.gmm.weights_[ii])
            self.slopes.append(eig_vecs[-1, :] / linalg.norm(eig_vecs[-1, :]))
            # self.slopes.append([-uu[1], uu[0]])

            if plot_results:
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan2(uu[1], uu[0])
                angle = 180.0 * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, eig_vals[0], eig_vals[1], angle, color=color)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)
        
        # Get mean
        self.slopes = np.array(self.slopes).T

        # Check which slopes are pointing downwards
        ind_neg = (self.slopes[1, :] < 0)
        self.slopes[:, ind_neg] *= (-1)

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)
        
        self.mean_slope = np.sum(
            self.slopes * np.tile(self.weights, (self.slopes.shape[0], 1)), axis=1
        )

        slope = self.mean_slope[1] / self.mean_slope[0]

        if plot_results:
            ax.scatter(self.data[0, :], self.data[1, :], color='k')
            # ax.scatter([], [], color='k')
            self.x_lim = ax.get_xlim()
            self.y_lim = ax.get_ylim()

            y1 = self.y_lim[0] + (self.x_lim[1] - self.x_lim[0]) * slope

            ax.plot(self.x_lim, np.array([self.y_lim[0], y1]) + 0, '--', color='r',
                    linewidth=2, label=f"{round(slope, 6)} m / s")

            ax.set_xlim(self.x_lim)
            ax.set_ylim(self.y_lim)

            ax.legend()

        return slope

    def plot_all_clusters(self, color='#ADD8E6'):
        fig, ax = plt.subplots()

        ax.scatter(self.data[0, :], self.data[1, :], color='k')
        
        for ii, (mean, covar) in enumerate(zip(self.gmm.means_, self.gmm.covariances_)):
            v, w = linalg.eigh(covar)
            v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
            u = w[0] / linalg.norm(w[0])
            
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180.0 * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180.0 + angle, color=color)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)
            ax.add_artist(ell)

def main():
    plt.close('all')
    plt.ion()
    my_detector = WaveDetector(data_file="corr_0_40.csv")
    # my_detector.plot_all_clusters()
    
    slope = my_detector.do_slope_estimation()
    print('slope', slope)
    
    # breakpoint()


if (__name__) == "__main__":
    main()
