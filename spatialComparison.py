# -*- coding: utf-8 -*-
"""
Functions and Classes to compare and cluster
spatial gene distributions

nigel Aug 2019
"""

import os
import pickle
import pprint as pp
import datetime
import seaborn as sns
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase

from sklearn.metrics import mutual_info_score
from skimage import measure
from scipy import stats
from numpy.linalg import norm

from typing import Tuple, Union
from smoothing import SmoothImgs, plotALot
from _utils import parse_folder, read_h5py, get_abundant_genes

#
# ====================================================================
#             Spatial comparision classes and functions
# @author: Nigel (Aug 2019)
# ====================================================================
# 
def _checkShape(x: np.ndarray, y: np.ndarray):
    """
    check if shapes of 2 arrays match
    """
    assert x.shape == y.shape, (f"Dimensions of image 1 {x.shape} "
                                f"do not match image 2 {y.shape}")

#
# ====================================================================
#                          Distance Metrics
# ====================================================================
#

def calcImageMI(x, y, bins):
    """
    adapted from stackexchange
    """
    _checkShape(x, y)

    c_xy = np.histogram2d(x.ravel(), y.ravel(),
                          bins)[0]

    return mutual_info_score(None, None,
                             contingency=c_xy)


def calcMIcustom(x, y, bins):
    """
    own version
    """
    _checkShape(x, y)

    c_xy = np.histogram2d(x.ravel(), y.ravel(),
                          bins)[0]
    # assert sum(c_xy)==1,"cxy does not sum to 1"

    p_xy = c_xy / np.sum(c_xy)

    # nonzero mask
    nz = p_xy > 0
    # marginal probabilities
    p_x = np.sum(p_xy, axis=1, keepdims=True)
    p_y = np.sum(p_xy, axis=0, keepdims=True)
    px_py = p_x * p_y

    return np.sum(
        p_xy[nz] * np.log2(p_xy[nz] / px_py[nz])
    )


def normXCorr(x, y):
    """
    Nigel: seems to be identical to pearson correlation
    
    Xinrui: NCC does not substract the local mean value of intensities
    NCC is the same as cos(x,y), maybe the code needs amendment.
    
    """
    _checkShape(x, y)


#    return np.mean(
#        (x - np.mean(x)) * (y - np.mean(y)) /
#        (np.std(x) * np.std(y))
#    )
    
    # - Xinrui 
    return np.mean((x*y)/(np.std(x)*np.std(y)))
    # cosine distance
#    return np.dot(x,y)/(norm(x)*norm(y))
    


def pearson(x, y):
    """
    use scipy stat's pearsonr function
    """
    _checkShape(x, y)

    return stats.pearsonr(x.flat, y.flat)[0]


def _rawToPDF(x, y):
    """
    convert raw values to probability distribution
    """
    return (x / np.sum(x), y / np.sum(y))


def _kl(x_prob, y_prob):
    """
    core function for KL-divergence (used in both JS and KL)
    """
    nonzero_mask = x_prob != 0
    x_prob = x_prob[nonzero_mask]
    y_prob = y_prob[nonzero_mask]

    return np.sum(
        x_prob * np.log2(x_prob / y_prob)
    )


def kl(x, y):
    """
    Kullback-Leibler divergence
    of x with reference to y
    NOTE: this is not symmetric!
    FIXME: this won't work when one distribution has zeros
    
    """
    _checkShape(x, y)
    assert 0 not in y, ("zero values found in second array")

    x_prob, y_prob = _rawToPDF(x, y)

    return _kl(x_prob, y_prob)


def js(x, y):
    """
    Jensen-shannon divergence
    """
    _checkShape(x, y)

    x_prob, y_prob = _rawToPDF(x, y)
    # _checkShape(x_prob, y_prob)

    M = (x_prob + y_prob) / 2

    return (_kl(x_prob, M) + _kl(y_prob, M)) / 2


def ssim(x, y):
    """
    scikit image's ssim metric

    FIXME: not really sure what this does yet -nigel
    """
    _checkShape(x, y)

    return measure.compare_ssim(x, y)


#
# ====================================================================
#          THE MAIN CLASS for selecting gene panel
# ====================================================================
#

class AnalyzeSpots(object):

    def __init__(self,
                 smoothImg: SmoothImgs,
                 ):
        """
        Parameters
        ----------
            

        """

        self.bin_size = smoothImg.bin_size
        self.sigma = smoothImg.sigma
        self.annotation_csv = smoothImg.annotation_csv
        self.genes_to_exclude = smoothImg.genes_to_exclude
        
        self.gene_index_dict = smoothImg.gene_index_dict
        self.num_genes = smoothImg.num_genes
        self.max_coords = smoothImg.max_coords
        
        self.image_shape = smoothImg.image_shape
        self.bin_edges = smoothImg.bin_edges
        self.img_array = smoothImg.smoothed_img_array
        self.mean_array = smoothImg.smoothed_mean_array
        
        self.verbose = smoothImg.verbose
        
        #
        # create a list of genes in index order
        # -------------------------------------
        self.ordered_genes = [None, ] * self.num_genes
        for gene in self.gene_index_dict:
            self.ordered_genes[self.gene_index_dict[gene]['index']] = gene
#        if self.verbose:
#            print(f"Ordered list of genes:\n")
#            pp.pprint(self.ordered_genes)
            
        # list of currently implemented methods
        self.metric_list = ["mutual information",
                            "normalized crosscorr",
                            "pearson",
                            "JS divergence",
                            "SSIM",
                            ]
        # FIXME: normalized cross corr is the same as pearson
        # FIXME: maybe should combine them?
        # Xinrui: NCC does not substract the local mean value of intensities

        # get time at which this script was started
        self.script_time = datetime.datetime.now()
        self.time_str = self.script_time.strftime("%Y%m%d_%H%M")
        

    def _checkMetricImplemented(self,
                                metric: str,
                                ) -> None:
        """
        raise error if the metric being called has not been implemented
        """
        if metric not in self.metric_list:
            raise NotImplementedError(
                f"Method {metric} not recognised.\n"
                f"Possible methods are:\n" +
                "\n".join([f" ({num+1}) {metric}"
                           for num, metric
                           in enumerate(self.metric_list)])
            )

    def _makeRowColours(self,
                        annotation_series: pd.Series,
                        palette: str = "Set1",  # palette for categorical data
                        cmap=cm.Blues,  # colormap for numeric data
                        verbose=True,
                        ) -> Tuple[pd.Series,
                                   Union[dict, tuple],
                                   np.ndarray]:
        """
        take a series of numerical or categorical values and
        map to a number of colours, creating
        a row or column colour series recognised by sns.clustermap

        return tuple of:
        (1) row-colour pandas series. this goes into sns.clusterplot's row_color
        (2) either a look-up table dictionary (for categorical columns)
            or     a tuple of min and max values (for numerical columns)
        (3) list of unique labels
        """

        # check if series is numerical type
        # ---------------------------------

        if is_numeric_dtype(annotation_series):
            # if np.issubdtype(annotation_series.dtype, np.number):
            annotation_series = annotation_series.fillna(0)
            series_min = annotation_series.min()
            series_max = annotation_series.max()
            norm = Normalize(vmin=series_min, vmax=series_max,
                             clip=True)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            labels = annotation_series.unique()

            return annotation_series.apply(mapper.to_rgba), (norm, cmap), labels

        # else assume series is categorical
        # ---------------------------------

        else:
            annotation_series = annotation_series.fillna("unknown")
            labels = annotation_series.unique()

            lut = dict(
                zip(labels,
                    # sns.hls_palette(len(labels), l=0.5, s=0.8)
                    sns.color_palette(palette,
                                      n_colors=len(labels),
                                      desat=.5))
            )

            if verbose:
                print(f"Look-up table for annotation colours:\n")
                pp.pprint(lut)

            return annotation_series.map(lut), lut, labels

    def clusterPlot(self,
                    metric: str = "mutual information",
                    metric_kws: dict = None,
                    symmetric: bool = True,  # this should usually be True??
                    self_correlation: bool = False,
                    savedir: str = "",
                    verbose: bool = True,
                    plot_unclustered: bool = False,
                    col_for_legend: str = "zone",
                    col_for_cbar: str = "number of probes",
                    ):
        """
        calculate, plot and cluster the correlation matrix
        """
        self._checkMetricImplemented(metric)

        corr_matrix = np.empty((self.num_genes, self.num_genes),
                               dtype=np.float64)
        corr_matrix.fill(np.nan)
        

        # the main title of the plot
        title = (f"Clustered Pairwise {metric}\n"
                 f"bin size = {self.bin_size[0]} pix "
                 f"by {self.bin_size[1]} pix")
        if metric == "mutual information":
            title += f"\n(bins = {metric_kws['bins']})"

        #
        # Set diagonal fill values for given metric
        # -----------------------------------------
        #
        
        if metric in ["mutual information",
                      "JS divergence", "normalized crosscorr",
                      "pearson", "SSIM"]:
            fill_diagonal = 0.
        else:
            fill_diagonal = 0.
        # FIXME: might want to set diagonals values differently for some metrics.
        
        
        for gene1 in self.gene_index_dict:
            gene1_idx = self.gene_index_dict[gene1]['index']
            for gene2 in self.gene_index_dict:
                gene2_idx = self.gene_index_dict[gene2]['index']

                # ------------------------------
                # skip self-correlation of genes
                # ------------------------------
                if not self_correlation and gene2_idx == gene1_idx:
                    corr_matrix[gene1_idx, gene2_idx] = fill_diagonal
                    continue

                #
                # skip if already calculated
                # --------------------------
                if not np.isnan(corr_matrix[gene1_idx, gene2_idx]):
                    continue
                
                gene1_array = self.img_array[gene1_idx, ...]
                gene2_array = self.img_array[gene2_idx, ...]

                if verbose:
                    print(f"Gene 1 {gene1} array shape: {gene1_array.shape}\n"
                          f"Gene 2 {gene2} array shape: {gene2_array.shape}\n")

                #
                # run metric of choice
                # --------------------
                #

                if metric == "mutual information":
                    dist = calcImageMI(gene1_array, gene2_array,
                                       bins=metric_kws["bins"])
                    
                    # dist = calcMIcustom(gene1_array, gene2_array,
                    #                    bins=metric_kws["bins"])
                elif metric == "normalized crosscorr":
                    dist = normXCorr(gene1_array, gene2_array, )
                elif metric == "pearson":
                    dist = pearson(gene1_array, gene2_array, )
                elif metric == "JS divergence":
                    dist = js(gene1_array, gene2_array, )
                elif metric == "SSIM":
                    dist = ssim(gene1_array, gene2_array, )
                else:
                    raise ValueError(f"Metric {metric} not recognised")

                # enter result into correlation matrix
                corr_matrix[gene1_idx, gene2_idx] = dist
                if symmetric:
                    corr_matrix[gene2_idx, gene1_idx] = dist

        #
        # set plotting params for given metrics
        # -------------------------------------
        #
        print(f'plotting for {metric} ...')
        # these metrics start from 0
        if metric in ["mutual information",
                      "JS divergence"]:
            center = None
            cmap = "hot"

        # these metrics can have +ve and -ve values
        elif metric in ["normalized crosscorr",
                        "pearson", "SSIM"]:
            center = 0
            cmap = "vlag"

        # default settings
        else:
            center = None
            cmap = "hot"

        if verbose:
            print(f"Filled correlation matrix:\n"
                  f"{corr_matrix}")

        #
        # set up row colours
        # ------------------
        #

        if self.annotation_csv is not None:

            #
            # parse annotation csv file into a dataframe
            # ------------------------------------------
            #

            genes_df = pd.DataFrame({"name": self.ordered_genes}, )
            # print(f"genes dataframe initial:\n {genes_df}")
            self.annotation_df = pd.read_csv(self.annotation_csv)
            # print(f"annotation dataframe:\n {self.annotation_df}")
            self.genes_df = genes_df.merge(self.annotation_df,
                                           how="left", on="name",
                                           copy=False)
            self.genes_df.set_index("name", inplace=True)

            if verbose:
                print(f"genes dataframe final:\n "
                      f"{self.genes_df}\n{self.genes_df.dtypes}")

            #
            # Generate Row-colours
            # --------------------
            # generate row-colours dataframe,
            # also saving lut dictionary and category labels
            #

            row_colours = []
            lut_labels_dict = {}

            for col in self.genes_df.columns:
                # sequential palette for numerical annotations
                if col in ["number of probes", ]:
                    row_colour, lut, labels = self._makeRowColours(self.genes_df[col],
                                                                   "Blues")
                else:
                    row_colour, lut, labels = self._makeRowColours(self.genes_df[col],
                                                                   "Set1")
                row_colours.append(row_colour)
                lut_labels_dict[col] = (lut, labels)

            row_colours = pd.concat(row_colours, axis=1)

            if verbose:
                print(f"Dict of Look-up table and labels:\n")
                pp.pprint(lut_labels_dict)

                print(f"Row colours df:\n"
                      f"{row_colours}\n{row_colours.dtypes}")

        else:
            row_colours = None

        #
        # Plot clustermap
        # ---------------
        #

        sns.set_style("dark")

        # plot heatmap without clustering
        if plot_unclustered:
            fig_mat, ax_mat = plt.subplots(figsize=(9, 9))
            sns.heatmap(corr_matrix,
                        ax=ax_mat,
                        square=True,
                        yticklabels=self.ordered_genes,
                        xticklabels=self.ordered_genes
                        )

        g = sns.clustermap(pd.DataFrame(data=corr_matrix,
                                        index=self.ordered_genes,
                                        columns=self.ordered_genes
                                        ),
                           square=False,
                           yticklabels=True,
                           xticklabels=True,
                           # yticklabels=self.ordered_genes,
                           # xticklabels=self.ordered_genes,
                           center=center,
                           cmap=cmap,
                           row_colors=row_colours,
                           )

        #
        # set up annotation legend and/or colorbar
        # ----------------------------------------
        #

        if self.annotation_csv is not None:
            lut, labels = lut_labels_dict[col_for_legend]

            # create a new axes on the bottom right of plot
            labels_ax = g.fig.add_axes([0.4, 0.01, 0.58, 0.08])
            labels_ax.axis('off')

            for label in labels:
                labels_ax.bar(0, 0, color=lut[label],
                              label=label, linewidth=0)
            labels_ax.legend(loc="lower right", ncol=4,
                             frameon=False,
                             )

            cbar_params, labels = lut_labels_dict[col_for_cbar]
            norm, cmap = cbar_params

            # create a new axes on the bottom right of plot
            cbar_ax = g.fig.add_axes([0.35, 0.04, 0.15, 0.01])

            cb1 = ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                               orientation='horizontal')
            cb1.set_label(col_for_cbar)

        g.ax_heatmap.tick_params(axis='both',
                                 which='major',
                                 labelsize=5)
        g.fig.subplots_adjust(top=0.94)
        g.fig.suptitle(title, fontsize=12, fontweight="bold")
        
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        filename = (f"{metric.replace(' ','_')}"
                       f"_clusterplot.png")
        g.fig.savefig(os.path.join(savedir, filename), dpi=600, )

        # list of genes in order shown on clustermap
        reordered_genes = [self.ordered_genes[idx]
                           for idx in
                           g.dendrogram_row.reordered_ind]
        if verbose:
            print(f"Reorderd indices:\n"
                  f"{g.dendrogram_row.reordered_ind}\n"
                  f"Reordered genes:\n{reordered_genes}")
        
        g.fig.clear()
        plt.close()
        return corr_matrix, reordered_genes
    
#----------------------------------------------------------------
#  MAIN SCRIPT
#----------------------------------------------------------------
if __name__ == '__main__':
    
    script_time = datetime.datetime.now()
    time_str = script_time.strftime("%Y%m%d_%H%M")
    
    # -------------------------------------------------------
    # Ask user for the folder with datasets
    # -------------------------------------------------------
    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please select directory with hdf5 files")
    # data_path = filedialog.askopenfilename(title="Please select hdf file")
    root.destroy()
    
    norm_method = "no_norm"
    for root, dirs, files in os.walk(data_path):
        print(f"Current dir: {root}")
        hdf5_file_list, annotation_csv, genes_to_exclude = parse_folder(root)
        
        # --- Analysing each coord*.hdf5 file in current dir ---
        for hdf5_file in hdf5_file_list:
            gene_index_dict, raw_max_coords, num_genes, blank_genes = \
            read_h5py(hdf5_file, verbose = False, 
                      genes_to_exclude = genes_to_exclude)
            
            
            spatialcompdir = os.path.join(os.path.dirname(hdf5_file),
                                          f"spatialComp_{norm_method}_{time_str}")
            if not os.path.exists(spatialcompdir):
                os.makedirs(spatialcompdir)
            
            
            # ---------------------------
            # Smoothing -----------------
            # ---------------------------
            # parameters
            bin_size = 60
            sigma = 60
            
            smooth_imgs = SmoothImgs(gene_index_dict,
                                     raw_max_coords,
                                     num_genes, 
                                     spatialcompdir,
                                     bin_size= bin_size,
                                     norm_method= norm_method,
                                     sigma = sigma,
                                     verbose=False)
            
            
            # Spatial comparison (Optional) 
            # ------------------------------------
            
            # Running only one metric, e.g JS divergence
            aspots = AnalyzeSpots(smooth_imgs)
            metric = 'JS divergence'

            matrix, reordered_genes = \
            aspots.clusterPlot(metric = 'JS divergence', 
                               self_correlation = False, 
                               savedir=spatialcompdir, 
                               verbose = False, 
                               col_for_cbar = 'Number of probs',
                               col_for_legend = 'Zone',
                               )
            
            matrix_filename = os.path.join(spatialcompdir, 
                                           f"{metric.replace(' ','_')}"
                                           f"_matrix.pkl")
            with open(matrix_filename, 'wb') as f:
                pickle.dump(matrix, f)
            reordered_genes_filename = os.path.join(spatialcompdir, 
                                                    f"{metric.replace(' ','_')}"
                                                    f"_reordered_genes.pkl")
            with open(reordered_genes_filename, 'wb') as f:
                pickle.dump(reordered_genes, f)
                
            # Smooth and plot images (check sigma)
            plotALot(img_array = smooth_imgs.smoothed_img_array,
                     gene_index_dict = smooth_imgs.gene_index_dict,
                     reordered_genes = reordered_genes,
                     savedir = spatialcompdir, 
                     title = f'smoothed {norm_method} (binsize = {bin_size} sigma= {sigma})')
            
            bin_size = 12
            sigma = 60
            
            smooth_imgs = SmoothImgs(gene_index_dict,
                                     raw_max_coords,
                                     num_genes, 
                                     spatialcompdir,
                                     bin_size= bin_size,
                                     norm_method= norm_method,
                                     sigma = sigma,
                                     verbose=False)
            
            plotALot(img_array = smooth_imgs.smoothed_img_array,
                     gene_index_dict = smooth_imgs.gene_index_dict,
                     reordered_genes = reordered_genes,
                     savedir = spatialcompdir, 
                     title = f'smoothed {norm_method} (binsize = {bin_size} sigma= {sigma})')
            
#            # Running all implemented matrix
#            for metric in aspots.metric_list:
#                print(f'------------ {metric} ---------------')
#                if metric in ['normalized crosscorr', 
#                              'pearson', 
#                              'SSIM'] :
#                    matrix, reordered_genes = \
#                    aspots.clusterPlot(metric = metric, 
#                                       self_correlation = False,
#                                       savedir=corr_path, 
#                                       verbose = False,
#                                       )
#                elif metric == 'mutual information':
#                    matrix, reordered_genes = \
#                    aspots.clusterPlot(metric = metric, 
#                                       metric_kws = {'bins': 50},
#                                       self_correlation = False,
#                                       savedir=corr_path, 
#                                       verbose = False,
#                                       )
#                elif metric == 'JS divergence':
#                    matrix, reordered_genes = \
#                    aspots.clusterPlot(metric = 'JS divergence', 
#                                       self_correlation = False, 
#                                       savedir=corr_path, 
#                                       verbose = False, 
#                                       col_for_cbar = 'Number of probs',
#                                       col_for_legend = 'Zone',
#                                       )
#                
#                
#                # Save variables to file
#                # ----------------------------------
#                matrix_filename = os.path.join(corr_path, 
#                                               f"{metric.replace(' ','_')}"
#                                               f"_matrix.pkl")
#                with open(matrix_filename, 'wb') as f:
#                    pickle.dump(matrix, f)
#                reordered_genes_filename = os.path.join(corr_path, 
#                                                        f"{metric.replace(' ','_')}"
#                                                        f"_reordered_genes.pkl")
#                with open(reordered_genes_filename, 'wb') as f:
#                    pickle.dump(reordered_genes, f)
                # -----------------------------------
            
            # ------- End of spatial comparison --------------------