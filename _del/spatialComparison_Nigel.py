# -*- coding: utf-8 -*-
"""
Functions and Classes to compare and cluster
spatial gene distributions

nigel Aug 2019
"""

import h5py
import tkinter as tk
import os
import pprint as pp
import datetime
import copy
import warnings
import seaborn as sns
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas

from sklearn.metrics import mutual_info_score
from skimage import measure
from scipy import stats
from tkinter import filedialog
from scipy.ndimage import gaussian_filter

from typing import Tuple, Dict, Union
from _utils import roundUp

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

#
#    return np.mean(
#        (x - np.mean(x)) * (y - np.mean(y)) /
#        (np.std(x) * np.std(y))
#    )
    
    # - Xinrui 
    return np.mean((x*y)/(np.std(x)*np.std(y)))
    


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
                 hdf5_files,
                 bin_size: tuple = (400, 400),
                 smooth: bool = True,
                 sigma: tuple = None,
                 annotation_csv: str = None,
                 exclude_genes: list = (),
                 verbose: bool = False,
                 ):
        """
        Parameters
        ----------
            hdf5_file : str or list
                either a filepath or list of filepaths to
                hdf5 files containing spot coordinates

            bin size: 2-tuple of ints or int
                y and x dimensions (in pixels) of each bin
                if int is given, both dimensions are smoothed by same sigma

            smooth:
                whether to use gaussian smoothing

            sigma: float or int
                sigma of gaussian smoothing filter in PIXELS

            annotation_csv: str
                csv file with gene annotations
                (e.g. known tissue region, num probes etc.)

        """
        if isinstance(hdf5_files, str):
            self.hdf5_file_list = [hdf5_files, ]
        elif isinstance(hdf5_files, (list, tuple)):
            self.hdf5_file_list = hdf5_files

        self.bin_size = bin_size

        self.smooth = smooth
        if smooth:
            if isinstance(sigma, tuple):
                assert len(sigma) == 2, "sigma should have only 2 values"
                self.sigma = sigma
            if isinstance(sigma, (float, int)):
                self.sigma = (sigma, sigma)

        self.annotation_csv = annotation_csv
        self.exclude_genes = exclude_genes
        self.verbose = verbose

        self.num_genes = None
        self.max_coords_list = []  # max pixel coordinates for each hdf5 file
        self.bin_edges_list = []  # list of bin-edge arrays
        self.image_shape_list = []  # list of image sizes
        self.img_arrays = []  # genes x ydim x xdim arrays
        self.mean_arrays = []  # ydim x xdim arrays or mean gene counts
        self.gene_index_dict = {}
        self.bin_edges = {}
        self.image_shape = np.zeros(len(self.bin_size),
                                    dtype=np.int32)

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

        if verbose:
            print("_" * 50 + "\nHDF5 files to parse:\n" +
                  "\n".join([f" - {file}"
                             for file in self.hdf5_file_list]) + "\n")

        self.getAllParams()

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

    def _smooth(self,
                img_array: np.ndarray,
                sigma: tuple = None,
                ):
        """
        gaussian filter an image in 2D
        or a series of 2D images in a 3D array

        parameters
        ----------
            img_array:
                either a 2D image array (ndarray) e.g. mean img
                or a 3D image array of 2D images (2nd and 3rd dim)
            sigma:
                sigma of gaussian kernel in PIXELS
        """
        assert img_array.ndim in (2, 3), (f"image array dimension is {img_array.ndim}."
                                          f"\nMust be either 2 or 3")

        # scale sigma to bin size
        if sigma is None:
            sigma_scaled = (1, 1)
        else:
            sigma_scaled = (sigma[0] / self.bin_size[0],
                            sigma[1] / self.bin_size[1])

        if img_array.ndim == 2:
            return gaussian_filter(img_array,
                                   sigma=sigma_scaled,
                                   )
        elif img_array.ndim == 3:
            return gaussian_filter(img_array,
                                   sigma=(0, sigma_scaled[0], sigma_scaled[1]),
                                   )

    def getAllParams(self,
                     # verbose=False,
                     ) -> None:

        for file_num, file in enumerate(self.hdf5_file_list):

            # get the params for the specific file
            gene_index_dict, num_genes, max_coords = self._getParams(file, )

            bin_edges, image_shape = self._getBinEdges(max_coords, )

            #
            # check that gene counts and genes match that of the first file
            # -------------------------------------------------------------
            #
            if file_num == 0:
                self.num_genes = copy.copy(num_genes)
                self.gene_index_dict = copy.copy(gene_index_dict)
            else:
                assert num_genes == self.num_genes, (
                    f"Number of genes ({num_genes}) in\n{file}\n"
                    f"do not match number of genes ({self.num_genes}) "
                    f"in\n{self.hdf5_file_list[0]}"
                )
                assert set(gene_index_dict.keys()) == set(self.gene_index_dict.keys()), (
                    f"Genes contained in\n{file} do not match "
                    f"genes contained in\n{self.hdf5_file_list[0]}"
                )

            #
            # Append relevant parameters to lists
            # -----------------------------------
            # includes the max coordinates, bin edges and shape
            # of the binned image
            #
            self.max_coords_list.append(max_coords)
            self.bin_edges_list.append(bin_edges)
            self.image_shape_list.append(image_shape)

        #
        # create a list of genes in index order
        # -------------------------------------
        #
        self.ordered_genes = [None, ] * self.num_genes
        for gene in self.gene_index_dict:
            self.ordered_genes[self.gene_index_dict[gene]] = gene
        if self.verbose:
            print(f"Ordered list of genes:\n")
            pp.pprint(self.ordered_genes)

    def _getParams(self,
                   hdf5_file,
                   # verbose=False,
                   ) -> Tuple[Dict, int, np.ndarray]:
        """
        get parameters from the first hdf5 file given
        """
        if self.verbose:
            print("-" * 60 +
                  f"\n Getting params for:\n {hdf5_file}\n")

        gene_index_dict = {}
        num_genes = 0
        max_coords = np.array((0., 0.))

        with h5py.File(hdf5_file, 'r') as f:
            for gene in f:
                if gene not in self.exclude_genes:
                    try:
                        gene_index_dict[gene] = f[gene].attrs["index"]
                    except:
                        warnings.warn("Gene-index attribute not found in hdf5 file. "
                                      "Creating arbitary index to match gene-names.")
                        gene_index_dict[gene] = num_genes

                    if self.verbose:
                        # print(f[gene][:, :2])
                        print(f" Gene {gene} has shape: {f[gene].shape}")

                    if f[gene].ndim == 2:
                        max_temp = f[gene][:, :2].max(axis=0)
                        max_coords = np.maximum(max_coords, max_temp)
                        if self.verbose:
                            print(f"{gene} max coord: {max_temp}")
                    else:
                        if self.verbose:
                            print(f"{gene} has no spots.")

                    num_genes += 1

        if self.verbose:
            print(f"Maximum coordinates = {max_coords}\n")

        max_coords = roundUp(max_coords, self.bin_size)

        if self.verbose:
            print(f"Rounded-up coordinates = {max_coords}\n"
                  f"Total number of genes = {num_genes}\n"
                  f"Gene to index dictionary:")
            pp.pprint(gene_index_dict)

        return gene_index_dict, num_genes, max_coords

    def _getBinEdges(self,
                     max_coords: np.ndarray,
                     # verbose=False,
                     ) -> Tuple[dict, np.ndarray]:
        """
        get an array of bin edges
        and image shape
        for a given hdf coordinates file
        """

        bin_edges = {}
        image_shape = np.zeros(len(self.bin_size),
                               dtype=np.int32)

        for dim, bin_size in enumerate(self.bin_size):
            bin_edges[dim] = np.arange(0, max_coords[dim] + 1,
                                       bin_size)
            image_shape[dim] = len(bin_edges[dim]) - 1

        if self.verbose:
            print(f"Bin Edges Y:\n{bin_edges[0]}\n"
                  f"Bin Edges X:\n{bin_edges[1]}\n"
                  f"Image shape: {image_shape}\n")

        return bin_edges, image_shape

    def readAllH5(self,
                  **kwargs,
                  ) -> None:
        """
        read all hdf5 files in the given file list
        saving into img_arrays and mean_arrays attributes of the object
        """
        for file_num, file in enumerate(self.hdf5_file_list):
            img_array, mean_array = self._readH5toImg(
                file,
                self.bin_edges_list[file_num],
                self.image_shape_list[file_num],
                **kwargs
            )

            # smooth images with gaussian filter
            if self.smooth:
                img_array = self._smooth(img_array, sigma=self.sigma)
                mean_array = self._smooth(mean_array, sigma=self.sigma)

            # add to lists of image arrays
            self.img_arrays.append(img_array)
            self.mean_arrays.append(mean_array)

    def _readH5toImg(self,
                     hdf5_file: str,
                     bin_edges: np.ndarray,
                     image_shape: np.ndarray,
                     check_img: list = None,  # a list of image indexes to check
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        read a hdf5 file into an array of images
        with first dimension equal to number of genes
        per-gene count maps are inserted in gene index order
        returns (per-gene array, mean array)
        """
        img_array = np.zeros(
            (self.num_genes, image_shape[0], image_shape[1],)
        )

        if self.verbose:
            print(f"Initialized image array with shape {img_array.shape}")

        with h5py.File(hdf5_file, 'r') as f:
            for gene in self.gene_index_dict.keys():
#                print(f[gene])
                if f[gene].ndim == 2:
                    gene_hist = np.histogram2d(f[gene][:, 0],
                                               f[gene][:, 1],
                                               bins=[bin_edges[0],
                                                     bin_edges[1]],
                                               )[0]
                    assert gene_hist.shape == tuple(image_shape), (
                        f"gene {gene} histogram shape {gene_hist.shape} "
                        f"does not match {image_shape}"
                    )
                    img_array[self.gene_index_dict[gene], ...] = gene_hist

        mean_array = np.mean(img_array, axis=0)

        if check_img is not None:
            for image_num in check_img:
                print(img_array[image_num, ...])
                plt.imshow(img_array[image_num, ...],
                           cmap="hot")
                plt.show()
            plt.imshow(mean_array, cmap="hot")
            plt.show()
        
        
        return img_array, mean_array

    def _readH5toImgSingle(self,
                           file_num: int,
                           gene: str,
                           ) -> np.ndarray:
        """
        read a single gene from a hdf5 file into an images
        returns count map for that gene
        """
        hdf5_file = self.hdf5_file_list[file_num]
        bin_edges = self.bin_edges_list[file_num]
        image_shape = self.image_shape_list[file_num]

        with h5py.File(hdf5_file, 'r') as f:

            # check if the gene is one of the keys of the hdf5 file
            assert gene in self.gene_index_dict.keys(), (
                f"Gene {gene} not found in coords hdf5 file"
            )

            if f[gene].ndim == 2:

                # print(np.array(f[:, 0]), np.array(f[:, 0]).shape)
                gene_hist = np.histogram2d(f[gene][:, 0],
                                           f[gene][:, 1],
                                           bins=[bin_edges[0],
                                                 bin_edges[1]],
                                           )[0]

                assert gene_hist.shape == tuple(image_shape), (
                    f"gene {gene} histogram shape {gene_hist.shape} "
                    f"does not match {image_shape}"
                )

                # smooth images with gaussian filter
                if self.smooth:
                    return self._smooth(gene_hist, sigma=self.sigma)
                else:
                    return gene_hist
            else:
                return np.zeros(tuple(image_shape))

    def plotALot(self,
                 gene_list,  # list of genes to plot
                 savedir="",
                 title="images",
                 grid=(4, 8),  # grid to plot for each figure
                 figsize=(16, 9),
                 dpi=300,
                 ):
        """
        plot a lot of intensity maps
        from a list of genes
        """
        genes_per_plot = grid[0] * grid[1]
        num_plots, remainder = divmod(len(gene_list), (genes_per_plot))
        # add an extra plot if
        # number of genes is not perfectly divisible by number of plots
        if remainder != 0:
            num_plots += 1

        for img_num in range(len(self.img_arrays)):

            # set up index for number of genes already plotted
            # ------------------------------------------------
            reordered_idx = 0

            for plot_num in range(num_plots):

                # set up figure canvas
                # --------------------
                fig = Figure(figsize=figsize, dpi=dpi)
                canvas = FigCanvas(fig)
                fig.set_canvas(canvas)

                for gridpos in range(genes_per_plot):

                    # check if we have reached end of gene list
                    # -----------------------------------------
                    if reordered_idx == len(gene_list) - 1:
                        break

                    # create temporary axes reference
                    # -------------------------------
                    ax = fig.add_subplot(grid[0], grid[1],
                                         gridpos + 1)

                    # plot the current gene
                    # ---------------------
                    array_idx = self.gene_index_dict[
                        gene_list[reordered_idx]
                    ]
                    ax.imshow(self.img_arrays[img_num][array_idx, ...],
                              cmap="hot")
                    ax.set_title(gene_list[reordered_idx])

                    # increment gene index
                    # --------------------
                    reordered_idx += 1

                fig.suptitle(title + f"\n{self.hdf5_file_list[img_num]}"
                                     f"\n({plot_num + 1} of {num_plots})")
                fig.tight_layout(rect=(0, 0, 1, .94))

                # save the plot
                # -------------
                savename = (f"image_{img_num + 1}_{title.replace(' ','_')}"
                            f"_{plot_num + 1}of{num_plots}_{self.time_str}.png")
                if not os.path.exists(savedir):
                    os.mkdir(savedir)
                fig.savefig(os.path.join(savedir, savename),
                            dpi=dpi)

                # close the canvas
                # ----------------
                canvas.close()
                fig.clear()

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
            gene1_idx = self.gene_index_dict[gene1]
            for gene2 in self.gene_index_dict:
                gene2_idx = self.gene_index_dict[gene2]

                #
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

                #
                # combine separate images
                # -----------------------
                #

                gene1_list, gene2_list = [], []
                for img in self.img_arrays:
                    gene1_list.append(img[gene1_idx, ...])
                    gene2_list.append(img[gene2_idx, ...])

                gene1_array = np.concatenate(gene1_list, axis=None)
                gene2_array = np.concatenate(gene2_list, axis=None)

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

        if savedir:
            if not os.path.exists(savedir):
                os.mkdir(savedir)
            filename = (f"clusterplot_"
                        f"{metric.replace(' ','_')}_"
                        f"{self.time_str}.png")
            g.fig.savefig(os.path.join(savedir, filename), dpi=800, )

        # list of genes in order shown on clustermap
        reordered_genes = [self.ordered_genes[idx]
                           for idx in
                           g.dendrogram_row.reordered_ind]
        if verbose:
            print(f"Reorderd indices:\n"
                  f"{g.dendrogram_row.reordered_ind}\n"
                  f"Reordered genes:\n{reordered_genes}")
        
        g.fig.clear()
        return corr_matrix, reordered_genes

def sel_genes(metric, min_corr, hdf5_files, data_path,
              bin_size = (200, 200),
              smooth = True, 
              sigma = 100,
              annotation_csv = None,
              excluded_genes = [],
              verbose=False,
              ):
    
    print(f'Initiating spatial comparison, excluding genes: {excluded_genes}')
    
    aspots = AnalyzeSpots(hdf5_files, 
                          bin_size, 
                          smooth, 
                          sigma, 
                          annotation_csv, 
                          excluded_genes, 
                          verbose)
    aspots.readAllH5()
    
    if metric not in aspots.metric_list:
        print(f'Metric not implemented!')
        
    elif metric == 'mutual information':
        matrix, reordered_genes = aspots.clusterPlot(metric = 'mutual information',
                                               self_correlation = False, 
                                               metric_kws = {'bins':50, },
                                               savedir=data_path, 
                                               verbose = False, 
                                               )
    elif metric in ['normalized crosscorr', 'pearson', 'SSIM'] :
        matrix, reordered_genes = aspots.clusterPlot(metric = metric, 
                                               self_correlation = False,
                                               savedir=data_path, 
                                               verbose = False,
                                               )
    elif metric == 'JS divergence':
        matrix, reordered_genes = aspots.clusterPlot(metric = 'JS divergence', 
                                                     self_correlation = False, 
                                                     savedir=data_path, 
                                                     verbose = False, 
                                                     col_for_cbar = 'Number of probs',
                                                     col_for_legend = 'Zone',
                                                     )
        
        images_savepath = os.path.join(data_path, f'images_js_{aspots.time_str}')
        aspots.plotALot(reordered_genes, title='JS divergence', savedir=images_savepath, )
    
    selected_indices = np.unique(np.where(matrix > min_corr))
    selected_genes = [reordered_genes[i] for i in selected_indices]
    excluded_genes = excluded_genes + list(set(reordered_genes) - set(selected_genes))
    
    print(f'\n {len(selected_genes)} genes are selected: ')
    print(selected_genes)
    print(f'\n {len(excluded_genes)} genes are excluded: {excluded_genes}')
    
    return aspots, selected_genes, excluded_genes

# -------------------------------------------------------------
#                     Script
# -------------------------------------------------------------
#

if __name__ == "__main__":

    # -------------------------------------------------------
    # Ask user for the folder of hdf files or single hdf file
    # -------------------------------------------------------
    #

    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please select directory with hdf5 files")
    # data_path = filedialog.askopenfilename(title="Please select hdf file")
    root.destroy()

    # ---------------------------------------------------------------------
    # Get .hdf5 files and annotation .csv files from folder and subfolders
    # Naming conventions:
    #   .hdf5 files begin with 'coord'
    #   annotation files begin with 'gene_annot'
    # ---------------------------------------------------------------------
    #
    for root, dirs, files in os.walk(data_path):
        print(f"Current dir: {root}")
        # print(f"List of files: {files} \n")
        hdf5_file_list = []
        annotation_csv = None
        exclude_genes_txt = None
        
        for f in files:
            if f.startswith("coord") and f.endswith(".hdf5"):
                hdf5_file_list.append(os.path.join(root, f))
            elif f.startswith("gene_annot") and f.endswith(".csv"):
                annotation_csv = os.path.join(root, f)
            elif f.startswith("exclude_genes") and f.endswith(".txt"):
                exclude_genes_txt = os.path.join(root, f)
        print(f"\n --- List of hdf5 files --- ")
        pp.pprint(hdf5_file_list)
        
        if len(hdf5_file_list) > 0:
            if exclude_genes_txt is not None:
                print(f"Import excluded_genes from file {exclude_genes_txt} ...")
                excluded_genes = [line.rstrip('\n') for line in open(exclude_genes_txt, 'r')]
            else:
                print(f"No excluded genes...")
                excluded_genes = []
            
            # Run all implemented metrics 
#            implemented_metrics = ["mutual information",
#                                   "normalized crosscorr",
#                                   "pearson",
#                                   "SSIM",
#                                   "JS divergence", 
#                                   ]
            
            # Specify one/several metrics
            implemented_metrics = ["JS divergence"]
            min_corr = 0
            for metric in implemented_metrics:
                print(f"\n----Plotting gene-gene correlation heatmap using {metric} ---------\n")
                aspots, selected_genes, excluded_genes = sel_genes(metric, min_corr,
                                                                   hdf5_file_list, root,
                                                                   bin_size = (12,12),
                                                                   smooth = True,
                                                                   sigma = 60,
                                                                   annotation_csv = annotation_csv, 
                                                                   excluded_genes = excluded_genes,
                                                                   verbose = True,
                                                                   )
                