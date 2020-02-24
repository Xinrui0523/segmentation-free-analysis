# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:26 2020

Functions for binning and smoothing coordinates.

@author: zhouxr Jan 2020
"""

from _utils import roundUp

import os
import datetime
import warnings
import numpy as np
import tkinter as tk
from tkinter import filedialog
from _utils import parse_folder, read_h5py
from typing import Tuple
from scipy.ndimage import gaussian_filter
from scipy.stats import zscore

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas


def plotALot(img_array,
             gene_index_dict,  # list of genes to plot
             reordered_genes = None,
             savedir="",
             title="images",
             grid=(3, 6),  # grid to plot for each figure
             figsize=(16, 9),
             dpi=300,
             ):
    """
    plot a lot of images from a list of genes
    """
    genes_per_plot = grid[0] * grid[1]
    num_plots, remainder = divmod(len(gene_index_dict), (genes_per_plot))
    # add an extra plot if
    # number of genes is not perfectly divisible by number of plots
    if remainder != 0:
        num_plots += 1
        
    if reordered_genes is None:
        reordered_genes = [None, ] * len(gene_index_dict)
        for gene in gene_index_dict:
            reordered_genes[gene_index_dict[gene]["index"]] = gene
        
    # set up index for number of genes already plotted
    # ------------------------------------------------
    array_idx = 0        
    for plot_num in range(num_plots):
        # set up figure canvas
        # --------------------
        fig = Figure(figsize=figsize, dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)
        
        for gridpos in range(genes_per_plot):
            # check if we have reached end of gene list
            # -----------------------------------------
            if array_idx == len(gene_index_dict):
                break
                
            # create temporary axes reference
            # -------------------------------
            ax = fig.add_subplot(grid[0], grid[1], gridpos + 1)
            
            # plot the current gene (array_idx)
            # ---------------------
            gene = reordered_genes[array_idx]
            ax.imshow(img_array[gene_index_dict[gene]["index"], ...], cmap="hot")
            ax.set_title(gene)
            ax.grid(False)
            
            # increment gene index
            # --------------------
            array_idx += 1
        fig.suptitle(title + f" ({plot_num + 1} of {num_plots})")
        fig.tight_layout(rect=(0, 0, 1, .94))
        
        # save the plot
#        time_now = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        savename = (f"{title.replace(' ','_')}"
                    f"_{plot_num + 1}of{num_plots}.png")
        
        if not os.path.exists(savedir):
            os.mkdir(savedir)
        fig.savefig(os.path.join(savedir, savename),dpi=dpi)
            
        canvas.close()
        fig.clear()

class SmoothImgs(object):
    
    def __init__(self,
                 gene_index_dict,
                 raw_max_coords,
                 num_genes,
                 smoothdir,
                 bin_size: tuple = (10, 10),
                 sigma: tuple = None,
                 norm_method: str = None,
                 annotation_csv: str = None,
                 genes_to_exclude: list = (),
                 verbose: bool = False,
                 ):
        """
        Parameters
        ----------
            gene_index_dict
            raw_max_coords
            num_genes
                obtained from reading the .hdf5 file
            
            smoothdir: string
                specify the folder for saving smoothed images
            
            bin_size: 2-tuple of ints or int
                y and x dimensions (in pixels) of each bin
            
            sigma: float or int
                sigma of gaussian smoothing filter in PIXELS
            
            density: bool
                False: smooth the bin counts
                True: smooth the probability density
                ---------
                Not working well. Default False
            
            normalize: string
                min_max: normalize the bin counts to min and max
                    x = (x-min)/(max-min)
                zscore: normalize to zscore 
                    x = (zscore(x)+1)/2
                density: normalize to bin counts density
                    x = x/sum(x) 
            
            annotation_csv: str
                csv file with gene annotations (e.g. known zones, # probes, ...)
            
            genes_to_exclude: list
                a list of genes to exclude
                
            verbose: bool
                if True, print details during processing
        """
        self.gene_index_dict = gene_index_dict
        self.raw_max_coords = raw_max_coords
        self.num_genes = num_genes
                
        if isinstance(bin_size, tuple):
            assert len(bin_size) == 2, "bin_size should be 2-tuple of ints or int"
            self.bin_size = bin_size
        elif isinstance(bin_size, int):
            self.bin_size = (bin_size, bin_size)
        else:
            warnings.warn("bin_size should be 2-tuple of ints or int")
                
        if isinstance(sigma, tuple):
            assert len(sigma) == 2, "sigma should have only 2 values"
            self.sigma = sigma
        elif isinstance(sigma, (float, int)):
            self.sigma = (sigma, sigma)
        elif sigma is None:
            self.sigma = (1, 1)
        
        self.norm_method = norm_method
        
        self.annotation_csv = annotation_csv
        self.genes_to_exclude = genes_to_exclude
        self.verbose = verbose
        
        self.smoothdir = smoothdir
        
        self._getAllPrams()
        self.smooth()
    
    def _getAllPrams(self):
        self.max_coords = roundUp(self.raw_max_coords, self.bin_size)
        self._getBinEdges()
        
        self._readH5toImg()
    
    def _getBinEdges(self, 
                     ) -> Tuple[dict, np.ndarray]:
        """
        Get an array of bin edges and image shape 
        of a given coordinates file
        """
        
        bin_edges = {}
        image_shape = np.zeros(len(self.bin_size), 
                               dtype=np.int32)
        
        for dim, bin_size in enumerate(self.bin_size):
            bin_edges[dim] = np.arange(0, self.max_coords[dim]+1, bin_size)
            image_shape[dim] = len(bin_edges[dim])-1
        
#        if self.verbose:
#            print(f"Bin edges Y: {bin_edges[0]} \n"
#                                  f"Bin edges X: {bin_edges[1]} \n"
#                                  f"Image shape: {image_shape} \n")
        
        self.bin_edges = bin_edges
        
        # image_shape = [y, x]
        self.image_shape = image_shape
        
#        return bin_edges, image_shape
        
        # --------- End of _getBinEdges() --------
    
    def _readH5toImg(self, 
                    check_img: list = [], # a list of image indexes to check
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a .hdf5 file to an array of images
        with the first dimension equal to the number of genes
        per-gene count maps are inserted in gene index order
        
        Parameters:
        -----------
            check_img: a list of image indexes to check
                plotting the image array for the listed genes
        
        Return: 
        -----------
            (per-gene array, mean_array)
        """
        
        
        
        # image_shape = [y, x]
        # img_array = [gene, y, x]
        img_array = np.zeros(
                (self.num_genes, self.image_shape[0], self.image_shape[1],)
                )
        if self.verbose:
            print(f"Initialize image array with shape {img_array.shape}")
        
        for gene in self.gene_index_dict.keys():
            
            gene_hist, _, _ = np.histogram2d(self.gene_index_dict[gene]["Array"][:,0],
                                             self.gene_index_dict[gene]["Array"][:,1],
                                             bins = [self.bin_edges[0],self.bin_edges[1]],
                                             )
            if self.norm_method == None:
                pass
            elif self.norm_method == "zscore":
                # gene_hist = (zscore(gene_hist)+1)/2
                zscore_ghist = (gene_hist-np.mean(gene_hist))/np.std(gene_hist)
                gene_hist = (zscore_ghist+1)/2
            elif self.norm_method == "density":
                gene_hist = gene_hist/(sum(sum(gene_hist)))
            elif self.norm_method == "min_max":
                gene_hist = (gene_hist - np.min(gene_hist))/(np.max(gene_hist)-np.min(gene_hist))
            else:
                warnings.warn(f"Normalization method NOT implemented!\n"
                              f"Smoothing without normalization ...")
            
            assert gene_hist.shape == tuple(self.image_shape), (
                    f"gene {gene} histogram shape {gene_hist.shape}"
                    f"does not match {self.image_shape}"
                    )
            img_array[self.gene_index_dict[gene]["index"], ...] = gene_hist
            
            if self.gene_index_dict[gene] in check_img:
                savedir = os.path.join(self.smoothdir, 'readH5toImg')
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                    
                print(f"Checking image {self.gene_index_dict[gene]} - {gene}...")
                plt.imshow(img_array[self.gene_index_dict[gene]["index"], ...],
                           cmap = "hot")
                plt.title(f"image {self.gene_index_dict[gene]}-{gene}")
                plt.savefig(os.path.join(savedir, 
                                         f"img_{gene}_"
                                         f"{self.gene_index_dict[gene]}"
                                         f".png"), dpi=300)
                plt.close()
    
        mean_array = np.mean(img_array, axis = 0)
        
#        # visualize mean_array ----------
#        plt.imshow(mean_array, cmap="hot")
#        plt.title(f"mean array image bin_size: {self.bin_size}")
#        plt.savefig(os.path.join(savedir,
#                                 f"mean_array_image.png"), dpi=300)
#        plt.close()
#        # --------------------------------
        
        self.img_array = img_array
        self.mean_array = mean_array
        
    
    def smooth(self,
               ):
        """
        Guassian filter an image in 2D 
        or a series of 2D images in a 3D array
        
        Parameters
        ----------
            sigma:
                standard deviation of gaussian kernel in pixels
        """
        
#        # Update self.sigma if no consistent with the initiating
#        if isinstance(sigma, tuple):
#            assert len(sigma) == 2, "sigma should have only 2 values"
#            self.sigma = sigma
#        elif isinstance(sigma, (float, int)):
#            self.sigma = (sigma, sigma)
            
        assert self.img_array.ndim in (2, 3), (
                f"image array dimension is {self.img_array.ndim}."
                f"\nMust be either 2 or 3")
        # scale sigma to bin size
        if self.sigma is None:
            sigma_scaled = (1, 1)
        else:
            sigma_scaled = (self.sigma[0]/self.bin_size[0],
                            self.sigma[1]/self.bin_size[1])
        
        smoothed_mean_array = gaussian_filter(self.mean_array, 
                                              sigma = sigma_scaled)
        if self.img_array.ndim == 2:
            smoothed_img_array = gaussian_filter(self.img_array,
                                                 sigma = sigma_scaled)
            
        elif self.img_array.ndim == 3:
            smoothed_img_array = gaussian_filter(self.img_array,
                                                 sigma = (0, sigma_scaled[0],
                                                          sigma_scaled[1]))
        self.smoothed_img_array = smoothed_img_array
        self.smoothed_mean_array = smoothed_mean_array
        
#        # ------------------------
#        # Plot smoothed mean array
#        # ------------------------
#        savedir = os.path.join(self.smoothdir, 'readH5toImg')
#        if not os.path.exists(savedir):
#            os.makedirs(savedir)
#        plt.imshow(smoothed_mean_array, cmap="hot")
#        plt.title(f"mean array image bin_size: {self.bin_size} sigma_scaled: {sigma_scaled[0]}")
#        plt.savefig(os.path.join(savedir,
#                                 f"mean_array_smoothed_image_{sigma_scaled[0]}.png"), dpi=300)
#        plt.close()
        
    # -------- End of _smooth() ------------
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
    
    
    for root, dirs, files in os.walk(data_path):
        print(f"Current dir: {root}")
        hdf5_file_list, annotation_csv, genes_to_exclude = parse_folder(root)
        
        # --- Analysing each coord*.hdf5 file in current dir ---
        for hdf5_file in hdf5_file_list:
            gene_index_dict, raw_max_coords, num_genes, blank_genes = \
            read_h5py(hdf5_file, verbose = True, 
                      genes_to_exclude = genes_to_exclude)
            
            
            spatialcompdir = os.path.join(os.path.dirname(hdf5_file),
                                          f"spatialComp_{time_str}")
            if not os.path.exists(spatialcompdir):
                os.makedirs(spatialcompdir)
            
            
            # ---------------------------
            # Smoothing -----------------
            # ---------------------------
            # parameters
            bin_size = 60
            sigma = 60
            norm_method = ""
            
            smooth_imgs = SmoothImgs(gene_index_dict,
                                     raw_max_coords,
                                     num_genes, 
                                     spatialcompdir,
                                     bin_size= bin_size,
                                     norm_method= norm_method,
                                     sigma = sigma,
                                     verbose=False)
            path_to_imgs = os.path.join(spatialcompdir, "imgs_per_gene")
            if not os.path.exists(path_to_imgs):
                os.makedirs(path_to_imgs)
            
            # Smooth and plot images (check sigma)
            plotALot(img_array = smooth_imgs.smoothed_img_array,
                     gene_index_dict = smooth_imgs.gene_index_dict,
                     savedir = path_to_imgs, 
                     title = f'smoothed {norm_method} (binsize = {bin_size} sigma= {sigma})')


