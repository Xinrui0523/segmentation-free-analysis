# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 17:27:15 2020

@author: zhouxr
"""
import os
import gc
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
from matplotlib.legend import Legend
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.spatial.distance import cosine
from spherecluster import SphericalKMeans
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import seaborn as sns; 

import datetime
import tkinter as tk
from tkinter import filedialog
from smoothing import SmoothImgs, plotALot
from _utils import parse_folder, read_h5py, get_abundant_genes
import warnings

warnings.simplefilter(action='ignore', category = FutureWarning)

sns.set()
colorlist = ('#000000', '#ff851b', '#ff4136', '#0074d9',
             '#85144b', '#0C5C17', '#ffdc00', '#4EEBEB',
             '#f012be', '#FFF5EE', '#01ff70', '#4c72b0',
             '#55a868', '#8172b2', '#64b5cd', '#9a6200',
             '#7a9703', '#ad0afd', '#516572', '#ffc5cb')

implemented_models = ['kmeans', 'sphericalKMeans', 'spectral']
             
class DatasetClustering(object):
    
    def __init__(self,
                 smooth_imgs: SmoothImgs,
                 train_size = 0.05,
                 sampling_method = "random",
                 random_state = None,
                 vebose = False,
                 ):
        '''
        sampling_method:
            "random": random sampling 
            "equal": sampling with equal distance
            default: random
        '''
        self.image_shape = smooth_imgs.image_shape
        
        self.gene_index_dict = smooth_imgs.gene_index_dict
        self.genes = list(smooth_imgs.gene_index_dict.keys())
        
        self.img_array = smooth_imgs.smoothed_img_array
        
        self.train_size = train_size
        self.sampling_method = sampling_method
        self.random_state = random_state
        
#        self.data_pixel = None
#        self.gene_pixel = None
#        self.train_idx = None
#        self.train_data = None
#        self.x_tr = None
#        
#        self.silhouette_vals = None
        
        
    def get_train_data(self):
        x, y = np.mgrid[0: self.image_shape[1], 
                        0: self.image_shape[0]]
        impt = np.vstack([x.ravel(), y.ravel()]).T
        self.data_pixel = pd.DataFrame(impt, columns=["x", "y"])
        for gene in self.gene_index_dict:
            self.data_pixel[gene] = self.img_array[self.gene_index_dict[gene]["index"],
                            impt[:,1], impt[:,0]]
        
        self.gene_pixel = self.data_pixel[self.genes].T
        
        if self.sampling_method == "equal":
            # Sampling with equal distance
            self.train_idx = np.arange(0, len(self.data_pixel), int(1/self.train_size))
            self.train_data = self.data_pixel.iloc[self.train_idx][self.genes]
        else:
            # Random sampling
            print(f"Random sampling {self.train_size} of data...")
            self.train_data, _ = train_test_split(self.data_pixel,
                                                  train_size = self.train_size,
                                                  random_state = self.random_state)
            self.train_idx = self.train_data.index
            
        self.x_tr = self.train_data[self.genes]


    # Clustering methods ----------------------------
    def trainModel(self,
                   model_name = "kmeans",
                   n_clusters = 10,
                   **kwargs):
        assert model_name in implemented_models
        self.n_clusters = n_cluster
        
        if model_name == 'kmeans':
            model = KMeans(n_clusters = n_clusters,
                           **kwargs)
        elif model_name == 'sphericalKMeans':
            model = SphericalKMeans(n_clusters = n_clusters,
                                    **kwargs)
        elif model_name == 'spectral':
            model = SpectralClustering(n_clusters = n_clusters,
                                       **kwargs)
        
        print(f"Training model {model_name} using {self.train_size} samples...")
        model.fit(self.x_tr)
        self.y_tr = model.labels_
        
        print(f"Predicting labels ...")
        clf = LinearSVC().fit(self.x_tr, self.y_tr)
        self.data_pixel["cluster"] = clf.predict(self.data_pixel[self.genes])
        
        self.model = model
    
    def plot_clusters(self, 
                      savedir,
                      savename,
                      dpi = 300):
        fig = plt.figure()
        ax = fig.gca()
        ax.grid(False)
        cmaps = mcl.LinearSegmentedColormap.from_list('mylist',
                                                      colorlist[:self.n_clusters],
                                                      self.n_clusters)
            
        image = np.zeros(self.image_shape)
        image[self.data_pixel.y, self.data_pixel.x] = self.data_pixel["cluster"]
        im = ax.imshow(image, cmap = cmaps)
        
        ax.set_title(f'clustering map - {savename}', y=1.02)
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=0.2, pad=0.2)
        cbar = fig.colorbar(im, cax=cax, orientation="vertical")
        cbar.set_ticks([])
        for c, label in enumerate(np.arange(1, self.n_clusters+1)):
            cbar.ax.text(15, c, label, ha='center', va='center')
        
        figname = os.path.join(savedir, savename)
        plt.savefig(figname, dpi=dpi)
        print(f'Clustering map saving as {figname}')
        plt.close()
        
        
    def plot_silhouette_coefficient(self, 
                                    savedir, savename, 
                                    metric = 'euclidean',
                                    plot_bg = False, dpi=300):
        self.silhouette_vals = silhouette_samples(self.x_tr, 
                                                  self.y_tr, 
                                                  metric = metric)
        
        bars = []
        showing_labels = []
        
        fig = plt.figure()
        ax = fig.gca()
        ax.grid(False)
        
        y_ticks = []
        y_lower, y_upper = 0, 0
        for i, (cluster, color) in enumerate(zip(np.unique(self.data_pixel['cluster']),
                                             colorlist[:self.n_clusters])):
            if plot_bg:
                cluster_samples = self.data_pixel.iloc[self.train_idx].cluster
                cluster_silhouette_vals = self.silhouette_vals[cluster_samples == cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                
                showing_labels.append(cluster+1)
                bar = ax.barh(range(y_lower, y_upper), 
                              width = cluster_silhouette_vals, 
                              color = color,
                              linewidth = 0)
                bars.append(bar)
                ax.text(-0.03, (y_lower + y_upper)/2, str(cluster+1))
                            
                y_lower += len(cluster_silhouette_vals)
            else:
                if cluster!=0:
                    cluster_samples = self.data_pixel.iloc[self.train_idx].cluster
                    cluster_silhouette_vals = self.silhouette_vals[cluster_samples == cluster]
                    cluster_silhouette_vals.sort()
                    y_upper += len(cluster_silhouette_vals)
                    
                    showing_labels.append(cluster+1)
                    bar = ax.barh(range(y_lower, y_upper), 
                                  cluster_silhouette_vals, 
                                  color = color, 
                                  linewidth = 0)
                    # bar[0].set_color(color)
                    bars.append(bar)
                    ax.text(-0.03, (y_lower + y_upper)/2, str(cluster+1))
                                
                    y_lower += len(cluster_silhouette_vals)
                
                
        ax.legend(bars, showing_labels, loc='upper left')
        # Get the average silhouette score 
        avg_score = np.mean(self.silhouette_vals)
        avgline = ax.axvline(avg_score, linestyle='--', linewidth = 0.5, 
                             color='green')
        leg1 = Legend(ax, [avgline], [f'avg_score = {round(avg_score,2)}'], 
                      loc = 'lower right')
        ax.add_artist(leg1)
        
        ax.set_yticks(y_ticks)
        ax.set_xlabel('Silhouette coefficient score')
        ax.set_ylabel('Cluster labels')
        ax.set_title('Silhouette plot for clusters', y=1.02)   
        
        figname = os.path.join(savedir, savename)
        
        plt.tight_layout()
        plt.savefig(figname, dpi=dpi)
        print(f'plot silhouette coefficient saving as {figname}')
        plt.close()
        
if __name__ == '__main__':
    
    script_time = datetime.datetime.now()
    time_str = script_time.strftime("%Y%m%d_%H%M")
    
    # -------------------------------------------------------
    # Ask user for the folder with datasets
    # -------------------------------------------------------
    root = tk.Tk()
    root.withdraw()
    data_path = filedialog.askdirectory(title="Please select directory with hdf5 files")
    root.destroy()
    
    # --------------------------
    # Parameters
    norm_method = "no_norm"
    bin_size = 12
    sigma = 60
    
    n_cluster = 10
    train_size= 0.05
    # --------------------------
    
    for root, dirs, files in os.walk(data_path):
        print(f"Current dir: {root}")
        hdf5_file_list, annotation_csv, genes_to_exclude = parse_folder(root)
        
        # --- Analysing each coord*.hdf5 file in current dir ---
        for hdf5_file in hdf5_file_list:
            gene_index_dict, raw_max_coords, num_genes, blank_genes = \
            read_h5py(hdf5_file, verbose = False, 
                      genes_to_exclude = genes_to_exclude)
            
            smoothdir = os.path.join(os.path.dirname(hdf5_file),
                                     f"kmeans_{norm_method}_cosine_{time_str}")
            if not os.path.exists(smoothdir):
                os.makedirs(smoothdir)
                
#            # Update gene_index_dict, remove blank genes 
#            # and less abundant genes
#            # Comment out to use all genes
#            less_abundant_genes_file = os.path.join(smoothdir, 
#                                                    'genes_to_exclude.txt')
#            excluded_genes = get_abundant_genes(gene_index_dict, 
#                                                blank_genes,
#                                                less_abundant_genes_file)
#            genes_to_exclude = blank_genes + excluded_genes
#            gene_index_dict, raw_max_coords, num_genes, _ = \
#            read_h5py(hdf5_file, verbose = True, 
#                      genes_to_exclude = genes_to_exclude)
#            
#            print(f"After removing blank genes and less abundant genes"
#                  f"\t {len(gene_index_dict)}")
            # -----------------------------------------------------------------
            
            # Smoothing ----------------------------------------------------                        
            smooth_imgs = SmoothImgs(gene_index_dict,
                                     raw_max_coords,
                                     num_genes, 
                                     smoothdir,
                                     bin_size= bin_size, 
                                     sigma = sigma,
                                     norm_method = norm_method,
                                     verbose=False)
#            
#            # Plot smoothed images --------------------------------------------------
#            path_to_imgs = os.path.join(smooth_imgs.smoothdir, "imgs_per_gene")
#            if not os.path.exists(path_to_imgs):
#                os.makedirs(path_to_imgs)
#            
#            # Smooth and plot images (check sigma)
#            plotALot(img_array = smooth_imgs.smoothed_img_array,
#                     gene_index_dict = smooth_imgs.gene_index_dict,
#                     savedir = path_to_imgs, 
#                     title = f'smoothed images (binsize = {bin_size} sigma= {sigma})')
            
            
            # clustering ------------------------------------
            # set saving directory for clustering results
            path_to_clustering = os.path.join(smooth_imgs.smoothdir,
                                              "clustering_results")
            if not os.path.exists(path_to_clustering):
                os.makedirs(path_to_clustering)
            
            # Initializing dataset clustering
            print(f"Initializing dataset for clustering ...")
            clustering = DatasetClustering(smooth_imgs, 
                                           train_size = 0.3, 
                                           sampling_method = 'equal',
                                           random_state = None)
            clustering.get_train_data()
            
            model_name = "kmeans"
            clustering.trainModel(model_name)
            clustering.plot_clusters(path_to_clustering, model_name)
            clustering.plot_silhouette_coefficient(path_to_clustering, 
                                                   f'sihouette_{model_name}')
