# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 21:40:08 2020

@author: zhouxr Jan 2020
"""
import os
import pickle
import datetime

import tkinter as tk
from tkinter import filedialog
from _utils import parse_folder, read_h5py, get_abundant_genes
from smoothing import SmoothImgs, plotALot
from kmeans import Kmeans_clustering

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
            read_h5py(hdf5_file, verbose = True, genes_to_exclude = genes_to_exclude)
            
            smoothdir = os.path.join(os.path.dirname(hdf5_file),
                                     f"results_remove_about_blank_{time_str}")
            if not os.path.exists(smoothdir):
                os.makedirs(smoothdir)
            
            # Update gene_index_dict, remove blank genes 
            # and less abundant genes
            # Comment out to use all genes
            less_abundant_genes_file = os.path.join(smoothdir, 
                                                    'genes_to_exclude.txt')
            excluded_genes = get_abundant_genes(gene_index_dict, 
                                                blank_genes,
                                                less_abundant_genes_file)
            genes_to_exclude = blank_genes + excluded_genes
            gene_index_dict, raw_max_coords, num_genes, _ = \
            read_h5py(hdf5_file, verbose = True, 
                      genes_to_exclude = genes_to_exclude)
            
            print(f"After removing blank genes and less abundant genes"
                  f"\t {len(gene_index_dict)}")
            # -----------------------------------------------------------------
            
            # Smoothing ----------------------------------------------------
            # parameters
            bin_size = 12
            sigma = 60
            
            smooth_imgs = SmoothImgs(gene_index_dict,
                                     raw_max_coords,
                                     num_genes, 
                                     smoothdir,
                                     bin_size= bin_size, 
                                     sigma = sigma,
                                     normalize = "zscore",
                                     verbose=True)
            
            path_to_imgs = os.path.join(smooth_imgs.smoothdir, "imgs_per_gene")
            if not os.path.exists(path_to_imgs):
                os.makedirs(path_to_imgs)
            
            # Smooth and plot images (check sigma)
            plotALot(img_array = smooth_imgs.smoothed_img_array,
                     gene_index_dict = smooth_imgs.gene_index_dict,
                     savedir = path_to_imgs, 
                     title = f'smoothed images (binsize = {bin_size} sigma= {sigma})')
            

            # KMeans clustering ------------------------------------
            print("----- Kmeans clustering -------")
            path_to_clustering = os.path.join(smooth_imgs.smoothdir,
                                              "clustering_results")
            if not os.path.exists(path_to_clustering):
                os.makedirs(path_to_clustering)
            
            # Initializing for kmeans clustering
            clustering = Kmeans_clustering(smooth_imgs)
            
            # Elbow plot
#            # Parameters for checking 'nclusters'
#            min_ncluster = 5
#            max_ncluster = 15
#            step = 2
#            clustering.plot_elbow(path_to_clustering, 
#                                  list(range(min_ncluster,
#                                             max_ncluster, 
#                                             step)))
            
            # kmeans clustering for a specific 'nclusters'
            n_clusters = 10
            print(f"----- Kmeans clustering: {n_clusters} clusters")
            clustering.kmeans(n_clusters= n_clusters, training_portion = 0.1)
            
            print("----- Plotting silhouette coefficient")
            clustering.silhouette_coefficient(path_to_clustering, "kmeans")
            
            # End of Kmeans clustering -------------------------------------------