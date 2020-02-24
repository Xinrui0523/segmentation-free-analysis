# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 23:03:12 2020

@author: zhouxr
"""
import os

import tkinter as tk
import pprint as pp

from tkinter import filedialog
from NonSegmentedClasses import read_h5py
from Kmeans_modularize import Kmeans_clustering
from spatialComparison_Nigel import roundUp

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
        # Parsing files in this folder --------------------------
        print(f"Current dir: {root}")
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
        print()
        # -------------------------------------------------------
        
        if len(hdf5_file_list) > 0:
            if exclude_genes_txt is not None:
                print(f"Import excluded_genes from file {exclude_genes_txt} ...")
                excluded_genes = [line.rstrip('\n') for line in open(exclude_genes_txt, 'r')]
            else:
                print(f"No excluded genes...")
                excluded_genes = []
            
            for hdf5_file in hdf5_file_list:
                # k-means clustering for each coords file
                pos_dic, height, width = read_h5py(hdf5_file,
                                                   verbose = False,
                                                   exclude_genes = excluded_genes)
                height = roundUp(height, 10)
                width = roundUp(width, 10)
                clustering = Kmeans_clustering(hdf5_file, 
                                               height, 
                                               width, 
                                               downsample=10)
                clustering.read_h5py(excluded_genes)
                clustering.gaussian_filter(sigma=5)
                clustering.kmeans(n_clusters=10)
                clustering.plot_kmeans_result(savepath=root)
                clustering.hierarchical_clusting(m_clusters=7, form='average')
                clustering.plot_heatmap(root,1,1)
            