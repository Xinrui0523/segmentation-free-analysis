# -*- coding: utf-8 -*-
"""
Functions and Classes of kmeans clustering for pixels
on smoothed images. 

@author: linl Dec 2019
Updated: Xinrui Jan 2020
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
from matplotlib.legend import Legend
import os
import gc
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.gridspec as gridspec
import datetime

import tkinter as tk
from tkinter import filedialog
from _utils import parse_folder, read_h5py, get_abundant_genes
from smoothing import SmoothImgs, plotALot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from mpl_toolkits.axes_grid1 import make_axes_locatable

#------------------------------------------------------------------------------
colorlist = ('#000000', '#ff851b', '#ff4136', '#0074d9',
             '#85144b', '#0C5C17', '#ffdc00', '#4EEBEB',
             '#f012be', '#FFF5EE', '#01ff70', '#4c72b0',
             '#55a868', '#8172b2', '#64b5cd', '#9a6200',
             '#7a9703', '#ad0afd', '#516572', '#ffc5cb')

#------------------------------------------------------------------------------
  
def get_clusters(linked, dist=1.5):
    #function to get different clusters based on distance cutoff
    
    get_max = len(linked)+1
    dictt = {}
    for i in range(get_max):
        dictt[i] = [i,[i,]]
    counter = get_max 
    for i in linked:
        dictt[int(counter)] = [i[0:2],dictt[int(i[0])][1]+dictt[int(i[1])][1]]
        counter += 1
    n= len(linked[:,-2][linked[:,-2] > dist]) + 1 #num of clusters
    thres = 2*(get_max-1)-n+1
    clusters = []
    for i in range(1,n):
        linked[-i][:2] <= thres
        clusters += [linked[-i][:2][linked[-i][:2] <= thres],]
    return [dictt[i][1] for i in np.concatenate(clusters)]



#==============================================================================
#                                MAIN CLASS    
#==============================================================================
class Kmeans_clustering(object):
    
    def __init__(self,
                 smooth_imgs: SmoothImgs,
                 verbose = False,
                 ):
       
        self.image_shape = smooth_imgs.image_shape
        self.gene_index_dict = smooth_imgs.gene_index_dict
       
        self.verbose = verbose
        
        self.genes = list(smooth_imgs.gene_index_dict.keys())
        
        genes_sorted = [None, ] * len(self.gene_index_dict)
        for gene in self.gene_index_dict:
            genes_sorted[self.gene_index_dict[gene]["index"]] = gene        
        self.genes_sorted = genes_sorted
        
        
        self.img_array = smooth_imgs.smoothed_img_array
        
        self.n_clusters = None
        self.corr_maxtix = []
               
#------------------------------------------------------------------------------
    def _pca(self):
        
        C = np.cov(self.data_pixel[self.genes].T) #pca on covariance
        a,b,c = np.linalg.svd(C) #pca on covariance matrix
        pc = np.matmul(self.data_pixel[self.genes].values,a[:,:3])
        self.data_pixel['pc1'] = pc[:,0]

        max_min = np.percentile(self.data_pixel['pc1'],99)-np.percentile(self.data_pixel['pc1'],1)
        self.data_pixel['pc1'] = (self.data_pixel['pc1']-np.percentile(self.data_pixel['pc1'],1))/max_min
        self.data_pixel['pc1'] = np.clip(self.data_pixel['pc1'],0,1)
     
#------------------------------------------------------------------------------
    def _std_data(self):
        # FIXME: no meaning for standarlizing pixels ??
        print('Normalizing data ...')
        self.data_std = StandardScaler().fit_transform(self.data_pixel)
        
#------------------------------------------------------------------------------
        
    def kmeans(self, 
               n_clusters,
               training_portion = 0.03,
               init = 'k-means++',
               n_init = 10,
               max_iter = 300,
               tol = 0.0001,
               precompute_distance = 'auto',
               random_state = None,
               copy_x = True,
               n_jobs = 6,
               algorithm = 'auto',
               ):
        """
        Parameters:
        ------------
            n_clusters: int
                The number of clusters to form 
                i.e. the number of centroids to generate
            init: {'k-means++', 'random'} 
                or ndarray of shape (n_clusters, n_features)
                Method for initialization, defaults to 'k-means++'
            n_init: int, default=10
                Number of the time k-means algorithm to be run 
                with different centroid seeds. 
                The final results will be the best output 
                of n_init consecutive runs in terms of inertia
            max_iter: int, default=300
                maximum number of iterations of the k-means algorithm 
                for a single run
            tol: float, default = 1e-4
                relative tolerance with regards to inertia 
                to declare convergence
            precompute_distances: 'auto' or bool, default='auto'
                Precompute distance (faster but takes more memory)
                'auto': 
                    do not precompute distance 
                    if n_samples * n_clusters > 12 million.
                True: always precompute distances
                False: never precompute distances
            random_state: int, RandomState instance, default=None
                Use an int to make the randomness deterministic
            n_jobs: int, the number of thread to use
        """
        self.n_clusters = n_clusters
        # Check dimensions/axes
        x, y = np.mgrid[0: self.image_shape[1], 
                        0: self.image_shape[0]]
        impt = np.vstack([x.ravel(),y.ravel()]).T
        self.data_pixel = pd.DataFrame(impt,columns=['x','y'])
        
        for gene in self.gene_index_dict:
            self.data_pixel[gene] = self.img_array[self.gene_index_dict[gene]['index'], 
                            impt[:,1], impt[:,0]]
            
        km = KMeans(self.n_clusters,
                    n_jobs=n_jobs,
                    random_state=random_state)
        
        # data_train: train kmeans model with 5% of data due to memory issues
        train_idx = int(1/training_portion)
        self.data_train = self.data_pixel.iloc[::train_idx][self.genes]
        
        print('--- kmeans training ---')
        km.fit(self.data_train[self.genes])
                
        # Predict labels for all data
        print('--- predicting labels ---')
        self.data_pixel['pred'] = km.predict(self.data_pixel[self.genes])
        
        # Get the correlation of each cluster
        print('--- Getting correlation of each cluster ---')
        self._pca() 
        
        temp = self.data_pixel[['pred','pc1']].groupby('pred').apply(np.mean).reset_index(drop=1)
        temp.rename(columns={'pc1':'pc1_mean'},inplace=True)
        self.data_pixel = pd.merge(self.data_pixel,temp,how='left',on='pred')
        
        uq =  np.unique(self.data_pixel.pc1_mean)
        self.data_pixel['cluster']=np.zeros((len(self.data_pixel['pred']==0),1))
        for i in range(len(uq)):
           self.data_pixel.loc[self.data_pixel['pc1_mean']==uq[i],'cluster'] = (i+1)%self.n_clusters
        
        print('--- Calculating silhouette scores ---')
        self.data_train_pred_labels = self.data_pixel.iloc[::train_idx]['cluster']
        self.silhouette_vals = silhouette_samples(self.data_train, 
                                                  self.data_train_pred_labels)
        
        self.centroids = km.cluster_centers_  
        self.km_inertia_ = km.inertia_
        self.km = km

#------------------------------------------------------------------------------
    # FIXME: GENE-GENE correlation can use Nigel's code (with different metrics)
    def _corr(self):
        self.corr_maxtix = np.corrcoef(self.data_pixel[self.genes].T)
#------------------------------------------------------------------------------        
        
    def hierarchical_clusting(self, savepath, savename, m_clusters=7, form='average', dpi=300):
        #hierarchical clusting
        # FIXME: same as Nigel's spatial comparison??
        assert m_clusters>1 , 'error: cluster number should be larger than 1'
        self._corr()    
        
        linked = linkage(self.corr_maxtix, form)
        dist= (linked[-(m_clusters-1)]+linked[-(m_clusters)])[2]/2
        plt.figure()
        dendro  = dendrogram(linked,
                             orientation='top',
                             labels=list(self.genes),
                             distance_sort='descending',leaf_font_size=4,
                             show_leaf_counts=True,color_threshold=dist)
        figname = os.path.join(savepath, savename)
        plt.savefig(figname, dpi=dpi)
    
        self.genes_sorted = sorted(get_clusters(linked,dist=dist),key=lambda x : np.argwhere(np.array(dendro['leaves'])==x[0]))
        keys_sorted_name = list(map(lambda x : list(map(lambda x : list(self.genes)[x],x)),self.genes_sorted))
        self.genes_sorted = np.concatenate(self.genes_sorted)
        self.genes_sorted = [self.genes[x] for x in self.genes_sorted] 
#------------------------------------------------------------------------------   
           
    def plot_kmeans_result(self, savepath, savename, dpi=300):
        
        cmaps = mcl.LinearSegmentedColormap.from_list('mylist',
                                                      colorlist[:self.n_clusters],
                                                      self.n_clusters)   

        image = np.zeros(self.image_shape)
        image[self.data_pixel.y,self.data_pixel.x] = self.data_pixel.cluster
        plt.imshow(image,cmap=cmaps)
        plt.title(savename,fontsize=12,fontweight='bold')
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(self.n_clusters+1))
        cbar.set_ticklabels(np.arange(self.n_clusters+1))
#        plt.savefig(savepath + '_kmeans_allRNA_inimage_sigma{}_cluster{}.png'.format(self.sigma,self.n_clusters),dpi=dpi)
        figname = os.path.join(savepath, savename)
        plt.savefig(figname, dpi=dpi)
        print(f'kmeans results saving as {figname}')
        plt.close()
        gc.collect()
#------------------------------------------------------------------------------   
           
    def plot_heatmap(self, savepath, savename, choose=1, vmax=1, dpi=300):
        
        # choose : 0 show the heatmap in orignal order
        # choose : 1 show the heatmap after gene clusterd
        # vmax : after normalization, gene density range in 0-1. Change vmax, all values above vmax will have same color
        if choose == 0:
            order = self.genes
        elif choose == 1:
            order = self.genes_sorted       

        self.data_pixel.sort_values('cluster',inplace=True)
        self.data_pixel[self.genes] = self.data_pixel[self.genes]/self.data_pixel[self.genes].max()#normalization
        
        cmaps2 = mcl.LinearSegmentedColormap.from_list('mylist',colorlist[1:self.n_clusters],self.n_clusters-1)
   
        fig = plt.figure(figsize=(40,20))
        fig.suptitle('heatmap in all clusters',fontsize=20,fontweight='bold')
        gs = gridspec.GridSpec(10,10)
        plt.subplot(gs[1:,2:10])
        sns.heatmap(self.data_pixel[self.data_pixel['cluster']!=0][order].iloc[::2],xticklabels=1,vmax=vmax,vmin=0,cmap='RdBu_r')
        ##plot 50% of pixels due to memory issues
        plt.tight_layout()
        plt.title('heatmap',fontsize=15,fontweight='bold')

        plt.subplot(gs[1:,:2])
        plt.imshow(np.repeat(np.expand_dims(self.data_pixel[self.data_pixel['cluster']!=0]['cluster'].iloc[::100],axis=1),2000,1),cmap=cmaps2)
        plt.title('Kmeans segmentation',fontsize=15,fontweight='bold')
        plt.xlabel('{} cluster'.format(self.n_clusters),fontsize=15,fontweight='bold')
        
        figname = os.path.join(savepath, savename)
        plt.savefig(figname,dpi=dpi)
        print(f'heatmap saving as {figname}')
        plt.close()
        gc.collect()
        
# -------------------------------------------------------
    def plot_elbow(self, savepath, list_k = list(range(5,20,2)) , dpi=300):
        """
        Calculating the sse between data points 
        and their assigned clusters' centroids
        
        Plotting the sse against a list of k clusters
        """
        sse = []
    #    clustering = Kmeans_clustering(smooth_imgs)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
            
        for k in list_k:
#            clustering.kmeans(n_clusters=k)
#            sse.append(clustering.km_inertia_)
            self.kmeans(n_clusters=k)
            sse.append(self.km_inertia_)
            
            print("----- Plotting kmeans results -------")
#            clustering.plot_kmeans_result(savepath=savepath, 
#                                          savename = f"kmeans_{k}")
            self.plot_kmeans_result(savepath, f"kmeans_{k}")
        
        # Plot sse against k
        plt.figure()
        plt.plot(list_k, sse, '-o')
        plt.xlabel("Number of clusters *k*")
        plt.ylabel("Sum of squared distance")
        
        elbow_figname = os.path.join(savepath, 
                                     f"elbow_k_{list_k[0]}_{list_k[-1]}")
        plt.savefig(elbow_figname, dpi=dpi)
        plt.close()
        
        self.sse = sse
    
# --------------------------------------------------------
    def silhouette_coefficient(self, 
                               savepath, savename,
                               dpi = 300,
                               ):
        
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        cmaps = mcl.LinearSegmentedColormap.from_list('mylist',
                                                      colorlist[:self.n_clusters],
                                                      self.n_clusters)
        
        # FIXME: plotting what?
        # Scatter plot of data with labels? tSNE? kmeans clustering plot?
        image = np.zeros(self.image_shape)
        image[self.data_pixel.y,self.data_pixel.x] = self.data_pixel.cluster
        im = ax1.imshow(image,cmap=cmaps)
        
        ax1.set_title('kmeans clustering', y=1.02)
        
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes('right', size = 0.2, pad = 0.2)
        cbar = fig.colorbar(im, cax=cax, orientation = 'vertical')
        cbar.set_ticks([])
        for c, label in enumerate(np.arange(1, self.n_clusters+1)):
#            print(f"{c}\t{label}")
            cbar.ax.text(15, c, label, ha='center', va='center')
#        cbar.set_label(np.arange(1, self.n_clusters+1))
        
        y_ticks = []
        y_lower, y_upper = 0, 0
        
        for i, cluster in enumerate(np.unique(self.data_pixel['pred'])):            
            cluster_silhouette_vals = self.silhouette_vals[self.data_train_pred_labels == cluster]
            cluster_silhouette_vals.sort()
            y_upper += len(cluster_silhouette_vals)
            
            if cluster!=0:
                ax2.barh(range(y_lower, y_upper), 
                         cluster_silhouette_vals, 
                         color = colorlist[:self.n_clusters][cluster])
                ax2.text(-0.03, (y_lower + y_upper)/2, str(i+1))
            
            y_lower += len(cluster_silhouette_vals)
        
        # Get the average silhouette score 
        avg_score = np.mean(self.silhouette_vals)
        avgline = ax2.axvline(avg_score, linestyle='--', linewidth = 0.5, 
                              color='green', label=f'{avg_score}')
        leg1 = Legend(ax2, [avgline], [f'avg_score = {round(avg_score,2)}'], 
                      loc = 'lower right')
        ax2.add_artist(leg1)
        
        ax2.set_yticks(y_ticks)
        ax2.set_xlabel('Silhouette coefficient score')
        ax2.set_ylabel('Cluster labels')
        ax2.set_title('Silhouette plot for clusters', y=1.02)

        asp = np.diff(ax2.get_xlim())[0]/np.diff(ax2.get_ylim())[0]
        ax2.set_aspect(asp)       
        
        figname = os.path.join(savepath, savename)

        plt.tight_layout()
        plt.savefig(figname, dpi=dpi)
        print(f'kmeans results saving as {figname}')
        plt.close()
        gc.collect()


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
    
    norm_method = "zscore"
    
    for root, dirs, files in os.walk(data_path):
        print(f"Current dir: {root}")
        hdf5_file_list, annotation_csv, genes_to_exclude = parse_folder(root)
        
        # --- Analysing each coord*.hdf5 file in current dir ---
        for hdf5_file in hdf5_file_list:
            gene_index_dict, raw_max_coords, num_genes, blank_genes = \
            read_h5py(hdf5_file, verbose = True, 
                      genes_to_exclude = genes_to_exclude)
            
            smoothdir = os.path.join(os.path.dirname(hdf5_file),
                                     f"kmeans_{norm_method}_{time_str}")
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
            # parameters
            bin_size = 12
            sigma = 60
            
            smooth_imgs = SmoothImgs(gene_index_dict,
                                     raw_max_coords,
                                     num_genes, 
                                     smoothdir,
                                     bin_size= bin_size, 
                                     sigma = sigma,
                                     norm_method = norm_method,
                                     verbose=True)
            
            path_to_imgs = os.path.join(smooth_imgs.smoothdir, "imgs_per_gene")
            if not os.path.exists(path_to_imgs):
                os.makedirs(path_to_imgs)
            
            # Smooth and plot images (check sigma)
            plotALot(img_array = smooth_imgs.smoothed_img_array,
                     gene_index_dict = smooth_imgs.gene_index_dict,
                     savedir = path_to_imgs, 
                     title = f'smoothed {norm_method} (binsize = {bin_size} sigma= {sigma})')
            
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
            clustering.kmeans(n_clusters= n_clusters, training_portion = 0.01)
            
            print("----- Plotting silhouette coefficient")
            clustering.silhouette_coefficient(path_to_clustering, "kmeans")
            
            # End of Kmeans clustering -------------------------------------------