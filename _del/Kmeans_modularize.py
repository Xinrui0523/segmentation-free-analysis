#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 08:52:21 2019
Functions and Classes to us kmeans clustering and
clustering genes and plot heatmap

@author: linl

updated on 15 Jan 2020 by Xinrui
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcl
import h5py
import cv2
import os
import re
import gc
from sklearn.cluster import KMeans
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.gridspec as gridspec
import pprint as pp
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
#------------------------------------------------------------------------------

def gaussian_pdf(x,sigma): #unnormalized pdf
    
    return np.exp(-x**2/(2*sigma**2))    
#------------------------------------------------------------------------------
    
def Gaussian_kernel(sigma):
    
    k = min(int(((sigma*6)//2+1)*2),1900)
    gaussian_kernel = gaussian_pdf(np.linspace(-k//2,k//2,k+1),sigma)
    gaussian_kernel = np.expand_dims(gaussian_kernel/np.sum(gaussian_kernel),-1)  
    return gaussian_kernel
#------------------------------------------------------------------------------

#==============================================================================
#                                MAIN CLASS    
#==============================================================================
class Kmeans_clustering(object):
    
    def __init__(self,
                 hdf5_file,
                 height,
                 width,
                 downsample,
                 verbose = False,
                 ):
        
       self.hdf5_file = hdf5_file 
       self.height = height
       self.width = width
       self.downsample = downsample
       self.verbose = verbose
       
       self.sigma = None
       self.keys = []
       self.keys_sorted = []
       self.images = []
       self.data_pixels = {}
       self.all_genes = {}
       self.corr_maxtix = []
       self.n_clusters = None
       self.exclude_genes = []
       
#------------------------------------------------------------------------------
       
    def read_h5py(self, exclude_genes):
        #read h5py file as dataframe       
    
        with h5py.File(self.hdf5_file, 'r') as f:
            for key in f:
                self.all_genes[key] = {}
                row = 0
                col = 0
                for attr in f[key].attrs:
                    self.all_genes[key][attr] = f[key].attrs[attr]
                self.all_genes[key]['Array'] = np.array(f.get(key))
                try:
                    row = max(row,self.all_genes[key]['Array'][:,1].max())
                    col = max(col,self.all_genes[key]['Array'][:,0].max())
                except : None
                if key == "0":
                    for group in f.get(key):
                        self.all_genes[key][group] = np.array(f["{}/{}".format(key, group)])  
        
        self.exclude_genes = exclude_genes
        for gene in self.exclude_genes:
            del self.all_genes[gene]
#------------------------------------------------------------------------------
    
    def gaussian_filter(self, sigma):
        
        self.sigma = sigma
        self.keys = sorted(self.all_genes.keys(),key=lambda x : ('Blank' in x,x))
        self.images = np.zeros((len(self.all_genes),int(self.height//self.downsample),
                                int(self.width//self.downsample))).astype(np.float32)
        gaussian_kernel = Gaussian_kernel(self.sigma)
        for counter in range(len(self.all_genes)):
            for i,j in self.all_genes[self.keys[counter]]['Array'][:,:2]:
                i,j = int(np.round(i//self.downsample)),int(np.round(j//self.downsample))
                self.images[counter,i,j] += 1
        
            self.images[counter]= cv2.filter2D(cv2.filter2D(self.images[counter],-1,gaussian_kernel),
                                -1,gaussian_kernel.T)
            if self.verbose:
                print ('Done RNA:',self.keys[counter])        
#------------------------------------------------------------------------------
    
    def pca(self):
        
        C = np.cov(self.data_pixel[self.keys].T) #pca on covariance matrix
        a,b,c = np.linalg.svd(C) #pca on covariance matrix
        pc = np.matmul(self.data_pixel[self.keys].values,a[:,:3])
        self.data_pixel['pc1'] = pc[:,0]

        max_min = np.percentile(self.data_pixel['pc1'],99)-np.percentile(self.data_pixel['pc1'],1)
        self.data_pixel['pc1'] = (self.data_pixel['pc1']-np.percentile(self.data_pixel['pc1'],1))/max_min
        self.data_pixel['pc1'] = np.clip(self.data_pixel['pc1'],0,1)
     
#------------------------------------------------------------------------------
        
    def kmeans(self, n_clusters):
        
        self.n_clusters = n_clusters
        x,y = np.mgrid[0:int(self.height//self.downsample),0:int(self.width//self.downsample)]
        impt = np.vstack([x.ravel(),y.ravel()]).T
        self.data_pixel = pd.DataFrame(impt,columns=['x','y'])
        for i in range(len(self.keys)):
            self.data_pixel[self.keys[i]] = self.images[i,impt[:,0],impt[:,1]]
            
        km = KMeans(self.n_clusters,n_jobs=6,random_state=233) #use 6 threads
        km.fit(self.data_pixel.iloc[::20][self.keys]) #train kmeans model with 5% of data due to memory issues
        self.data_pixel['pred'] = km.predict(self.data_pixel[self.keys]) 
        
        self.pca() #get the correlation of each cluster
        
        temp = self.data_pixel[['pred','pc1']].groupby('pred').apply(np.mean).reset_index(drop=1)
        temp.rename(columns={'pc1':'pc1_mean'},inplace=True)
        self.data_pixel = pd.merge(self.data_pixel,temp,how='left',on='pred')
        
        uq =  np.unique(self.data_pixel.pc1_mean)
        self.data_pixel['cluster']=np.zeros((len(self.data_pixel['pred']==0),1))
        for i in range(len(uq)):
           self.data_pixel.loc[self.data_pixel['pc1_mean']==uq[i],'cluster'] = (i+1)%self.n_clusters
#------------------------------------------------------------------------------
        
    def corr(self):
        
        self.corr_maxtix = np.corrcoef(self.data_pixel[self.keys].T)
#------------------------------------------------------------------------------        
        
    def hierarchical_clusting(self, m_clusters=7, form='average'):
        #hierarchical clusting
        
        assert m_clusters>1 , 'error: cluster number should be larger than 1'
        self.corr()
        
        linked = linkage(self.corr_maxtix, form)
        dist= (linked[-(m_clusters-1)]+linked[-(m_clusters)])[2]/2
        dendro  = dendrogram(linked,
                             orientation='top',
                             labels=self.keys,
                             distance_sort='descending',leaf_font_size=4,
                             show_leaf_counts=True,color_threshold=dist)
    
        self.keys_sorted = sorted(get_clusters(linked,dist=dist),key=lambda x : np.argwhere(np.array(dendro['leaves'])==x[0]))
        keys_sorted_name = list(map(lambda x : list(map(lambda x : self.keys[x],x)),self.keys_sorted))
        self.keys_sorted = np.concatenate(self.keys_sorted)
        self.keys_sorted = [self.keys[x] for x in self.keys_sorted] 
#------------------------------------------------------------------------------   
           
    def plot_kmeans_result(self, savepath, dpi=300):
        
        colorlist = ('#000000', '#ff851b', '#ff4136', '#0074d9', '#85144b','#2ecc40','#ffdc00',
                 '#39cccc', '#f012be','#ffffff','#01ff70','#4c72b0','#55a868','#8172b2',
                 '#64b5cd','#9a6200','#7a9703','#ad0afd','#516572','#ffc5cb')
        cmaps = mcl.LinearSegmentedColormap.from_list('mylist',colorlist[:self.n_clusters],self.n_clusters)   

        image = np.zeros((int(self.height//self.downsample),int(self.width//self.downsample)))
        image[self.data_pixel.x,self.data_pixel.y] = self.data_pixel.cluster
        plt.imshow(image,cmap=cmaps)
        plt.title('kmeans_allRNA_inimage_sigma{}_cluster{}.png'.format(self.sigma,self.n_clusters),fontsize=12,fontweight='bold')
        cbar = plt.colorbar()
        cbar.set_ticks(np.arange(self.n_clusters+1))
        cbar.set_ticklabels(np.arange(self.n_clusters+1))
#        plt.savefig(savepath + '_kmeans_allRNA_inimage_sigma{}_cluster{}.png'.format(self.sigma,self.n_clusters),dpi=dpi)
        figname = os.path.join(savepath, 
                               f'kmeans_allRNA_inimage_sigma{self.sigma}'
                               f'_cluster{self.n_clusters}.png')
        plt.savefig(figname, dpi=dpi)
        print(f'kmeans results saving as {figname}')
        plt.close()
        gc.collect()
#------------------------------------------------------------------------------   
           
    def plot_heatmap(self, savepath, choose=1, vmax=1, dpi=300):
        
        # choose : 0 show the heatmap in orignal order
        # choose : 1 show the heatmap after gene clusterd
        # vmax : after normalization, gene density range in 0-1. Change vmax, all values above vmax will have same color
        if choose == 0:
            order = self.keys
        elif choose == 1:
            order = self.keys_sorted       

        self.data_pixel.sort_values('cluster',inplace=True)
        self.data_pixel[self.keys] = self.data_pixel[self.keys]/self.data_pixel[self.keys].max()#normalization
        
        colorlist = ('#000000', '#ff851b', '#ff4136', '#0074d9', '#85144b','#2ecc40','#ffdc00',
                 '#39cccc', '#f012be','#ffffff','#01ff70','#4c72b0','#55a868','#8172b2',
                 '#64b5cd','#9a6200','#7a9703','#ad0afd','#516572','#ffc5cb')
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
        
        figname = os.path.join(savepath, 
                               f'heatmap_sigma{self.sigma}'
                               f'_cluster{self.n_clusters}'
                               f'_vmax{vmax}.png')
        plt.savefig(figname,dpi=dpi)
        print(f'heatmap saving as {figname}')
        plt.close()
        gc.collect()
#------------------------------------------------------------------------------
        
#==============================================================================
#                                SCRIPT  
#==============================================================================        

if __name__ == "__main__":
    root_path = 'dataset8/'
    file_path = os.listdir(root_path)

    data_path = []
    file_name = []
    #TODO: choose which dataset to handle
    dataset = 4

    for path in file_path: #get all file path under roor_path files
        data_path.append(root_path + path)    
        for dirlist in os.listdir(root_path + path): #get all coords .hdf5 file name s
            if dirlist.endswith('.hdf5') and dirlist.startswith('coords'):
                file_name.append(root_path + path + '/' + dirlist) 
    
    #give genes to excludes
    genes_to_exclude = [f"Blank-{num}" for num in range(1,6)]
    print("Genes to exclude:")
    pp.pprint(genes_to_exclude)
    #read orignal image's width-height ratio, we save it in tha files name
    ratio = np.array(re.findall('\d+',data_path[dataset])[-2:]).astype(int)
#------------------------------------------------------------------------------
    clustering = Kmeans_clustering(file_name[dataset],                                                              
                               height=ratio[1]*2960,
                               width=ratio[0]*2960,
                               downsample=10)
    
    clustering.read_h5py(genes_to_exclude)
    clustering.gaussian_filter(sigma=5)
    clustering.kmeans(n_clusters=10)
    clustering.plot_kmeans_result(savepath=file_path[dataset])
    clustering.hierarchical_clusting(m_clusters=7, form='average')
    clustering.plot_heatmap(file_path[dataset],1,1)
    
    