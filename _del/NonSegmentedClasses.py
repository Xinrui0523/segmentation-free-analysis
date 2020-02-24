# -*- coding: utf-8 -*-
"""
Classes and functions for non-segmented methods.

@author: zhouxr
@date: 15 Dec 2019
"""

import ssam

import os
import math 
import numpy as np
import warnings
import h5py

from datetime import datetime
from collections import OrderedDict

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigCanvas
from sklearn.neighbors import KDTree

plt.ioff()
#
# ====================================================================
#             Classes for reading and printing .h5df files
# ====================================================================
# 

def read_h5py(hdf5_file, verbose = True, exclude_genes = None):
    '''
    Read h5py file as dataframe
    
    hdf5_file: string
    verbose: bool
        if True: Print the contents of a hdf5 file, including attributes (if present), shape and datatype of each dataset
    exclude_genes: list 
    '''    
    pos_dic = OrderedDict()
    width = 0
    height = 0
    
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Reading .hdf5 files ... Number of keys: {len(list(f.keys()))}")
        if verbose:
            print(f"\nKeys:\n {list(f.keys())}\n")
        
        for key in f:            
            pos_dic[key] = {}
            row = 0
            col = 0
            
            if verbose:
                print('_' * 20 + f" {key}" + '_' * 20 + "\n")
                
            if key == "0":
                for group in f.get(key):
                    pos_dic[key][group] = np.array(f[f"{key}/{group}"])
                    if verbose:
                        print(f[f"{key}/{group}"])
                        print(np.array(f[f"{key}/{group}"]))
            else:
                for attr in f[key].attrs:
                    pos_dic[key][attr] = f[key].attrs[attr]
                    if verbose:
                        print(f" - {attr}: {f[key].attrs[attr]}")
                pos_dic[key]['Array'] = np.array(f[key])
                
                if (pos_dic[key]['Array'].shape)[1] in [2, 3]:
                    try:
                        # row: maximum y; axis = 1 --> y axis
                        # col: maximum x; axis = 0 --> x axis
                        row = max(row, pos_dic[key]['Array'][:,1].max())
                        col = max(col, pos_dic[key]['Array'][:,0].max())
                    except : None

                else:
                    warnings.warn(f"Invalid 2D or 3D coords..."
                                  f"{np.array(f[key].shape)[1]}")
#                    raise ValueError(f"Invalid 2D or 3D coords..."
#                                     f"{np.array(f[key].shape)[1]}")
            
            if verbose:
#                print(f"Array for {f[key]}\n\n"
#                                   f"{np.array(f[key])}\n")
                print(f"Array.shape for {f[key]}: \t"
                                         f"{np.array(f[key].shape)}")
                print(f"Max Row: {row} \n"
                      f"Max Col: {col} \n")
            width = max(width, col)
            height = max(height, row)
       # End of the for-loop
       

                 
    if exclude_genes is not None:
        for gene in exclude_genes:
            del pos_dic[gene]
    
    return pos_dic, math.ceil(width), math.ceil(height)


def merge_dic(dic_list):
    '''
    Merge dictionaries of mRNA loci array based on the gene name.
    '''
    merged_dic = dic_list[0]
    if len(dic_list) > 1:
        for key in dic_list[0].keys():
            merged_dic[key] = np.concatenate(list(dic[key]['Array'] for dic in dic_list))
    else:
        for key in dic_list[0].keys():
            merged_dic[key] = dic_list[0][key]['Array']
    
    return merged_dic


def pixel_to_um(pos_dic, um_per_pixel):
    '''
    Invert the unit of mRNA location from pixel to micrometer
    pos_dic: dictionary of mRNA locations
        - key: gene name
        - value: array of mRNA loci
            Default:
                z axis 2: depth (if any)
                y axis 1: row
                x axis 0: col
            if array.shape[1] > 3:
                let the user specifies dimensions for mRNA loci
                
    um_per_pixel: float 
        - iris9: 2960 * 2960 pixels = 209 um * 209 um --> 14 pixels/um
        - BSI: 2048 * 2048 pixels = 221 um * 221 um --> 10 pixels/um
    '''
    points_um = {}
    
    genes = list(pos_dic.keys())
    dim = (pos_dic[genes[0]].shape)[1]
    
    # xaxis: axes[0]
    # yaxis: axes[1]
    # zaxis: axes[2] if len(axes) > 2
    if dim > 3:
        axes = input(f'Array.shape[1] = {dim} \n'
                     f'Please specify the axes for x, y (and z) separated by "," : ')
        axes = tuple(int(i) for i in axes.split(','))
        assert len(axes)<=3 
    else:
        axes = tuple(int(i) for i in range(dim))
    
    print(f"axes: {axes}")
    
    height = 0 # row (max_y)
    width = 0 # col (max_x)
    for k in pos_dic.keys():
        assert dim == (pos_dic[k].shape)[1]
        points_um[k] = np.array(pos_dic[k]) * um_per_pixel
        points_um[k] = points_um[k][:, axes]
        height = max(height, points_um[k][:,1].max())
        width = max(width, points_um[k][:,0].max())
    height = int(math.ceil(height))
    width = int(math.ceil(width))
    print(f"width: {width} \t height: {height}")
    return points_um, width, height

def gaussian_kernel_kde(args):
    
    (bandwidth, gene_name, shape, locations) = args
    
    print(f"Processing gene {gene_name} ...")
    
    maxdist = int(bandwidth * 4)
    span = np.linspace(-maxdist, maxdist, maxdist*2+1)
    X, Y, Z = np.meshgrid(span, span, span)
    
    def create_kernel(x, y, z):
        X_ = (-x+X)/bandwidth
        Y_ = (-y+Y)/bandwidth
        Z_ = (-z+Z)/bandwidth        
        kernel = np.exp(-0.5*(X_**2 + Y_**2 + Z_**2))
        return kernel
    
    pd = np.zeros(shape)
    for loc in locations:
        print(loc)
        int_loc = [int(i) for i in loc]
        rem_loc = [i%1 for i in loc]
        kernel = create_kernel(*rem_loc)
        
        pos_start = [i-maxdist for i in int_loc]
        pos_end = [i+maxdist+1 for i in int_loc]
        
        kernel_pos_start = [abs(i) if i<0 else 0 for i in pos_start]
        kernel_pos_end = [maxdist*2+1-(i-j) if i>j else maxdist*2+1 for i,j in zip(pos_end, shape)]
        
        pos_start = [0 if i<0 else i for i in pos_start]
        pos_end = [j if i>=j else i for i,j in zip(pos_end, shape)]
        # pos_end = [min(i,j) for i,j in zip(pos_end, shape)]
        
        slices = tuple([slice(i,j,1) for i,j in zip(pos_start, pos_end)])
        kernel_slices = tuple([slice(i,j,1) for i,j in zip(kernel_pos_start, kernel_pos_end)])
        pd[slices] += kernel.swapaxes(0,1)[kernel_slices]
    
    # Probability density
    pd/=pd.sum()
    
    # Gene expression
    pd*=len(locations)  
    
    return pd

def plot_raw_dataset_um(points_um, 
                     savedir='',
                     title='images',
                     grid=(3, 3),
                     figsize=(8,8),
                     size = 0.1,
                     dpi=300,
                     invert_yaxis=True,
                     ):
    '''
    Plot the mRNA spots for each gene.
    '''
    
    print('Plotting mRNA spots of selected genes...')
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    genes_per_plot = grid[0] * grid[1]
    num_plots, remain = divmod(len(points_um), genes_per_plot)
    
    if remain !=0:
        num_plots += 1
    
    gene_list = list(points_um.keys())
    gene_idx = 0
    
    for plot_num in range(num_plots):
        fig = Figure(figsize=figsize,dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)
        
        for gridpos in range(genes_per_plot):
            if gene_idx == len(points_um):
                break
            ax = fig.add_subplot(grid[0], grid[1], gridpos+1)

            ax.scatter(points_um[gene_list[gene_idx]][:,1],
                       points_um[gene_list[gene_idx]][:,0], 
                       s = size,
                       )
            ax.set_aspect('equal')
            ax.set_title(gene_list[gene_idx])
            
            if invert_yaxis:
                ax.invert_yaxis()
            gene_idx += 1
        
        fig.suptitle(title + f'\n {plot_num + 1} of {num_plots}')
        fig.tight_layout(rect=(0, 0, 1, .94))
        
        savename = (f'raw_image_{plot_num+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        fig.savefig(os.path.join(savedir, savename), dpi=dpi)
        
        canvas.close()
        fig.clear()
        plt.close()

def load_ssam_dataset(pos_dic_um, selected_genes, width, height):
    '''
    Create an SSAMDataset using selected genes, mapping onto a pre-defined image.
    '''
    print('Loading SSAMDataset ...')
    mrna_loci_um = [pos_dic_um[g] for g in selected_genes]
    
    ds = ssam.SSAMDataset(selected_genes, 
                          mrna_loci_um, 
                          width, 
                          height, 
                          )

    return ds

def cal_kdes(SSAMds, kdes, **kwargs):
    '''
    Calculate KDE results for a SSAM dataset, estimating the density of mRNA
        SSAMds: Object SSAMDataset
        kdes: List of kde value
        **kwargs: Params for object SSAMAnalysis and running KDE
            - fast_kde: bool; default True
            - ncores: int; default 10
            - data_path: string; default '.'
            - verbose: bool; default True
            - use_mmap: bool; default False
    '''
    fast_kde = kwargs.get('fast_kde', True)
    ncores = kwargs.get('ncores', 10)
    data_path = kwargs.get('data_path', '.')
    verbose = kwargs.get('verbose', True)
    use_mmap = kwargs.get('use_mmap', False)
    
    for bandwidth in kdes:
        print(f'--- KDE = {bandwidth} --- Calculating...')
        kde_path = os.path.join(data_path, f'kde_{str(bandwidth).rstrip("0").rstrip(".")}') 
        if not os.path.exists(kde_path):
            os.makedirs(kde_path)
        analysis = ssam.SSAMAnalysis(SSAMds, ncores = ncores, 
                                     save_dir = kde_path,
                                     verbose = verbose,
                                     )
        if fast_kde:
            analysis.run_fast_kde(bandwidth=bandwidth, use_mmap=use_mmap)
        else:
            analysis.run_kde(bandwidth=bandwidth, use_mmap=use_mmap)    
    

def compare_kde_level(kde_files, 
                      savedir='.', 
                      title='plot',
                      grid = (2, 2),
                      figsize = (8, 8),
                      dpi = 300,
#                      invert_yaxis=True,
                      xlim=None, ylim=None,
                      ):
    '''
    Zooming in a particular case to select the proper KDE smoothing level.
        - kde_files: a dictionary of kde result for comparison;
            key: kde_level
            value: kde_file_name
        - savedir, samename: the output plot
        - xlim, ylim: the zooming region
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    kde_per_plot = grid[0]*grid[1]
    num_plots, remain = divmod(len(kde_files), kde_per_plot)
    if remain!=0:
        num_plots += 1
    
    kde_list = list(kde_files.keys())
    kde_idx = 0
    for plot_num in range(num_plots):
        fig = Figure(figsize=figsize, dpi=dpi)
        canvas = FigCanvas(fig)
        fig.set_canvas(canvas)
        
        for gridpos in range(kde_per_plot):
            if kde_idx == len(kde_files):
                break
            
            kde = kde_list[kde_idx]
            kde_file = kde_files[kde]            
            array_kde = np.load(kde_file)
            
            # KDE files naming convention (example): pdf_sd1_bw2_GENE.npy            
            ax = fig.add_subplot(grid[0], grid[1], gridpos+1)
            ax.imshow(array_kde[:,:,0], aspect='equal')
            scalebar = ScaleBar(1,'um',pad=0.2, font_properties={'size':12})
            ax.add_artist(scalebar)
            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)
#            if invert_yaxis:
#                ax.invert_yaxis()
            ax.set_title(f'kde = {kde}')
            kde_idx+=1

        fig.suptitle(f'{title}_{plot_num+1}/{num_plots}')
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        
        savename = (f'{title}_{plot_num+1}_'
                    f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        fig.savefig(os.path.join(savedir,savename),dpi=dpi)
        canvas.close()
        fig.clear()
        
    return True


def hist_mrna_expression(ds, savedir, 
                         figsize=[8,8],
                         dpi = 300,
                         **kwargs):
    '''
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
        pass
    viewport = kwargs.get('viewport', 0.5)
    min_exp = kwargs.get('min_exp', None)
    
    gindices = np.arange(len(ds.genes))
    for gidx in gindices:
        fig = plt.figure(figsize=figsize)
        n, bins, patches = plt.hist(ds.vf[..., gidx][np.logical_and(ds.vf[...,gidx]>0, ds.vf[...,gidx]<viewport)], bins=100, log=True, histtype=u'step')
        ax = plt.gca()
        ax.set_xlim=([0, viewport])
        ax.set_ylim=([n[0], n[-1]])
        if min_exp is not None:
            ax.axvline(min_exp, c='red', ls='--')
            pass

        ax.set_title(ds.genes[gidx])
        ax.set_xlabel('Expression (%s)' %ds.genes[gidx])
        ax.set_ylabel('Count')

        fig.tight_layout()
        fig.savefig(os.path.join(savedir, ds.genes[gidx]), dpi=dpi)
        plt.close()
    
    pass

def hist_l1_norm(ds, savedir, 
                 figsize=[8,8],
                 dpi = 300,
                 **kwargs):
    '''
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    
    viewport = kwargs.get('viewport', 0.5)
    min_norm = kwargs.get('min_norm', None)
    
    plt.figure(figsize=figsize)
    n, _, _ = plt.hist(ds.vf_norm[np.logical_and(ds.vf_norm>0, ds.vf_norm<viewport)], bins=1000, log=True, histtype=u'step')
    
    ax = plt.gca()
    if min_norm is not None:
        ax.axvline(min_norm, c='red', ls='--')
        pass
    ax.set_xlabel('Histogram of L1-norm (Total gene expression)')
    ax.set_ylabel('Count')
    
    plt.xlim([0, viewport])
    plt.ylim([n[-1], n[0]])
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, 'hist_l1_norm.png'), dpi=dpi)
#    plt.show()
    plt.close()
    pass

def hist_knn_density(rho, savedir,
                     figsize=[8,8],
                     dpi = 300,
                     **kwargs):    
    plt.figure(figsize=figsize)
    plt.hist(rho, bins=100, histtype=u'step')
    
    min_knn_density = kwargs.get('min_knn_density', None)
    if min_knn_density is not None:
        plt.axvline(min_knn_density, color='r', linestyle='--')

    ax=plt.gca()
    ax.set_xlabel('Local KNN density')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savedir, f'hist_knn_density.png'))
    plt.close()
    pass

def plot_l1norm_vf(ds, savedir, savename, 
                   mask = None, 
                   rotate= 0,
                   figsize = [8,8],
                   cmap = 'Greys',
                   dpi = 300,
                   xlim = None, ylim = None):
    '''
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.figure(figsize=figsize)
    ds.plot_l1norm(cmap=cmap, rotate=rotate)
    if mask is None:
        if rotate == 0 or rotate == 2:
            plt.scatter(ds.local_maxs[1], ds.local_maxs[0],
                        c='blue', s=0.5)
        elif rotate ==1 or rotate == 3:
            plt.scatter(ds.local_maxs[0], ds.local_maxs[1],
                        c='blue', s=0.5)
    else:
        if rotate == 0 or rotate == 2:
            plt.scatter(ds.local_maxs[1][mask], ds.local_maxs[0][mask], 
                        c='blue', s=0.5)
        elif rotate == 1 or rotate == 3:
            plt.scatter(ds.local_maxs[0][mask], ds.local_maxs[1][mask],
                        c='blue', s=0.5)
    
    scalebar = ScaleBar(1, 'um', pad=0.2, font_properties={'size':10})
    ax = plt.gca()    
    ax.add_artist(scalebar)
    ax.set_aspect('equal', adjustable='box')
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
#    plt.show()
    plt.savefig(os.path.join(savedir, savename))
    plt.close()
    
    pass

def plot_celltype(ds, savedir, savename, 
                  rotate = 0,
                  min_r = 0, 
                  dpi=300, 
                  xlim=None, ylim=None):
    '''
    Plot a cell-type map: 
        a correlation map betwen the centroids and the vector field
        showing only the vector field with >min_r correlation to centroids
    Params:
    - int rotate:
        0, 1, 2, 3
    - float min_r: 
        minimum correlation threshold for SHOWing on the cell-type map
    '''
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    plt.figure(figsize=[8,8])
    ds.plot_celltypes_map(rotate = rotate, min_r=min_r, set_alpha=False)
    
    scalebar = ScaleBar(1, 'um', pad=0.2, font_properties={'size':10})
    ax=plt.gca()
    ax.add_artist(scalebar)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    sns.despine(top=False, bottom=False, left=False, right=False)
    plt.tight_layout()
    plt.savefig(os.path.join(savedir,savename), dpi=dpi)
    plt.show()
    plt.close()
    
    pass
    
    
def plot_diagnostics(ds, savedir, rotate = 0,
                     known_signatures = [], dpi=300):
    '''
    Diagnostics for each cluster:
        - Panel 1: 
            Location of the localmax orginating from the cluster 
            embedded into L1 norm
        - Panel 2:
            Centroid embedded into the vector field
        - Panel 3:
            Centroid of the cluster
        - Panel 4:
            t-SNE or UMap embedding of localmax
    '''
    for idx in range(len(ds.centroids)):
        print(f'---Diagnostic plot {idx} of {len(ds.centroids)} ---')
        plt.figure(figsize=[50, 15])
        ds.plot_diagnostic_plot(idx, rotate = rotate)
        plt.tight_layout()
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        plt.savefig(os.path.join(savedir, f'diagplot_centroid_{idx}.png'), dpi=dpi)
        plt.close()
    pass

def run_ssam(ds, analysis, 
             kde_path, 
             subset_path,
             search_size = 3,
             min_norm = 0.04, 
             min_expression = 0.025,
             min_knn_density = 0.005,
             normalize = 'sctransform',
             min_centroid_correlation = 0.6,
             pca_dims = 10,
             resolution = 0.6, 
             filter_method = 'local',
             filter_params = {"block_size": 151,
                              "method": "mean",
                              "mode": "constant",
                              "offset": 0.03,
                              },
             embedding = 'tsne',
             dpi=300,
             xlim = None, 
             ylim = None,
             ):
    # Implemented normalization methods
    normalize_methods = ['sctransform']
    
    analysis.find_localmax(search_size=search_size, 
                           min_norm=min_norm, 
                           min_expression=min_expression)
    
    output_path = os.path.join(subset_path, 
                               f'norm_{min_norm}_exp_{min_expression}')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    plot_l1norm_vf(ds, output_path, 
                   f'l1norm_vf(before_knn)_{len(ds.local_maxs[0])}.png')
    
    local_max_array = np.array([ds.local_maxs[0], ds.local_maxs[1]]).T
    kdt = KDTree(local_max_array, leaf_size=10, metric='euclidean')
    rho = 100/(np.pi*kdt.query(local_max_array, k=30)[0][:,29]**2)
    
    hist_knn_density(rho, output_path)
    
    mask = rho>min_knn_density
    print('After removing supurious local maxs: %i' % sum(mask))
    plot_l1norm_vf(ds, output_path, 
                   f'l1norm_vf(after_knn)_{sum(mask)}.png', 
                   mask=mask)
    
    ds.local_maxs = tuple(ds.local_maxs[i][mask] for i in range(3))
    
    print('Normalizing ...')
    if normalize not in normalize_methods:
        print(f'Implemented normalization methods are: \n {normalize_methods}')
    elif normalize == 'sctransform':
        analysis.normalize_vectors_sctransform()
    
    print('Clustering vectors...')
    # Clustering normalized local maxima vectors
    analysis.cluster_vectors(pca_dims=pca_dims, 
                             resolution=0.6, 
                             prune=1.0/15.0, 
                             snn_neighbors=30,
                             max_correlation=1.0, 
                             metric='correlation',
                             subclustering=False,
                             centroid_correction_threshold=min_centroid_correlation,
                             )
    # Centroids as cell-type signatures
    analysis.map_celltypes()
    analysis.filter_celltypemaps(min_norm=filter_method, 
                                 filter_params=filter_params, 
                                 fill_blobs=True, 
                                 min_blob_area=5,
                                 )
    
    #----------------------------------------------------
    #   Plot cell-type map, embedding map (TSNE or UMAP)
    #----------------------------------------------------
    print('Plotting celltype map...')
    
    clustering_result_path = os.path.join(output_path, 
                                          (f'resolution_{resolution}_'
                                           f'pca_{pca_dims}_'
                                           f'min_centroid_correlation_{min_centroid_correlation}'
                                           ))
    fig_name = f'cell_type_map.png'
    plot_celltype(ds, clustering_result_path, fig_name, dpi=dpi)
    
    diagnostic_dir = os.path.join(clustering_result_path, 'dignostics')
    
    print('Plotting embedding map...')
    plt.figure(figsize=[8,8])
    if embedding == 'tsne':
        ds.plot_tsne(pca_dims=pca_dims, metric='correlation', 
                     s=15, run_tsne=True,)
    elif embedding == 'umap':
        ds.plot_umap(pca_dims=pca_dims, metric='correlation',
                     s=15, run_umap=True)
    else:
        print('Not implemented embedding method')
    plt.axis('off')
    plt.savefig(os.path.join(clustering_result_path, f'{embedding}.png'), dpi=dpi)
    plt.close()
    
    print('Plotting diagnostic plots for each cluster...')
    plot_diagnostics(ds, diagnostic_dir)
    
    return True
