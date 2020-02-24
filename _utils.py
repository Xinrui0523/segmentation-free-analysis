# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 11:08:33 2020

@author: zhouxr
"""
import os
import h5py
import numpy as np
import pprint as pp
import warnings

def roundUp(num, multiple):
    """
    round up to the nearest multiple (e.g. bin_size)
    """
    return num - num % multiple + multiple


def parse_folder(data_path):
    '''
    Specify a directory data_path,
    Return hdf5_file_list, annotation_csv, exclude_genes_txt
        hdf5_file_list: List for hdf5 files 
            - .hdf files start with 'coord'
        annotation_csv: String for the annotation file
            - annotation files start with 'gene_annot'
        genes_to_exclude: String for the text file listing all genes to exclude
    '''
    # -----------------------------------------------------
    # Get .hdf5 files and annotation .csv files from folder
    # Naming conventions:
    #   .hdf5 files begin with 'coord'
    #   annotation file begins with 'gene_annot'
    #   file listing excluded genes: 'genes_to_exclude'
    # -----------------------------------------------------
    hdf5_file_list = []
    annotation_csv = None
    genes_to_exclude_txt = None
    for file in os.listdir(data_path):
        if file.startswith("coord") and file.endswith(".hdf5"):
            hdf5_file_list.append(os.path.join(data_path, file))
        elif file.startswith("gene_annot") and file.endswith(".csv"):
            annotation_csv = os.path.join(data_path, file)
        elif file.startswith("genes_to_exclude") and file.endswith(".txt"):
            genes_to_exclude_txt = os.path.join(data_path, file)
#        elif file.startswith("param") and file.endswith(".txt"):
#            param_txt = os.path.join(data_path, file)
    
    if genes_to_exclude_txt is not None:
        genes_to_exclude = [line.rstrip('\n') for line in open(genes_to_exclude_txt, 'r')]
        print(f"Import {len(genes_to_exclude)} "
                        f"from file {genes_to_exclude_txt} ...")
    else:
        genes_to_exclude = []
        print(f"No file listing genes to exclude...")
            
    print(f"\n --- List of hdf5 files --- ")
    pp.pprint(hdf5_file_list)
    
    return hdf5_file_list, annotation_csv, genes_to_exclude



def read_h5py(hdf5_file, verbose = True, genes_to_exclude = []):
    '''
    Read h5py file as dataframe
    
    Parameters
    ----------
        hdf5_file: string
        verbose: bool
            if True: Print the contents of a hdf5 file, 
            including attributes (if present), shape and datatype of each dataset
        genes_to_exclude: list 
    '''    
    gene_index_dict = {}
    num_genes = 0
    max_coords = np.array((0., 0.))

    
    with h5py.File(hdf5_file, 'r') as f:
        print(f"Reading .hdf5 files ... "
              f"Number of keys: {len(list(f.keys()))}")
        if verbose:
            print(f"\nKeys:\n {list(f.keys())}\n")
        
        blank_gene_list = []
        for gene in f:
            if gene.startswith('Blank-'):
                blank_gene_list.append(gene)
                
            if gene not in genes_to_exclude:
                gene_index_dict[gene] = {}
                if gene == "0":
                    for group in f.get(gene):
                        gene_index_dict[gene][group] = np.array(f[f"{gene}/{group}"])
                        if verbose:
                            print(f[f"{gene}/{group}"])
                            print(np.array(f[f"{gene}/{group}"]))
                else:
                    for attr in f[gene].attrs:
                        gene_index_dict[gene][attr] = f[gene].attrs[attr]
                        if verbose:
                            print(f" - {attr}: {f[gene].attrs[attr]}")
                    gene_index_dict[gene]['Array'] = np.array(f[gene])
                    
                    # Check whether there is Gene-index attribute in .hdf5 file
                    try:
                        gene_index_dict[gene] = f[gene].attrs["index"]
                    except:
                        warnings.warn("Gene-index attribute not found in .hdf5 file. \n"
                                      "Creating arbitrary index to match gene names...")
                        gene_index_dict[gene]["index"] = num_genes
                    
                    if f[gene].ndim == 2:
                        _shape_check = f[gene].shape
                        if _shape_check[1] in [2, 3]:                               
                            max_temp = f[gene][:,:2].max(axis=0)
                            max_coords = np.maximum(max_coords, max_temp)
                        else:
                            warnings.warn(f"Invalid 2D or 3D coords... \n"
                                          f"GENE {gene} \t"
                                          f"Array shape = {np.array(f[gene].shape)} \n"
                                          f"Using the 2nd and 3rd colums as y and x by default")
                            gene_index_dict[gene]["Array"] = f[gene][:, 1:3]

                            max_temp = f[gene][:,1:2].max(axis=0)
                            max_coords = np.maximum(max_coords, max_temp)
                            
                                
                    elif f[gene].ndim == 3:
                        max_temp = f[gene][:,:3].max(axis=0)
                        max_coords = np.maximum(max_coords, max_temp)
                    else:                        
                        raise ValueError(f"Invalid 2D or 3D coords...\n"
                                         f"ndim = {f[gene].ndim} \t"
                                         f"Array shape = {np.array(f[gene].shape)}")
                    num_genes += 1
                    
                if verbose:
                    print(f"For {f[gene]}: \n"
                                 f"\t max_coords: \t {max_temp} \n")
                
                    
        # End of the for-loop
                    
        print(f"\n For all arrays in this file: \n"
              f"max coordinates: {max_coords} \n")
#        if verbose:
#            print(f"\t Gene to index dictionary:")
#            pp.pprint(gene_index_dict)
    
    return gene_index_dict, max_coords, num_genes, blank_gene_list

def get_abundant_genes(gene_index_dict, blank_gene_list, output_file):
    '''
    Compare the gene expression level with the blank genes
    or other reference genes
    
    Remove the genes with lower expression level 
    than any of the reference genes (max(exp_ref_genes))
    
    Parameters:
    -----------
        gene_index_dict: obtained from read_h5py
        ref_genes: list of reference genes
    '''
    
    assert all(gene in list(gene_index_dict.keys()) for gene in blank_gene_list)
    exp_ref = 0
    for gene in blank_gene_list:
        exp_ref = max(exp_ref, gene_index_dict[gene]['Array'].shape[0])
    print(f"min expression level of genes: {exp_ref}")
    
    excluded_genes = []
    
    with open(output_file, 'w+') as f:        
        for gene in gene_index_dict:
            if gene_index_dict[gene]['Array'].shape[0] < exp_ref:
                excluded_genes.append(gene)
                f.write(f"{gene}\n")
    
    return excluded_genes