# Code for segmentation-free analysis
## List of scripts
- spatialComparison.py
  
  Functions and Classes to 
  1. compare and cluster spatial gene distributions (Nigel Aug 2019)
  2. select gene panels based on spatial gene distributions (Xinrui Jan 2020)
  
  Inputs:
    1. Data path to the files
      
      a. coordinate files starts with "coord" and ends with ".hdf5"
      
      b. (optional) annotation file starts with "gene_annot" and ends with ".txt"
      
      c. (optional) text file listing genes to exclude, starting with "exclude_genes".
      
   Outputs: 
     1. By default, the main script creates heatmap for all implemented matrix, excluding only genes imported from the file starting with "exclude_genes".
   
