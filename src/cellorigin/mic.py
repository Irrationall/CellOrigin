from multiprocessing import Pool, cpu_count
from anndata import AnnData
from minepy import MINE
import scipy.sparse as sp
import scanpy as sc
import numpy as np
import time

from typing import Optional, Dict
from packaging import version
import logging




# Timing decorator for performance tracking
def calc_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper




### MIC Calculation ###
def init_mine():
    """Initializer for multiprocessing to avoid reinitializing MINE."""
    global mine
    mine = MINE()




def compute_mi_pair(args):
    """Compute MIC for a single pair of data points."""
    i, j, data = args
    mine.compute_score(data[i, :], data[j, :])
    return (i, j, mine.mic())




@calc_time
def calculate_MIC(data, num_processes=None):
    """
    Calculate the MIC (Maximal Information Coefficient) matrix for the input data.

    Parameters:
    - data: 2D numpy array (cells x genes or PCA-reduced data)
    - num_processes: Number of parallel processes to use (default: CPU count)

    Returns:
    - mi_matrix: 2D symmetric MIC matrix (cells x cells)
    """
    num_cells = data.shape[0]

    # Determine the number of processes
    if num_processes is None:
        num_processes = cpu_count()

    # Prepare all unique pairs for MIC computation (upper triangular)
    pairs = [(i, j, data) for i in range(num_cells) for j in range(i + 1, num_cells)]

    # Use multiprocessing to compute MIC for all pairs
    with Pool(processes=num_processes, initializer=init_mine) as pool:
        results = pool.map(compute_mi_pair, pairs)

    # Convert results to a NumPy array for faster operations
    results_array = np.array(results)  # Convert list of tuples to 2D array

    # Extract columns from the results array
    row_indices = results_array[:, 0].astype(int)  # First column is row index
    col_indices = results_array[:, 1].astype(int)  # Second column is column index
    values = results_array[:, 2]                   # Third column is MIC values

    # Create a sparse matrix directly
    mic_matrix = sp.coo_matrix((values, (row_indices, col_indices)), shape=(num_cells, num_cells)).tocsr()

    return mic_matrix




def _prepare_data_for_pca(adata: AnnData, 
                          use_highly_variable: bool, 
                          hv_kwargs: Dict, 
                          use_scale: bool) -> AnnData:
    """
    Prepares the AnnData object for PCA by optionally subsetting to highly variable genes and scaling data.

    Parameters:
    - adata: AnnData object
    - use_highly_variable: Whether to subset to highly variable genes
    - hv_kwargs: Keyword arguments for `sc.pp.highly_variable_genes`
    - use_scale: Whether to scale the data

    Returns:
    - adata: Prepared AnnData object
    """

    if use_highly_variable:
        sc.pp.highly_variable_genes(adata, **hv_kwargs)
        logger.info(f"Found {adata.var.highly_variable.sum()} highly variable genes.")
    
    if use_scale:
        sc.pp.scale(adata)
        logger.info("Data scaled before PCA.")

    if use_highly_variable:
        if version.parse(sc.__version__) >= version.parse("1.10.0"):
            # Use `mask_var` argument for Scanpy versions >= 1.10.0
            logger.info("Using `mask_var='highly_variable'` for PCA.")
            sc.pp.pca(adata, mask_var='highly_variable')

        else:
            # Use `use_highly_variable` argument for older Scanpy versions
            logger.info("Using `use_highly_variable=True` for PCA (deprecated in scanpy 1.10.0).")
            sc.pp.pca(adata, use_highly_variable=True)
    else:
        # PCA without subsetting to HVGs
        sc.pp.pca(adata)
        logger.info("PCA performed without subsetting to HVGs.")

    logger.info("PCA completed.")

    return adata




logger = logging.getLogger(__name__)

@calc_time
def generate_mi_mat(adata, 
                     use_highly_variable: bool = False,
                     hv_kwargs: Optional[Dict] = None,
                     use_scale: bool = False,
                     n_components: int = 30,
                     num_processes: int = 1,
                     force_PCA: bool = False,
                     mic_key: str = 'MIC'):
    """
    Generate an MIC matrix and store it in the AnnData object.

    Parameters:
    - adata: AnnData object
    - use_highly_variable: Whether to use highly variable genes for PCA
    - hv_kwargs: Dictionary of keyword arguments for `sc.pp.highly_variable_genes`
    - use_scale: Whether to scale the data before PCA
    - subset: Whether to subset to highly variable genes
    - n_components: Number of PCA components to use
    - num_processes: Number of processes for MIC calculation
    - force_PCA: Whether to force PCA even if already present
    - mi_key: Key to store the MIC matrix in `adata.obsp`

    Returns:
    - adata: AnnData object with MIC matrix stored in `adata.obsp[mi_key]`
    """
    _adata = adata.copy()

    # Check if PCA is already computed and can be reused
    if 'X_pca' in _adata.obsm and not force_PCA:
        logger.info('X_pca already in adata.obsm. Using existing PCA matrix.')
        pca_mat = _adata.obsm['X_pca'][:, :n_components]

    else:
        if 'X_pca' in _adata.obsm and force_PCA:
            logger.info("Force PCA is enabled. Recomputing PCA and overwriting existing X_pca.")
            
        else:
            logger.info("Performing PCA with assigned parameters.")
        
        _adata = _prepare_data_for_pca(_adata, use_highly_variable, hv_kwargs, use_scale)
        pca_mat = _adata.obsm['X_pca'][:, :n_components]
        
    # Calculate MIC matrix
    logger.info("Calculating MIC matrix...")
    mi_matrix = calculate_MIC(pca_mat, num_processes=num_processes)

    # Set diagonal elements to zero (self-MIC not meaningful)
    #np.fill_diagonal(mi_matrix, 0)

    # Store MIC matrix in the AnnData object
    adata.obsp[mic_key] = mi_matrix
    logger.info(f"MIC matrix stored in adata.obsp[{mic_key}].")

    return adata

