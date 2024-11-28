import anndata
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from typing import List




class FatePredictor:
    def __init__(self, 
                 adata, 
                 reld_key: str, 
                 group_by: str, 
                 source: str, 
                 target: List[str],
                 resolve_ties: bool = True,
                 copy: bool = False):
        """
        Initialize the FatePredictor class with necessary parameters.

        Parameters:
        - adata: The input AnnData object.
        - reld_key: Key to the relative dimension matrix in obsp.
        - group_by: Column in obs specifying the cluster annotations.
        - source: Source cluster name.
        - target: List of target cluster names.
        - resolve_ties: Whether to resolve ties during classification.
        - copy: Whether to copy the input AnnData object.
        """
        self.adata = adata.copy() if copy else adata
        self.reld_key = reld_key
        self.group_by = group_by
        self.source = source
        self.target = target
        self.resolve_ties = resolve_ties


    def _validate_inputs(self):
    
        """Validate the input parameters."""

        if self.reld_key not in self.adata.obsp:
            raise ValueError(f"{self.reld_key} is not present in `adata.obsp`. Please check the AnnData object.")
        
        if self.group_by not in self.adata.obs:
            raise ValueError(f"{self.group_by} is not present in `adata.obs`. Please check the column names.")
        
        if self.source not in self.adata.obs[self.group_by].unique():
            raise ValueError(f"Source '{self.source}' is not present in the '{self.group_by}' column.")
        
        if not all(target_cluster in self.adata.obs[self.group_by].unique() for target_cluster in self.target):
            raise ValueError("One or more target clusters are not present in the dataset.")
        
        if len(self.target) < 2:
            raise ValueError("Please specify at least two target clusters for comparison.")
        

    def _create_anndata(self):
        """
        Create a new AnnData object by subsetting the input adata based on source and target indices.
        """
        # Validate inputs
        self._validate_inputs()

        # Get indices of source cells
        source_indices = self.adata.obs[self.adata.obs[self.group_by] == self.source].index

        # Get indices of target cells
        target_indices = self.adata.obs[self.adata.obs[self.group_by].isin(self.target)].index

        # Subset the relative dimension matrix
        reld = self.adata.obsp[self.reld_key]
        reld_subset = reld[source_indices, target_indices]

        # Create a new AnnData object
        new_adata = anndata.AnnData(X=reld_subset)

        # Set .obs_names and .var_names
        new_adata.obs_names = source_indices
        new_adata.var_names = target_indices

        # Add group metadata
        new_adata.obs[self.group_by] = self.source  # All obs are source
        new_adata.var[self.group_by] = self.adata.obs.loc[target_indices, self.group_by]  # Corresponding groups for targets

        self.new_adata = new_adata
        print(f"New AnnData object created: {len(source_indices)} sources × {len(target_indices)} targets.")
