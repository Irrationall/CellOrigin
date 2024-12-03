import anndata
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import numpy as np
from typing import List, Union




class FatePredictor:
    def __init__(self, 
                 adata, 
                 reld_key: str, 
                 group_by: str, 
                 source: str, 
                 target: List[str],
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

        # Get indices of source and target cells
        source_indices = self.adata.obs[self.adata.obs[self.group_by] == self.source].index
        target_indices = self.adata.obs[self.adata.obs[self.group_by].isin(self.target)].index

        # Get numeric indices of source and target cells
        source_numeric = self.adata.obs_names.get_indexer(source_indices)
        target_numeric = self.adata.obs_names.get_indexer(target_indices)

        # Subset the relative dimension matrix
        reld = self.adata.obsp[self.reld_key]
        reld_subset = reld[source_numeric.reshape(-1, 1), target_numeric]

        # Create a new AnnData object
        new_adata = anndata.AnnData(X=reld_subset)

        # Set .obs_names and .var_names
        new_adata.obs_names = source_indices
        new_adata.var_names = target_indices

        # Add group metadata
        new_adata.obs[self.group_by] = self.source  # All obs are source
        new_adata.var[self.group_by] = self.adata.obs.loc[target_indices, self.group_by]  # Corresponding groups for targets

        self.reld_adata = new_adata
        print(f"New AnnData for stat test created: {len(source_indices)} sources × {len(target_indices)} targets.")



    def _resolve_tie(self, na_value):
        """
        Resolve ties in predicted fates by performing pairwise Mann-Whitney U tests.
        If there are exactly two clusters in `predicted_fate`, decide the fate using the same criteria.
        Add a column 'resolve_tie' indicating if a tie was resolved.
        """

        if 'resolve_tie' not in self.reld_adata.obs:
            self.reld_adata.obs['resolve_tie'] = False

        for row_idx in tqdm(self.reld_adata.obs.index, desc='Resolving ties...'):
            row_fates = self.reld_adata.obs.loc[row_idx, 'predicted_fate']

            if len(row_fates) == 2:  # Only resolve ties with exactly 2 clusters
                cluster_a, cluster_b = row_fates

                # Get the group masks for the clusters
                mask_a = self.reld_adata.var[self.group_by] == cluster_a
                mask_b = self.reld_adata.var[self.group_by] == cluster_b

                # Extract the group values
                values_a = np.nan_to_num(self.reld_adata[row_idx, mask_a].X.toarray().flatten(), nan=na_value)
                values_b = np.nan_to_num(self.reld_adata[row_idx, mask_b].X.toarray().flatten(), nan=na_value)



                # Perform Mann-Whitney U test
                _, pvalue = mannwhitneyu(values_a.flatten(), values_b.flatten(), alternative='two-sided')

                # Check criteria to resolve the tie
                if pvalue < 0.05 and np.median(values_a) < np.median(values_b):
                    self.reld_adata.obs.at[row_idx, 'predicted_fate'] = [cluster_a]
                    self.reld_adata.obs.at[row_idx, 'resolve_tie'] = True
                elif pvalue < 0.05 and np.median(values_b) < np.median(values_a):
                    self.reld_adata.obs.at[row_idx, 'predicted_fate'] = [cluster_b]
                    self.reld_adata.obs.at[row_idx, 'resolve_tie'] = True
    
    
    def predict_fate(self, 
                     resolve_tie=True,
                     na_value: Union[float, str] = np.inf):

        """
        Perform the statistical test for each row of reld_adataa against the clusters in var.
        Adds p-values for each cluster as new columns in reld_adata.obs.
        """

        if not hasattr(self, 'reld_adata'):
            print("reld_adata not found. Automatically creating reld_adata for furter test.")
            self._create_anndata()

        # Ensure 'self.group_by' column exists in var
        if self.group_by not in self.reld_adata.var:
            raise ValueError(f'The {self.group_by} column must be present in reld_adata.var for the test.')
        
        # Assign NA
        if na_value == "max" :
            na_value = self.reld_adata.X.nanmax() + ((self.reld_adata.X.nanmax())-(self.reld_adata.X.nanmin()))*0.01

        # Get unique clusters
        clusters = self.reld_adata.var[self.group_by].unique()
        cluster_masks = {cluster: (self.reld_adata.var[self.group_by] == cluster).values for cluster in clusters}

        # Prepare to store p-values
        p_values = np.zeros((self.reld_adata.n_obs, len(clusters)))

        # Perform the Mann-Whitney U test for each row
        for cluster_idx, cluster in enumerate(clusters):
            group_mask = cluster_masks[cluster]
            rest_mask = ~group_mask

            # Extract group and rest columns in bulk
            group_values = self.reld_adata.X[:, group_mask]
            rest_values = self.reld_adata.X[:, rest_mask]

            # Perform the test row-wise
            for row_idx in range(self.reld_adata.n_obs):
                _, pvalue = mannwhitneyu(
                    np.nan_to_num(group_values[row_idx].toarray().flatten(), nan=np.inf) if sp.issparse(group_values) else np.nan_to_num(group_values[row_idx], nan=na_value),
                    np.nan_to_num(rest_values[row_idx].toarray().flatten(), nan=np.inf) if sp.issparse(rest_values) else np.nan_to_num(rest_values[row_idx], nan=na_value),
                    alternative='less'
                )
                p_values[row_idx, cluster_idx] = pvalue

        # Add p-values to reld_adata.obs
        for cluster_idx, cluster in enumerate(clusters):
            self.reld_adata.obs[f"{cluster}_p"] = p_values[:, cluster_idx]

        # Apply FDR correction for each group's p-values separately
        for cluster_idx, cluster in enumerate(clusters):
            raw_pvalues = p_values[:, cluster_idx]
            _, fdr_corrected_pvalues, _, _ = multipletests(raw_pvalues, method='fdr_bh')

            # Add FDR-corrected p-values to reld_adata.obs
            self.reld_adata.obs[f"{cluster}_adj_p"] = fdr_corrected_pvalues


        print("Primary tests completed and p-values added to reld_adata.obs.")


        # Predict fate based on criteria
        predicted_fate = []

        for row_idx in tqdm(self.reld_adata.obs.index, desc='Classification in progress...'):
            row_fates = []
            for cluster_idx, cluster in enumerate(clusters):
                group_mask = cluster_masks[cluster]

                group_values = self.reld_adata[row_idx, group_mask].X.toarray().flatten() if sp.issparse(group_values) else group_values.flatten()
                rest_values = self.reld_adata[row_idx, ~group_mask].X.toarray().flatten() if sp.issparse(rest_values) else rest_values.flatten()

                # Check criteria: adj p-value < 0.05 and median(cluster) < median(rest)
                if (self.reld_adata.obs.loc[row_idx, f"{cluster}_adj_p"] < 0.05 and
                    np.median(np.nan_to_num(group_values, nan=na_value)) <
                    np.median(np.nan_to_num(rest_values, nan=na_value))
                    ):
                    row_fates.append(cluster)

            predicted_fate.append(row_fates)

        self.reld_adata.obs['predicted_fate'] = predicted_fate

        print("FDR adjusted p-values added to reld_adata.obs")

        if resolve_tie and len(clusters) > 2 :
            self._resolve_tie(na_value)

        def list_to_string(lst) :
            if not lst :
                return 'None'
            else :
                return ' & '.join(lst)
            
        self.reld_adata.obs['predicted_fate'] = self.reld_adata.obs['predicted_fate'].apply(list_to_string)
            
        return None


