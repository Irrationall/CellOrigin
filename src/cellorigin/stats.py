from anndata import AnnData
import pandas as pd
import scipy.sparse as sp
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import numpy as np
from typing import List, Union

class FatePredictor:
    def __init__(self,
                 adata: AnnData,
                 reld_key: str,
                 group_by: str,
                 source: str,
                 target: List[str],
                 copy: bool = False):
        """
        Initializes the FatePredictor class.

        Parameters:
        - adata: Input AnnData object.
        - reld_key: Key for the relative dimension matrix in obsp. This matrix represents the relative dimensions between cells.
        - group_by: Column in obs specifying cluster annotations.
        - source: Name of the source cluster.
        - target: List of target cluster names.
        - copy: Whether to copy the input AnnData object.
        """
        self.adata = adata.copy() if copy else adata
        self.reld_key = reld_key
        self.group_by = group_by
        self.source = source
        self.target = target

    def _validate_inputs(self):

        """Validates the input parameters."""

        if self.reld_key not in self.adata.obsp:
            raise ValueError(
                f"'{self.reld_key}' not found in `adata.obsp`. Please check your AnnData object."
            )

        if self.group_by not in self.adata.obs:
            raise ValueError(
                f"'{self.group_by}' not found in `adata.obs`. Please check the column name."
            )

        if self.source not in self.adata.obs[self.group_by].unique():
            raise ValueError(
                f"Source cluster '{self.source}' not found in the '{self.group_by}' column."
            )

        if not all(
            target_cluster in self.adata.obs[self.group_by].unique()
            for target_cluster in self.target
        ):
            raise ValueError("One or more target clusters are not found in the dataset.")

        if len(self.target) < 2:
            raise ValueError("Please specify at least two target clusters for comparison.")

    def _create_anndata(self):
        """
        Creates a new AnnData object by subsetting the input adata based on source and target indices.
        """
        # Validate inputs
        self._validate_inputs()

        # Get indices of source and target cells
        source_indices = self.adata.obs[
            self.adata.obs[self.group_by] == self.source
        ].index
        target_indices = self.adata.obs[
            self.adata.obs[self.group_by].isin(self.target)
        ].index

        # Get numeric indices of source and target cells
        source_numeric = self.adata.obs_names.get_indexer(source_indices)
        target_numeric = self.adata.obs_names.get_indexer(target_indices)

        # Subset the relative dimension matrix (use as is if input is already a sparse matrix)
        reld = self.adata.obsp[self.reld_key]
        reld_subset = reld[source_numeric.reshape(-1, 1), target_numeric]

        # Create a new AnnData object
        new_adata = AnnData(X=reld_subset)

        # Set .obs_names and .var_names
        new_adata.obs_names = source_indices
        new_adata.var_names = target_indices

        # Add group metadata
        new_adata.obs[self.group_by] = self.source  # All obs are from the source
        new_adata.var[self.group_by] = self.adata.obs.loc[
            target_indices, self.group_by
        ]  # Groups corresponding to targets

        self.reld_adata = new_adata
        print(
            f"New AnnData created for statistical testing: {len(source_indices)} source × {len(target_indices)} target."
        )

    def _resolve_tie(self, na_value):
        """
        Resolves ties in predicted fate by performing pairwise Mann-Whitney U tests.
        If there are exactly two clusters in `predicted_fate`, it determines the fate based on the same criteria.
        Adds a 'resolve_tie' column indicating whether a tie was resolved.
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
    
                # Extract the group values, treating 0 as NaN and replacing with na_value
                values_a = self.reld_adata[row_idx, mask_a].X.toarray().flatten()
                values_b = self.reld_adata[row_idx, mask_b].X.toarray().flatten()
    
                # Replace 0 with a large value, then replace that value with na_value
                values_a[values_a == 0] = na_value
                values_b[values_b == 0] = na_value
    
                # Perform Mann-Whitney U test
                _, pvalue = mannwhitneyu(values_a, values_b, alternative='two-sided')
    
                # Check criteria to resolve the tie
                if pvalue < 0.05 and np.median(values_a) < np.median(values_b):
                    self.reld_adata.obs.at[row_idx, 'predicted_fate'] = [cluster_a]
                    self.reld_adata.obs.at[row_idx, 'resolve_tie'] = True
                elif pvalue < 0.05 and np.median(values_b) < np.median(values_a):
                    self.reld_adata.obs.at[row_idx, 'predicted_fate'] = [cluster_b]
                    self.reld_adata.obs.at[row_idx, 'resolve_tie'] = True

    def predict_fate(self,
                     resolve_tie: bool = True,
                     na_value: Union[float, str] = "max",
                     use_fdr: bool = True):

        """
        Performs statistical tests on the clusters in var for each row in reld_adata.
        Adds p-values and adjusted p-values for each cluster as new columns in reld_adata.obs.
        Also adds a 'predicted_fate' column based on the specified criteria.

        Parameters:
        - resolve_tie: Whether to resolve ties in the predicted fate using Mann-Whitney U test between tied clusters.
        - na_value: Value to replace 0s with during the test. If "max", replaces 0s with a value slightly larger than the maximum non-zero value.
                     It's crucial to replace zeros with non-zero values because zero values may not accurately represent the absence of a biological relationship.
                     Instead, they often represent data that couldn't be captured due to technical limitations, potentially leading to false positives or negatives in statistical tests.
        - use_fdr: Whether to use FDR-adjusted p-values for determining the predicted fate.
        """

        if not hasattr(self, "reld_adata"):
            print(
                "reld_adata not found. Automatically creating reld_adata for further testing."
            )
            self._create_anndata()

        # Check if 'self.group_by' column exists in var
        if self.group_by not in self.reld_adata.var:
            raise ValueError(
                f"'{self.group_by}' column must be in reld_adata.var for testing."
            )

        # Set na_value (for replacing 0s with a value larger than the maximum)
        if na_value == "max":
            # Find the maximum of non-zero values
            non_zero_max = self.reld_adata.X.data.max() if sp.issparse(self.reld_adata.X) and np.any(self.reld_adata.X.data != 0) else np.max(self.reld_adata.X[self.reld_adata.X != 0], initial=0)
            non_zero_min = self.reld_adata.X.data.min() if sp.issparse(self.reld_adata.X) and np.any(self.reld_adata.X.data != 0) else np.min(self.reld_adata.X[self.reld_adata.X != 0], initial=0)

            # Set na_value if the maximum is greater than 0 (i.e., there are non-zero values)
            if non_zero_max > 0:
                na_value = non_zero_max + (non_zero_max - non_zero_min) * 0.01
            else:
                na_value = 1  # If all values are 0 or negative, set an arbitrary positive value

            print(f"na_value is set to {na_value}")

        # Get unique clusters
        clusters = self.reld_adata.var[self.group_by].unique()
        cluster_masks = {
            cluster: (self.reld_adata.var[self.group_by] == cluster).values
            for cluster in clusters
        }

        # Prepare to store p-values
        p_values = np.zeros((self.reld_adata.n_obs, len(clusters)))

        # Perform Mann-Whitney U test for each row
        for cluster_idx, cluster in enumerate(clusters):
            group_mask = cluster_masks[cluster]
            rest_mask = ~group_mask

            # Extract group and rest columns in bulk (handles sparse matrices)
            group_values = self.reld_adata.X[:, group_mask].toarray()
            rest_values = self.reld_adata.X[:, ~group_mask].toarray()

            # Perform test row-wise (treating 0s as na_value)
            for row_idx in range(self.reld_adata.n_obs):
                group_row = group_values[row_idx].flatten()
                rest_row = rest_values[row_idx].flatten()

                # Replace large_value (originally 0s) with na_value
                group_row[group_row == 0] = na_value
                rest_row[rest_row == 0] = na_value

                _, pvalue = mannwhitneyu(
                    group_row, rest_row, alternative="less"
                )
                p_values[row_idx, cluster_idx] = pvalue

        # Add p-values to reld_adata.obs
        for cluster_idx, cluster in enumerate(clusters):
            self.reld_adata.obs[f"{cluster}_p"] = p_values[:, cluster_idx]

        print("Initial testing is complete, and p-values are added to reld_adata.obs.")

        # Apply FDR correction separately for p-values of each group
        for cluster_idx, cluster in enumerate(clusters):
            raw_pvalues = p_values[:, cluster_idx]
            _, fdr_corrected_pvalues, _, _ = multipletests(
                raw_pvalues, method="fdr_bh"
            )

            # Add FDR-corrected p-values to reld_adata.obs
            self.reld_adata.obs[f"{cluster}_adj_p"] = fdr_corrected_pvalues

        print("FDR-adjusted p-values are added to reld_adata.obs.")

        # Predict fate based on criteria
        predicted_fate = []

        FDR_OR_NOT = "adjusted p-values" if use_fdr else "p-values"
        print(f"{FDR_OR_NOT} are used for classification.")

        for row_idx in tqdm(self.reld_adata.obs.index, desc="Classifying..."):
            row_fates = []

            for cluster_idx, cluster in enumerate(clusters):
                group_mask = cluster_masks[cluster]

                # Extract group and rest values from sparse matrix (treating 0s as na_value)
                group_values = self.reld_adata[row_idx, group_mask].X.toarray().flatten()
                rest_values = self.reld_adata[row_idx, ~group_mask].X.toarray().flatten()

                # Replace large_value (originally 0s) with na_value
                group_values[group_values == 0] = na_value
                rest_values[rest_values == 0] = na_value

                

                p_value_key = f"{cluster}_adj_p" if use_fdr else f"{cluster}_p"

                # Check criteria: (adjusted: optional) p-value < 0.05 and median(cluster) < median(rest)
                if (
                    self.reld_adata.obs.loc[row_idx, p_value_key] < 0.05
                    and np.median(group_values) < np.median(rest_values)
                ):
                    row_fates.append(cluster)

            predicted_fate.append(row_fates)

        self.reld_adata.obs["predicted_fate"] = predicted_fate

        if resolve_tie and len(clusters) > 2:
            self._resolve_tie(na_value)

        def list_to_string(lst):
            if not lst:
                return "None"
            else:
                return " & ".join(lst)

        self.reld_adata.obs["predicted_fate"] = self.reld_adata.obs[
            "predicted_fate"
        ].apply(list_to_string)

        return None
