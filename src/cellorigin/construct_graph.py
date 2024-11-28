import numpy as np
import networkx as nx
import warnings
import scipy.sparse as sp
import scanpy as sc
from tqdm import tqdm
from scipy.sparse import coo_array



def calc_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result
    return wrapper




def _generate_graph_original(matrix):
    """
    Generate a graph using the "Original" method (ensure graph is connected).

    Parameters:
    - matrix: Sparse MIC matrix (CSR or CSC format)

    Returns:
    - graph: NetworkX graph
    - threshold: MIC threshold used
    """
    low, high = 0, 1

    while high - low > 1e-4:
        threshold = (low + high) / 2.0

        # Apply threshold to sparse matrix
        temp_matrix = matrix.copy()
        temp_matrix.data[temp_matrix.data < threshold] = 0
        temp_matrix.eliminate_zeros()
        
        # Create graph
        graph = nx.from_scipy_sparse_array(temp_matrix)

        # Check connectivity
        if nx.is_connected(graph):
            high = threshold
        else:
            low = threshold

    # Finalize threshold and apply it to the matrix
    threshold = low
    matrix.data[matrix.data < threshold] = 0
    matrix.eliminate_zeros()
    graph = nx.from_scipy_sparse_array(matrix)

    return graph, threshold





def _generate_graph_multiples(matrix: sp.spmatrix, 
                              target_edgenum: int):
    """
    Generate a graph to match a target number of edges using the "Multiples" method.

    Parameters:
    - matrix: Sparse MIC matrix (CSR or CSC format)
    - target_edgenum: Target number of edges

    Returns:
    - graph: NetworkX graph
    - threshold: MIC threshold used
    """
    low, high = 0, 1
    golden_ratio = (1 + 5 ** 0.5) / 2
    best_graph = None
    best_threshold = None
    best_diff = float('inf')

    with tqdm(total=float('inf'), desc='Constructing graph in progress...') as pbar:
        while high - low > 1e-5:
            # Golden section search points
            mid1 = high - (high - low) / golden_ratio
            mid2 = low + (high - low) / golden_ratio

            # Evaluate both points
            graph1, diff1 = _get_graph_and_diff(matrix, mid1, target_edgenum)
            graph2, diff2 = _get_graph_and_diff(matrix, mid2, target_edgenum)

            # Update best result
            if diff1 < best_diff:
                best_diff = diff1
                best_graph = graph1
                best_threshold = mid1

            if diff2 < best_diff:
                best_diff = diff2
                best_graph = graph2
                best_threshold = mid2

            # Narrow the search space
            if diff1 > diff2:
                low = mid1
            else:
                high = mid2

            pbar.update()

        pbar.close()

    return best_graph, best_threshold




def _get_graph_and_diff(matrix: sp.spmatrix, 
                        threshold: float, 
                        target_edgenum: int):
    """
    Apply a threshold to the MIC matrix, construct a graph, and compute the edge difference.

    Parameters:
    - matrix: Sparse MIC matrix (CSR or CSC format)
    - threshold: Threshold value to apply
    - target_edgenum: Target number of edges

    Returns:
    - graph: NetworkX graph generated from the thresholded matrix
    - diff: Absolute difference between the current and target number of edges
    """
    # Apply threshold to the sparse matrix
    temp_matrix = matrix.copy()
    temp_matrix.data[temp_matrix.data < threshold] = 0
    temp_matrix.eliminate_zeros()

    # Construct the graph
    graph = nx.from_scipy_sparse_array(temp_matrix)

    # Calculate the difference in edge count
    edge_num = graph.number_of_edges()
    diff = abs(edge_num - target_edgenum)

    return graph, diff




@calc_time
def generate_graph(adata,
                   mic_key, 
                   method="Original", 
                   number_of_multiples=None):
    """
    Generate a graph from a sparse triangular MIC matrix.

    Parameters:
    - mic_matrix: Sparse upper triangular MIC matrix (CSR format)
    - method: Method to generate the graph ("Original" or "Multiples")
    - number_of_multiples: Target edge count as a multiple of nodes (used only with "Multiples" method)

    Returns:
    - graph: Generated NetworkX graph
    - threshold: MIC threshold used to generate the graph
    """
    # Validate inputs
    method_list = ["Original", "Multiples"]
    
    if mic_key not in adata.obsp :
        raise ValueError(f"{mic_key} is not present in `adata.obsp`. Please check the AnnData object.")

    if method not in method_list:
        raise ValueError(f"Invalid method '{method}'. Method should be one of {method_list}.")

    # Convert sparse matrix to dense (upper triangular) for processing
    # Get MIC matrix
    if sp.issparse(adata.obsp[mic_key]) :
        matrix = adata.obsp[mic_key].copy()
    else :
        matrix = sp.csr_matrix(adata.obsp[mic_key].copy())

    # Symmetrize the matrix (sparse triangular to full)
    matrix = (matrix + matrix.T).tocsr()

    # Generate the graph based on the selected method
    if method == "Original":
        graph, threshold = _generate_graph_original(matrix)
    elif method == "Multiples":
        if number_of_multiples is None:
            raise ValueError("`number_of_multiples` must be specified for the 'Multiples' method.")
        target_edges = matrix.shape[0] * number_of_multiples
        print(f"Target number of edges: {target_edges}")
        graph, threshold = _generate_graph_multiples(matrix, target_edges)

        # Warn if the graph has disconnected components
        if nx.number_connected_components(graph) > 1:
            warnings.warn("The graph has more than one connected component.")

    # Print graph statistics
    print(f"MIC threshold: {threshold}")
    print(f"Number of nodes: {len(graph.nodes())}")
    print(f"Number of edges: {len(graph.edges())}")
    print(f"Is graph connected: {nx.is_connected(graph)}")
    print(f"Number of connected components: {nx.number_connected_components(graph)}")

    return graph, threshold