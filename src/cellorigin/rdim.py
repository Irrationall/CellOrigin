"""Main functions for calculating relative dimensions in a graph."""

import os
import time
import warnings
from functools import partial, wraps
from multiprocessing import Pool
import networkx as nx
import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm




# Limit the number of threads used by numerical libraries
os.environ['OMP_NUM_THREADS'] = '1'       # For OpenMP (used by many libraries)
os.environ['MKL_NUM_THREADS'] = '1'       # For Intel MKL
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # For OpenBLAS
os.environ['NUMEXPR_NUM_THREADS'] = '1'   # For NumExpr



# Constants
PRECISION = 1e-8




def calc_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time for {func.__name__}: {end_time - start_time:.2f} seconds")
        return result
    return wrapper




def run_all_sources(
    graph, 
    times, 
    use_spectral_gap=True, 
    disable_tqdm=False, 
    batch_size=100, 
    n_workers=1
):
    """
    Compute relative dimensions of all nodes in a graph.

    Parameters:
    - graph: NetworkX graph
    - times: Array of time steps for diffusion
    - use_spectral_gap: Whether to normalize Laplacian by spectral gap
    - disable_tqdm: Disable progress bar
    - batch_size: Number of nodes to process in each batch
    - n_workers: Number of workers for parallel processing

    Returns:
    - relative_dimensions: Array of relative dimensions
    - peak_times: Array of peak times
    """

    # Precompute degree-related constants
    degrees = np.array([deg for _, deg in graph.degree(weight="weight")])
    total_degree = degrees.sum()
    len_graph = len(graph)
    constant = total_degree / len_graph

    # Prepare sources
    sources = np.zeros((len_graph, len_graph), dtype=np.float64)
    np.fill_diagonal(sources, constant / degrees)

    # Compute Laplacian and spectral gap
    laplacian, spectral_gap = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)

    # Initialize output arrays
    relative_dimensions = np.full((len_graph, len_graph), np.nan)
    peak_times = np.full((len_graph, len_graph), np.nan)

    # Divide tasks into batches
    batches = [
        (laplacian, sources[:, i:i + batch_size], times, spectral_gap, i)
        for i in range(0, len_graph, batch_size)
    ]

    # Parallel processing of batches
    results = Parallel(n_jobs=n_workers)(
        delayed(process_batch)(batch) for batch in tqdm(batches, disable=disable_tqdm)
    )

    # Collect results
    for rd_batch, pt_batch, start_idx in results:
        end_idx = start_idx + rd_batch.shape[1]
        relative_dimensions[:, start_idx:end_idx] = rd_batch
        peak_times[:, start_idx:end_idx] = pt_batch

    # Remove self-loops
    np.fill_diagonal(relative_dimensions, np.nan)
    np.fill_diagonal(peak_times, np.nan)

    print("All sources processed.")
    return relative_dimensions, peak_times




def process_batch(args):
    """
    Process a batch of sources.

    Parameters:
    - args: Tuple containing (laplacian, sources_batch, times, spectral_gap, start_idx)

    Returns:
    - rd_batch: Relative dimensions for the batch
    - pt_batch: Peak times for the batch
    - start_idx: Start index of the batch
    """
    laplacian, sources_batch, times, spectral_gap, start_idx = args
    trajectories = compute_node_trajectories(laplacian, sources_batch, times, disable_tqdm=True)
    rd_batch, pt_batch, _, _ = extract_relative_dimensions_vectorized(
        times, trajectories, sources_batch, spectral_gap
    )
    return rd_batch, pt_batch, start_idx




def construct_laplacian(graph, 
                        laplacian_type="normalized", 
                        use_spectral_gap=True):
    """
    Construct the graph Laplacian matrix.

    Parameters:
    - graph: NetworkX graph
    - laplacian_type: Type of Laplacian ("normalized" is currently supported)
    - use_spectral_gap: Whether to scale by the spectral gap

    Returns:
    - laplacian: Sparse Laplacian matrix
    - spectral_gap: Spectral gap of the Laplacian
    """
    if laplacian_type == "normalized":
        # Compute degree vector and normalized Laplacian
        degrees = np.array([graph.degree(i, weight="weight") for i in graph.nodes()])
        laplacian = sp.diags(1.0 / degrees).dot(nx.laplacian_matrix(graph))
    else:
        raise NotImplementedError("Only 'normalized' Laplacian is implemented.")

    spectral_gap = 1.0

    if use_spectral_gap:
        # Compute the spectral gap (second smallest eigenvalue)
        spectral_gap = abs(sp.linalg.eigs(laplacian, which="SM", k=2)[0][1])
        laplacian /= spectral_gap

    return laplacian, spectral_gap





def compute_node_trajectories(laplacian, 
                              initial_measure, 
                              times, 
                              disable_tqdm=False):
    """
    Compute diffusion trajectories for nodes.

    Parameters:
    - laplacian: Laplacian matrix
    - initial_measure: Initial measure for diffusion
    - times: List of time steps

    Returns:
    - node_trajectories: Array of trajectories over time
    """
    node_trajectories = [
        heat_kernel(laplacian, times[0], initial_measure),
    ]
    for i in range(1, len(times)):
        delta_t = times[i] - times[i - 1]
        node_trajectories.append(heat_kernel(laplacian, delta_t, node_trajectories[-1]))

    return np.array(node_trajectories)




def heat_kernel(laplacian, 
                timestep, 
                measure):
    """
    Compute the heat kernel for diffusion.

    Parameters:
    - laplacian: Laplacian matrix
    - timestep: Time step for diffusion
    - measure: Initial measure

    Returns:
    - Result of matrix exponential applied to the measure
    """
    return sp.linalg.expm_multiply(-timestep * laplacian, measure)




def extract_relative_dimensions_vectorized(times, 
                                           node_trajectories, 
                                           initial_measures, 
                                           spectral_gap):
    """
    Compute relative dimensions from node trajectories, vectorized over sources.

    Parameters:
    - times: List of time steps
    - node_trajectories: Diffusion trajectories (n_times, n_nodes, n_sources)
    - initial_measures: Initial measures for each node (n_nodes, n_sources)
    - spectral_gap: Spectral gap for normalization

    Returns:
    - relative_dimensions: Array of relative dimensions
    - peak_times: Array of peak times
    - peak_amplitudes: Array of peak amplitudes
    - diffusion_coefficient: Diffusion coefficient
    """
    n_nodes = node_trajectories.shape[1]
    stationary_prob = 1 / n_nodes  # Uniform stationary state

    # Find the peaks (amplitudes and positions)
    peak_amplitudes = np.max(node_trajectories, axis=0)  # Shape: (n_nodes, n_sources)
    peak_pos = np.argmax(node_trajectories, axis=0)      # Shape: (n_nodes, n_sources)

    # Convert peak positions into times
    peak_times = times[peak_pos]  # Shape: (n_nodes, n_sources)

    # Identify unreachable nodes
    unreachable = peak_amplitudes < stationary_prob + PRECISION

    # Identify missed peaks (peaks at the first time step)
    initial_measures_nonzero = initial_measures > 0
    missed_peaks = (peak_pos == 0) & (~initial_measures_nonzero)

    # Issue a warning if there are missed peaks
    if np.any(missed_peaks):
        warnings.warn(
            "Some peaks are not detected because they occur at the minimum time step. "
            "Consider reducing the minimum time step in `times` to capture these peaks."
        )

    # Mask out unreachable nodes and initial measures
    peak_times[unreachable] = np.nan
    peak_amplitudes[unreachable] = np.nan
    peak_times[initial_measures_nonzero] = np.nan
    peak_amplitudes[initial_measures_nonzero] = np.nan

    # Compute the relative dimensions
    diffusion_coefficient = 0.5 / spectral_gap
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_dimensions = (
            -2.0
            * np.log(peak_amplitudes)
            / (1.0 + np.log(peak_times) + np.log(4.0 * diffusion_coefficient * np.pi))
        )

    # Ensure no negative dimensions
    relative_dimensions[relative_dimensions < 0] = np.nan

    return relative_dimensions, peak_times, peak_amplitudes, diffusion_coefficient




def run_single_source(graph, 
                      times, 
                      source_node, 
                      use_spectral_gap=True):
    """
    Compute relative dimensions for a single source node in a graph.

    Parameters:
    - graph: NetworkX graph
    - times: List of time steps for diffusion
    - source_node: Node name in the graph to act as the source
    - use_spectral_gap: Whether to normalize Laplacian by spectral gap

    Returns:
    - results: Dictionary containing:
        - "relative_dimensions": Array of relative dimensions for all target nodes
        - "peak_amplitudes": Array of peak amplitudes
        - "peak_times": Array of peak times
        - "diffusion_coefficient": Diffusion coefficient used
        - "times": List of time steps
        - "node_trajectories": Array of diffusion trajectories
    """
    # Construct the Laplacian and compute the spectral gap
    laplacian, spectral_gap = construct_laplacian(graph, use_spectral_gap=use_spectral_gap)

    # Generate the initial measure for the source node
    initial_measure = get_initial_measure(graph, [source_node])

    # Compute diffusion trajectories for the given initial measure
    node_trajectories = compute_node_trajectories(laplacian, initial_measure, times)

    # Extract relative dimensions and related metrics
    relative_dimensions, peak_times, peak_amplitudes, diffusion_coefficient = extract_relative_dimensions_vectorized(
        times, node_trajectories, initial_measure[:, None], spectral_gap
    )

    # Package results into a dictionary
    results = {
        "relative_dimensions": relative_dimensions.flatten(),  # Array for all target nodes
        "peak_amplitudes": peak_amplitudes.flatten(),
        "peak_times": peak_times.flatten(),
        "diffusion_coefficient": diffusion_coefficient,
        "times": times,
        "node_trajectories": node_trajectories,
    }

    return results




def get_initial_measure(graph, 
                        nodes):
    """
    Create a measure with the correct mass distributed among the given nodes.

    Parameters:
    - graph: NetworkX graph
    - nodes: List of nodes to distribute the measure

    Returns:
    - measure: 1D numpy array of size n_nodes, representing the initial measure
    """
    # Initialize measure array
    measure = np.zeros(len(graph), dtype=np.float64)

    # Handle edge case: Empty graph or empty node list
    if len(graph) == 0:
        raise ValueError("The graph is empty. Cannot compute initial measure.")
    if not nodes:
        raise ValueError("The nodes list is empty. Provide at least one node.")

    # Compute total degree of the graph
    total_degree = sum(graph.degree(u, weight="weight") for u in graph)
    if total_degree == 0:
        raise ValueError("The graph has zero total degree. Check for disconnected nodes or weights.")

    # Compute measure for the specified nodes
    for node in nodes:
        degree = graph.degree(node, weight="weight")
        if degree == 0:
            raise ValueError(f"Node {node} has zero degree and cannot contribute to the measure.")
        measure[node] = total_degree / (len(graph) * degree * len(nodes))

    return measure




### RUN RDIM ###
@calc_time
def calculate_relative_dimension(graph: nx.Graph,
                                 time_arr: np.array,
                                 batch_size: int = 1,
                                num_processes : int = 1) :
    
    rdim = run_all_sources(graph, time_arr, batch_size=batch_size, n_workers=num_processes)[0]

    return rdim
