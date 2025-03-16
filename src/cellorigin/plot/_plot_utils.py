from matplotlib.colors import to_rgb, to_hex, LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np



def darken_color(color, 
                 amount=0.2):
 
    rgb = to_rgb(color)  # Convert color to RGB tuple
    darker_rgb = [max(0, c - amount) for c in rgb]  # Reduce brightness
    
    return to_hex(darker_rgb)  # Convert back to hex




def create_handles(unique_values, 
                   color_dict, 
                   marker='o'):
        
    return [
        Line2D(
            [0], [0], 
            marker=marker, 
            color='w', 
            label=val, 
            markersize=8, 
            markerfacecolor=color_dict[val], 
            markeredgecolor='black'
        ) 
        for val in unique_values
    ]




def make_cmap(values, 
              colormap=None, 
              custom_colors=None, 
    ):
    """
    Generate a color mapping for given values using either a predefined colormap
    or a set of custom colors with interpolation.

    Parameters:
    ----------
    values : list
        List of unique values to assign colors to.
    colormap : str or matplotlib.colors.Colormap, optional
        Name of a matplotlib colormap or a Colormap object.
    custom_colors : list, optional
        List of custom colors (as hex or named) to interpolate.
    n_colors : int, optional
        Number of colors to generate if using custom interpolation.

    Returns:
    -------
    color_dict : dict
        Dictionary mapping values to corresponding colors.
    """
    
    # Convert to NumPy array if not already
    if not isinstance(values, np.ndarray):
        values = np.array(values)

    # Check for duplicates
    if len(set(values)) != len(values):
        raise ValueError("There are duplicate entries in the `values` list. Ensure all values are unique.")


    num_values = len(values)

    # Case 1: Use a predefined Matplotlib colormap
    if colormap:
        cmap = plt.get_cmap(colormap)

        # If it's a categorical colormap, force interpolation
        if cmap.N <= 30:
            # Convert discrete colors into a continuous colormap
            cmap = LinearSegmentedColormap.from_list("custom_cmap", [cmap(i) for i in range(10)], N=num_values)
        
        colors = [to_hex(cmap(i / (num_values - 1))) for i in range(num_values)]


    # Case 2: Use custom colors with interpolation
    elif custom_colors :
        if len(custom_colors) < 2:
            raise ValueError("At least two custom colors are required for interpolation.")

        # Convert color names/hex to RGB
        rgb_colors = np.array([to_rgb(c) for c in custom_colors])

        # Create interpolation function
        x = np.linspace(0, 1, len(rgb_colors))  # Points corresponding to given colors
        interpolator = interp1d(x, rgb_colors, axis=0)

        # Generate interpolated colors
        new_x = np.linspace(0, 1, num_values)
        interpolated_colors = interpolator(new_x)

        # Convert interpolated RGB values back to hex
        colors = [to_hex(c) for c in interpolated_colors]

    # Case 3: Use default Matplotlib color cycle
    else:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = default_colors[:num_values]  # Take only the needed amount

    # Create dictionary mapping values to colors
    color_dict = {val: col for val, col in zip(values, colors)}
    
    return color_dict
