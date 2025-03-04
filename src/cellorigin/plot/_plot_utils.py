from matplotlib.colors import to_rgb, to_hex
from matplotlib.lines import Line2D



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
