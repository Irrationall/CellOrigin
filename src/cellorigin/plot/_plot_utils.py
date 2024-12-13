from matplotlib.colors import to_rgb, to_hex



def darken_color(color, 
                 amount=0.2):
 
    rgb = to_rgb(color)  # Convert color to RGB tuple
    darker_rgb = [max(0, c - amount) for c in rgb]  # Reduce brightness
    
    return to_hex(darker_rgb)  # Convert back to hex
