import matplotlib.colors as colors
import numpy as np
class cmaps():
    
    def __init__(self):
        
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        
        # simrad EK500 default colormap
        ek500 = [(1.0000, 1.0000, 1.0000), # white
                 (0.6235, 0.6235, 0.6235), # light grey
                 (0.3725, 0.3725, 0.3725), # dark grey
                 (0.0000, 0.0000, 1.0000), # dark blue
                 (0.0000, 0.0000, 0.5000), # light blue
                 (0.0000, 0.7490, 0.0000), # light green
                 (0.0000, 0.5000, 0.0000), # dark green
                 (1.0000, 1.0000, 0.0000), # yellow
                 (1.0000, 0.5000, 0.0000), # orange
                 (1.0000, 0.0000, 0.7490), # pink
                 (1.0000, 0.0000, 0.0000), # red
                 (0.6509, 0.3255, 0.2353), # light brown
                 (0.4705, 0.2353, 0.1568)] # dark brown
        self.ek500 = colors.LinearSegmentedColormap.from_list('ek500', ek500)
        self.ek500.set_bad(color='k', alpha=1)
        
        # coolwarm colormap with nan values set to black
        self.coolwarm = cm.coolwarm
        self.coolwarm.set_bad(color='k', alpha=1)
        
        # viridis colormap with nan values set to black
        self.viridis = cm.viridis
        self.viridis.set_bad(color='k', alpha=1)
        
class ClassColor():
    def __init__(self):
        
        import matplotlib.cm as cm
        import matplotlib.colors as colors
        rgb = [
            (1.0000, 1.0000, 1.0000),
            (1.0000, 1.0000, 0.0000), # yellow-1 
            (1.0000, 0.0000, 0.0000), # red-2 
            (0.0000, 0.0000, 1.0000), # dark blue-3 
            ]
        self.rgb = colors.LinearSegmentedColormap.from_list('rgb', rgb)
        # self.ek500.set_bad(color='k', alpha=1)


def create_custom_cmap(colors):
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt
    cmap = ListedColormap(colors)
    return cmap
