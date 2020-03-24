from matplotlib.colors import LinearSegmentedColormap

def get_blue_white_red_cmap(white_region = 0.1):
    region_2 = (0.5 - white_region)/2.
    region_3 = 0.5 - white_region/2.
    region_4 = 0.5 + white_region/2.
    region_5 = 1. - (0.5 - white_region)/2.
    
    cdict = {'red':  ((0.0, 0.0, 0.0),
                       (region_2, 0.0, 0.0),
                       (region_3, 1.0, 1.0),
                       (region_4, 1.0, 1.0),
                       (region_5, 1.0, 1.0),
                       (1.0, 0.5, 0.0)),
    
             'green': ((0.0, 0.0, 0.0),
                       (region_2, 0.0, 0.0),
                       (region_3, 1.0, 1.0),
                       (region_4, 1.0, 1.0),
                       (region_5, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 0.0, 0.5),
                       (region_2, 1.0, 1.0),
                       (region_3, 1.0, 1.0),
                       (region_4, 1.0, 1.0),
                       (region_5, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
            }

    return LinearSegmentedColormap('BlueRed', cdict)


def get_blue_red_cmap():

    cdict = {'red':  ((0.0, 0.0, 0.0),
                      (1.0, 1.0, 1.0)),
    
             'green': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),
    
             'blue':  ((0.0, 1.0, 1.0),
                       (1.0, 0.0, 0.0))
            }

    return LinearSegmentedColormap('BlueRed', cdict)