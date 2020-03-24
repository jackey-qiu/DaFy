import numpy as np
import matplotlib.pyplot as pyplot

def color_combine_mate(color_number=8,combine_type=0):
    colors={3:[["#7fc97f","#beaed4","#fdc086"],["#1b9e77","#d95f02","#7570b3"],['#a6cee3','#1f78b4','#b2df8a'],['#fbb4ae','#b3cde3','#ccebc5'],['#b3e2cd','#fdcdac','#cbd5e8'],['#e41a1c','#377eb8','#4daf4a'],['#66c2a5','#fc8d62','#8da0cb'],['#8dd3c7','#ffffb3','#bebada']],\
            4:[['#7fc97f','#beaed4','#fdc086','#ffff99'],['#1b9e77','#d95f02','#7570b3','#e7298a'],['#a6cee3','#1f78b4','#b2df8a','#33a02c'],['#fbb4ae','#b3cde3','#ccebc5','#decbe4'],['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4'],['#e41a1c','#377eb8','#4daf4a','#984ea3'],['#66c2a5','#fc8d62','#8da0cb','#e78ac3'],['#8dd3c7','#ffffb3','#bebada','#fb8072']],\
            5:[['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0'],['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e'],['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99'],['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6'],['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9'],['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00'],['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854'],['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3']],\
            6:[['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f'],['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02'],['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c'],['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc'],['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae'],['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33'],['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f'],['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462']],\
            7:[['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17'],['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d'],['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f'],['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd'],['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc'],['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628'],['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494'],['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69']],\
            8:[['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666'],['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666'],['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00'],['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec'],['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc','#cccccc'],['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf'],['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3'],['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5']]}
    return colors[color_number][combine_type]        
            
def show_effect():
    for i in range(3,9):
        fig=pyplot.figure()
        for j in range(8):
            colors=color_combine_mate(i,j)
            for k in range(len(colors)):
                pyplot.plot([0,1],np.array([0.1,0.1])*k+j*2,color=colors[k])
            pyplot.title("case of "+str(i)+" colors")
    pyplot.show()
