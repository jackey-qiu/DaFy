import id31_tools as id31
import constants as const
from colormaps import get_blue_white_red_cmap
import matplotlib.pyplot as plt

class coords_formatter:
    def __init__(self, img):
        self.img = img
    def format_coord(self, x, y):
        numrows, numcols = self.img.shape
        col = int(x+0.5)
        row = int(y+0.5)
        if col>=0 and col<numcols and row>=0 and row<numrows:
            z = self.img[row,col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f'%(x, y)

def get_th_del_gam(E_keV, H, K, L, a_star, c_star, alpha=0):
    '''
    Returns theta, delta and gamma for a given H, K, L position in a hexagonal unit cell.
    This assumes, that the tilt angle around the horizontal axis is set to zero.
    For alpha = 0 the H axis points to the right side of the detector along the horizontal axis.
    '''
    k_0 = id31.get_K_0(E_keV)
    return id31.get_angles(k_0, H, K, L, a_star, c_star, alpha)
    
def create_diff_img(img, ref_img, vmin=None, vmax=None, white_region=0.1):
    img = img - ref_img
    the_cmap = get_blue_white_red_cmap(white_region)
    plt.figure()
    ax = plt.subplot(111)
    im = ax.imshow(img, interpolation='None', vmin=vmin, vmax=vmax, cmap=the_cmap)
    cf = coords_formatter(img)
    ax.format_coord = cf.format_coord
    ax.set_xlim([0, 2048])
    ax.set_ylim([2048, 0])
    return ax

def get_img_filename(scan_no, img_no):
    return img_folder+'ihch1022_Au_Si_1_PE_'+str(scan_no).zfill(4)+'_'+str(img_no).zfill(4)+'.edf.gz'

    
if __name__ == '__main__':
    
    # set the energy
    E_keV = 40
    img_folder = '/home/finn/data/2015_11_IHCH1022/img/PE/'

    # create a difference image
    img = id31.PE_Image(get_img_filename(220, 0)).img
    ref_img = id31.PE_Image(get_img_filename(219, 0)).img
    ax = create_diff_img(img, ref_img, vmin=-1000, vmax=1000, white_region=0.1)
    id31.draw_heaxagonal_grid_2(ax, id31.get_K_0(E_keV), theta=0, alpha=0, d=745, a_star=const.a_star_Au, c_star=const.c_star_Au, color='yellow', linewidth=1)
    id31.draw_heaxagonal_grid_2(ax, id31.get_K_0(E_keV), theta=0, alpha=0, d=745, a_star=const.a_star_Co, c_star=const.c_star_Co, color='g', linewidth=1)
    plt.show()

    
    # calculate the angles for a given H,K,L position
    print get_th_del_gam(E_keV, 1, 1, 0, const.a_star_Au, const.c_star_Au)
    
    
    