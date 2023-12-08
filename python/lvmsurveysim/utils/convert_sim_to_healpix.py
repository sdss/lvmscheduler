# """Convert Survey Simulation to healpix Array of total coverage. Include secondary fit's file identifying the target with the highest priority in each spaxel"""
import astropy.io.fits as fits
from astropy_healpix import HEALPix
import healpy
from healpy.newvisufunc import projview
import numpy as np
import sys
import astropy
from astropy.coordinates import SkyCoord
from astropy.coordinates import Galactic
import astropy.table
import astropy.units as u
from lvmsurveysim.target import TargetList
import time

def convert(coversion_params):
    print_counter = 1 # time to print status in seconds
    for coversion_params_key in coversion_params.keys():
        print(coversion_params_key,":", coversion_params[coversion_params_key])

    image_hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    image_order = image_hdu_list[0].header['ORDERING']
    hp = HEALPix(nside=coversion_params['nside'], order=image_order, frame=Galactic())

    schedule = astropy.table.Table.read(coversion_params["file"])

    print(coversion_params["target_file"])
    targets = TargetList(target_file=coversion_params["target_file"])

    #Create the missing column in the schedule table: Priority
    schedule_priority = np.full(len(schedule['target']), -1)

    for target_i in range(len(targets)):
        target = targets[target_i].name
        if target != '-':
            target_mask = schedule['target'] == target
            schedule_priority[target_mask] = targets[target_i].priority
    
    schedule['priority'] = astropy.table.Column(schedule_priority)

    ### DEV
    #Mask out all the unobserved values. I don't know why they are bad, but they are.
    # obs_mask = schedule['target'] != "-"
    # Create a mapping between the values in the table and their healpix index.
    # This allows us to directly dump information from the table onto the array.
    # healpix_indicies = hp.skycoord_to_healpix(SkyCoord(schedule['ra'][obs_mask], schedule['dec'][obs_mask], unit=u.deg))
    ### DEV

    # This is how we will store all the different healpix arrays containing different information.
    healpix_dictionary = {}

    # To later create empty arrays we need to know the number of healpix pixels.
    npix=12*coversion_params['nside']**2

    #Populate the healpix array with the highest priority of that pixel.
    healpix_dictionary['target index'] = {}

    healpix_dictionary['priorities'] =  np.full(npix, -1)
    healpix_dictionary['priority_levels'] = np.sort(np.unique(schedule['priority']))
    # iterate over target priorities from lowest to highest
    for priority_level in healpix_dictionary['priority_levels']:
        t0 = time.time()
        priority_mask = (schedule['priority'] == priority_level) * (schedule['target'] != "-")
        #Note because of repeat visits the need to find unique values for the healpix arrays
        tmp0= len(schedule['ra'][priority_mask])
        tmp = tmp0
        print("processing priority level: %i"%(priority_level))
        for ra, dec in zip(schedule['ra'][priority_mask], schedule['dec'][priority_mask]):
            # find healpix pixels that cover ra,dec of target tiles
            heal_indices = hp.cone_search_skycoord(SkyCoord(ra, dec, unit=u.deg), radius=(0.25)*u.deg)
            # assign the target priority to these healpix pixels
            healpix_dictionary['priorities'][heal_indices] = priority_level
            complete = 1.- float(tmp)/float(tmp0)
            tmp = tmp -1
            
            if time.time() - t0 > print_counter:
                t0 = time.time()
                print("Progress {:2.1%}".format(complete), end="\r")
    
    return(healpix_dictionary)


def healpix_shader(data,
                    masks, 
                    outfile="tmp.png",
                    nest=True,
                    save=True,
                    gui=False,
                    graticule=True,
                    healpixMask=False, 
                    vmin=False, vmax=False, 
                    norm=1.0, pad=0.1,
                    scale=False, 
                    cmaps=["Blues", "Greens", "Reds"],
                    background='w',
                    title="",
                    show_colorbar=True,
                    plt=False):
    """Healpix shader.

    Parameters
    ----------
    data : numpy.ndarray
        data array of representing a full healpix
    masks : numpy.ma
        Masks of teh data array, which define which pixel gets which color
    nest : bol
        Is the numpy array nested, or ringed
    cmaps : matplotlib.cmaps
        Any maptplotlib cmap, called by name. The order must be the same order as the masks.
    Returns
    -------
    matplotlib.plt plot : 
        It returns a plot....
    """

    import healpy
    import numpy as np
    if plt == False:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import gc

    if gui == False:
        mpl.use('Agg')

    color_list = []
    for mask_i in range(len(masks)):
        if mask_i <= len(cmaps) -1:
            cmap = cmaps[mask_i]
        else:
            #Use the last cmap
            cmap = cmaps[-1]
        if type(cmap) == str:
            color_list.append(plt.get_cmap(cmap)(np.linspace(0.,1,128)))
        else:
            color_list.append(cmap(np.linspace(0.,1,128)))

    colors = np.vstack(color_list)
    cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    cmap.set_under(background)

    # Copy the data to a new array to be cropped and scaled
    I = data.copy()

    # Crop data at max an min values if provided.
    # Otherwise set to range of data
    if vmax==False:
        vmax = np.max(I)
    if vmin==False:
        vmin = np.min(I)

    # Crop data
    I[I > vmax] = vmax
    I[I < vmin] = vmin

    # Rescale all data from 0+pad to (normalization-pad)
    I = (I - vmin) * ( (norm - pad) - pad) / (vmax - vmin) + pad

    normalized_I = I #np.full(len(I), -1.6375e+30) # start with MW Halpha background

    # add the offset to the data to push it into to each color range
    # 0-1 is the 0-th background image, 1-2 the first target, etc ...
    if scale is not False:
        for i in range(len(masks)):
            normalized_I[masks[i]] = I[masks[i]].copy()*scale[i] + min(i, len(cmaps)-1) * norm
    else:
        for i in range(len(masks)):
            normalized_I[masks[i]] = I[masks[i]].copy() + min(i, len(cmaps)-1) * norm

    # deal with the lowest priority separately, the loop above misses it
    normalized_I[masks[0]] = 1.75

    # If there is a healpix mask apply it.
    normalized_I_masked = healpy.ma(normalized_I)

    if healpixMask != False:
        normalized_I_masked.mask = np.logical_not(healpixMask)

    #healpy.mollview(normalized_I_masked, nest=nest, cbar=show_colorbar, cmap=cmap, rot=(0,0,0), min=0, max=norm*len(masks), xsize=4000, title=title)
    projview(normalized_I_masked, projection_type='mollweide', 
             nest=nest, cbar=show_colorbar, cmap=cmap, rot=(0,0,0),
             min=0, max=norm*len(masks), xsize=4000, title=title,
             longitude_grid_spacing=30, latitude_grid_spacing=30,
             graticule=graticule, graticule_labels=graticule, xlabel='lon [deg]', ylabel='lat [deg]')
    if save == True:
        plt.savefig(outfile)
    if gui==True:
        plt.show()
    plt.clf()
    plt.close()
    gc.collect()

# can be run as 
# import lvmsurveysim.utils.convert_sim_to_healpix as h
# run({"file": "LCO_2023_5.fits", "target_file": "targets.yaml", "nside": 1024})
def run(params):
    healpix_dictionary = convert(params)

    image_hdu_list = fits.open("Halpha_fwhm06_1024.fits")
    image_nside = image_hdu_list[0].header['NSIDE']
    image_order = image_hdu_list[0].header['ORDERING']

    colors =[]
    masks = []
    colors_available = ["copper", "Greens","Blues", "Purples"]
    scale = [1, 1, 1]

    priority_min = -1

    healpix_dictionary["high_res_priorities"] = healpy.pixelfunc.ud_grade(healpix_dictionary['priorities'], image_nside, power=0.0)
    for priority_i, priority in enumerate(healpix_dictionary['priority_levels']):
        if priority >= (priority_min or -1):
            masks.append(healpix_dictionary['high_res_priorities'] == priority)
            if priority_i <= len(colors_available) -1:
                colors.append(colors_available[priority_i])
            else:
                colors.append(colors_available[-1])
            if priority_i <= len(scale) -1:
                scale.append(scale[-1])
            
    
    data = np.array(image_hdu_list[1].data.tolist())[:,0]
    log_I = np.log10(data)
    log_I_max = 2.0
    log_I_min = -1.0

    healpix_shader(log_I, masks, cmaps=colors, scale=scale, title=None, nest=True, show_colorbar=False,
                   vmin=log_I_min, vmax=log_I_max, outfile="%s_shaded_MW.png"%(params['file'].replace(".fits","")), gui=True)

# convert_sim_to_healpix file:LCO_2023_5.fits target_file:targets.yaml nside:1024
if __name__ == "__main__":
    "provide the schedule fits file, and target list"
    # params = {"file":None, "target_file":"None", "nside":1024}
    params = {"file": "baseline_beta-0.fits", "target_file": "targets.yaml", "nside": 1024}

    if len(sys.argv) > 1:
        for argument in sys.argv[1:]:
            key, value = argument.split(":")
            params[key] = value

    if params["target_file"] == "None":
        params["target_file"] = './targets.yaml'
        print("converting fits file name %s to target file %s"%(params["file"], params["target_file"]))

    run(params)

    