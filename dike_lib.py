# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:46:30 2020

@author: kbefus
"""

import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
from scipy.stats import spearmanr,pearsonr

import pandas as pd
import rasterio
from rasterio.mask import mask

#%%
# -------------- Raster functions --------------------
def xy_from_affine(tform=None,nx=None,ny=None):
    '''
    Construct position matrixes from affine.

    Parameters
    ----------
    tform : 
        Affine transform. The default is None.
    nx : int
        Number of columns. The default is None.
    ny : int
        Number of rows. The default is None.

    Returns
    -------
    X : np.ndarray
        Matrix of x positions of cell centers.
    Y : np.ndarray
        Matric of y positions of cell centers.

    '''
    X,Y = np.meshgrid(np.arange(nx)+0.5,np.arange(ny)+0.5)*tform
    return X,Y

def load_dem(demfname,shapes):
    '''
    Load DEM and spatial profile.

    Parameters
    ----------
    demfname : path
        Full path to DEM.
    shapes : 
        Feature class loaded in form required by rasterio.mask.mask.

    Returns
    -------
    DEM : np.ndarray
        Array of masked DEM values cropped to shapes.
    out_profile : dict
        Spatial dataset rasterio profile.

    '''
    with rasterio.open(demfname) as src:
        out_image, out_transform = mask(src, shapes, crop=True)
        out_profile = src.profile
    
    out_profile.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    out_image[out_image==out_profile['nodata']] = np.nan
    return out_image[0], out_profile

def stage_storage_dem(dem_fname=None,dem_array=None,
                      dem_profile=None, area_poly=None,
                      stage_range=None, nstages=50):
    '''
    Construct stage storage relationship from DEM.

    Parameters
    ----------
    dem_fname : str, optional
        Path to DEM raster. The default is None.
    dem_array : np.ndarray, optional
        DEM array already loaded into memory. The default is None.
    dem_profile : dict, optional
        Spatial information in rasterio profile form. The default is None.
    area_poly : fiona polygon, optional
        Polygon of area in DEM to consider for stage-storage relationship. The default is None.
    stage_range : np.ndarray,list, optional
        Minimum and maximum stages to calculate stages. Form [min, max]. The default is None.
    nstages : int, optional
        Number of stages to calculate across stage_range. The default is 50.

    Returns
    -------
    stage_storage_dict : dict
        Dictionary with keys 'stage','storage' and 'area', each containing a list
        of values the same length as the others.

    '''
    stage_storage_dict = {'stage':np.linspace(*stage_range,nstages)}    
    
    if dem_array is None:
        # Load DEM for area defined by area_poly feature
        dem_array,dem_profile = load_dem(dem_fname,area_poly)
    
    # Area of cell using cell dimensions    
    cell_area = dem_profile['transform'][0] * (-dem_profile['transform'][4]) # assumes no rotated grid
        
    storage = []
    area = []
    for istage in stage_storage_dict['stage']:
        stage_mask = dem_array<=istage
        area.append(cell_area*np.count_nonzero(stage_mask))
        storage.append(np.sum(cell_area*(istage-dem_array[stage_mask])))
    
    stage_storage_dict.update({'storage':storage,
                               'area':area})
    
    return stage_storage_dict

# -------------- End Raster functions --------------------

# ----------------- Discharge functions ---------------------------

def sluice_flow(water_elev_1=None,water_elev_2=None,flood_orientation=1,
                in_dict=None,return_components=False,
                grav=9.80665, ebb_free_headloss=True):
    '''
    Calculate discharge through sluice

    Parameters
    ----------
    water_elev_1 : TYPE, optional
        Water elevation at first location in meters (default to match gravity, can be any units if supplying gravity).
    water_elev_2 : float, optional
        Water elevation at second location in meters (default to match gravity, can be any units if supplying gravity). The default is None.
    flood_orientation: int, optional
        The default is 1, where water_elev_1 > water_elev_2 is the flood condition (i.e., water_elev_1 is on the marine side of a dike)
    in_dict : dict, optional
        Dictionary with flow coefficients and sluice geometry. The default is None.
    return_components : bool, optional
        Flag to return individual discharge components (True) or only return total (or combined) discharge. The default is False.
    grav : float, optional
        Gravitational acceleration, m**2/s. The default is 9.80665.

    Returns
    -------
    Q_dike_sluice_calc_ebb : float
        Total (or combined) discharge with units of m**3/s by default.
    Qdict : dict
        Dictionary with individual flow components from each condition. Only output if return_components=True
    '''
    
    water_depth_1 = water_elev_1 - in_dict['geom']['invert_elev'] # water depth above invert, marineside by default
    water_depth_2 = water_elev_2 - in_dict['geom']['invert_elev'] # water depth above invert, landside by default
    A_sluice_open = in_dict['geom']['open_height']*in_dict['geom']['open_width'] # area of sluice opening (for submerged flow comp)
    
    # Assign flow direction for accessing flood/ebb flow coefficients
    if flood_orientation == 1: # water_elev_1 represents "seaward" water level
        if water_depth_1 > water_depth_2: #  a flood condition
            coef_type = 'flood'
            src_depth, out_depth = water_depth_1, water_depth_2
            water_depth_ratio = water_depth_2/water_depth_1 # landward/seaward
            
        else: #  ebb
            coef_type = 'ebb'
            src_depth, out_depth = water_depth_2, water_depth_1
            water_depth_ratio = water_depth_1/water_depth_2 # seaward/landward
            
    else: # water_elev_2 represents "seaward" water  level
        if water_depth_1 < water_depth_2:
            coef_type = 'flood' # water_elev_1 < water_elev_2 is a flood condition
            src_depth, out_depth = water_depth_2, water_depth_1
            water_depth_ratio = water_depth_1/water_depth_2 # landward/seaward
            
        else: #water_elev_1 > water_elev_2 is ebb
            coef_type = 'ebb'
            src_depth, out_depth = water_depth_1, water_depth_2
            water_depth_ratio = water_depth_2/water_depth_1 # seaward/landward
            
    # Assign sign for directionality
    if coef_type == 'flood': # negative Q for flood
        Q_dir = -1.
    else:
        Q_dir = 1.
    
    
    src_depth_diff = src_depth - out_depth
    depth_condition = (src_depth > 0.) & \
                      (out_depth < src_depth) # water above invert on source side and water source deeper above invert than receiving water
    coef_dict = in_dict['flow_coef'][coef_type]
    
    with np.errstate(invalid='ignore'):
        # --- Establish conditions for partial switching of flow components ---
        
        # Sluice: Supercritical Weir Condition
        sluice_supcrit_weir_cond = (water_depth_ratio < (2./3.)) &\
                                       (src_depth < in_dict['geom']['open_height'])
        
        # Sluice: Free Sluice Condition
        sluice_free_cond = (water_depth_ratio < (2./3.)) &\
                               (src_depth > in_dict['geom']['open_height'])
        
        # Sluice: Subcritical Weir Condition
        sluice_subcrit_weir_cond = (water_depth_ratio >= (2./3.)) &\
                                       (src_depth < in_dict['geom']['open_height'])
        
        # Sluice: Submerged Orifice Condition
        sluice_submer_or_cond = (water_depth_ratio >= (2./3.)) & \
                                    (src_depth > in_dict['geom']['open_height']) &\
                                    (water_depth_ratio >= 0.8)
        
        # Sluice: Transitional Condition
        sluice_transit_cond = (water_depth_ratio >= (2./3)) &\
                                  (src_depth > in_dict['geom']['open_height']) &\
                                  (water_depth_ratio < 0.8)
                                 
        # --- Discharge (flow rate) calculations                
        # Supercritical Broad-crested Weir Flow
        Q_supcrit_weir = depth_condition * sluice_supcrit_weir_cond * \
                             (Q_dir * coef_dict['supercritical_weir']*(2./3.) * \
                              in_dict['geom']['open_width'] * src_depth * \
                                  np.sqrt((2./3)*grav*src_depth))
        
        # Free Sluice Flow
        if coef_type=='ebb' and not ebb_free_headloss:
            sluice_headloss = 0.
        else:
            sluice_headloss = in_dict['flow_coef']['max_flood'] * (1 - 0.5*(src_depth+out_depth)/in_dict['flow_coef']['flood_param'])
        # C_Swamee = ebb_flow_condition * sluice_free_cond*0.611*((src_depth-out_depth)/(src_depth+15*out_depth))**0.072 # Free Flow Sluice-Gate C_d by Prabhata K. Swamee, for comparison against C_d_ebb_free
        Q_free = depth_condition * sluice_free_cond * \
                     (Q_dir * coef_dict['freeflow'] * \
                      A_sluice_open * np.sqrt(2.*grav*(src_depth-sluice_headloss)))
        
        # Subcritical Broad-crested Weir Flow
        Q_subcrit_weir = depth_condition * sluice_subcrit_weir_cond *\
                             (Q_dir * coef_dict['subcritical_weir'] *\
                              in_dict['geom']['open_width'] * out_depth *\
                                  np.sqrt(2*grav*src_depth_diff))
        
        # Submerged Orifice Flow
        Q_submer_or = depth_condition * sluice_submer_or_cond * \
                        (Q_dir * coef_dict['submerged_orifice'] * \
                         A_sluice_open * np.sqrt(2*grav*src_depth_diff))
        
        # Transitional Flow
        Q_transit = depth_condition * sluice_transit_cond * \
                        (Q_dir * coef_dict['transitional'] * \
                         A_sluice_open*np.sqrt(2*grav*3.*src_depth_diff))
    
        # Sum all flow rates for Qtotal
        Q_sluice_calc = np.nansum((Q_free,
                                    Q_transit,
                                    Q_submer_or,
                                    Q_subcrit_weir,
                                    Q_supcrit_weir),axis=0)
    
    if not return_components:
        return Q_sluice_calc
    else:
        Qdict = {'free_{coef_type}':Q_free,
                 'transitional_{coef_type}':Q_transit,
                 'submerged_orifice_{coef_type}':Q_submer_or,
                 'subcritical_{coef_type}': Q_subcrit_weir,
                 'supercritical_{coef_type}':Q_supcrit_weir}
        return Q_sluice_calc, Qdict
    
    

def flood_sluice(landside_water_elev=None, marineside_water_elev=None,
                 in_dict=None,return_components=False,
                 grav=9.80665):
    '''Calculate discharge through sluice gate for flood tide.
    

    Parameters
    ----------
    landside_water_elev : float, optional
        Water elevation landward or upstream of the dike. The default is None.
    marineside_water_elev : float, optional
        Water elevation seaward or downstream of the dike. The default is None.
    in_dict : dict, optional
        Dictionary containing all information about the flow coefficients and
        construction of the dike. The default is None.
    return_components : bool, optional
        True returns the discharge calculated from each sluice equation. False
        only returns their sum. The default is False.
    grav : float, optional
        Gravitational acceleration in consistent units with the elevations. The default is 9.80665.

    Returns
    -------
    Q_dike_sluice_calc_flood, float
        Sum of discharge through sluice.
        
    Qdict, dict
        Dictionary with individual components of discharge. Only returned if return_components=True.

    '''
    
    # =============== Flood tide conditions ================
    # If (H_sea_lev > y_d_HR_lev): 
    # If sea level is greater than HR level -> Negative Flow (Flood Tide, Flap Gates Closed)
    # Flood = discharge back into estuary (Sluice only! Assume no leaks)
    # Sluice gate will always be submerged with flood condition
    
    # Levels relative to culvert invert at sluice.
    marineside_water_depth = marineside_water_elev - in_dict['geom']['invert_elev']
    landside_water_depth = landside_water_elev - in_dict['geom']['invert_elev']
    
    A_sluice_open = in_dict['geom']['open_height']*in_dict['geom']['open_width'] # area of sluice opening (for submerged flow comp)
    
    with np.errstate(invalid='ignore'):
        flood_flow_condition = marineside_water_depth > 0.
        
        water_depth_ratio = landside_water_depth/marineside_water_depth
        flood_free_cond = (marineside_water_depth > landside_water_depth) & (water_depth_ratio < (2./3.))
        flood_submer_or_cond = (marineside_water_depth > landside_water_depth) & (water_depth_ratio > 0.8)
        flood_transit_cond = (marineside_water_depth > landside_water_depth) & (water_depth_ratio > (2./3.)) & (water_depth_ratio < 0.8)
    
        # Free Sluice Flow
        HLsluice = in_dict['flow_coef']['max_flood']*(1.-0.5*(landside_water_depth+marineside_water_depth)/in_dict['flow_coef']['flood_param'])
        # C_Swamee = flood_free_cond*0.611*((marineside_water_depth-landside_water_depth)/(marineside_water_depth+15*landside_water_depth))**0.072  # Free Flow Sluice-Gate C_d by Prabhata K. Swamee, for comparison against C_d_flood_free
        Q_flood_free = flood_flow_condition * flood_free_cond * \
            (-in_dict['flow_coef']['flood']['freeflow']*A_sluice_open * \
             np.sqrt(2.*grav*(marineside_water_depth-HLsluice)))
        
        # Submerged Orifice Flow
        Q_flood_submer_or = flood_flow_condition * flood_submer_or_cond*\
                            (-in_dict['flow_coef']['flood']['submerged_orifice'] * \
                             A_sluice_open*np.sqrt(2*grav*(marineside_water_depth-landside_water_depth)))
        
        # Transitional Flow
        Q_flood_transit = flood_flow_condition * flood_transit_cond*(-in_dict['flow_coef']['flood']['transitional']*A_sluice_open*np.sqrt(2*grav*3*(marineside_water_depth-landside_water_depth)))
        
        # Qflood total
        Q_dike_sluice_calc_flood = np.nansum((Q_flood_free,Q_flood_transit,Q_flood_submer_or),axis=0)
    
    if not return_components:
        return Q_dike_sluice_calc_flood
    else:
        Qdict = {'free_flood':Q_flood_free,
                 'transitional_flood':Q_flood_transit,
                 'submerged_orifice_flood':Q_flood_submer_or,
                 }
        return Q_dike_sluice_calc_flood, Qdict
    

def ebb_sluice(landside_water_elev=None, marineside_water_elev=None,
                 in_dict=None,return_components=False,
                 grav=9.80665):
    '''Calculate discharge through sluice gate for ebb.
    

    Parameters
    ----------
    landside_water_elev : float, optional
        Water elevation landward or upstream of the dike. The default is None.
    marineside_water_elev : float, optional
        Water elevation seaward or downstream of the dike. The default is None.
    in_dict : TYPE, optional
        Dictionary containing all information about the flow coefficients and
        construction of the dike. The default is None.
    return_components : bool, optional
        True returns the discharge calculated from each sluice equation. False
        only returns their sum. The default is False.
    grav : float, optional
        Gravitational acceleration in consistent units with the elevations. The default is 9.80665.

    Returns
    -------
    Q_dike_sluice_calc_ebb, float
        Sum of discharge through sluice.
        
    Qdict, dict
        Dictionary with individual components of discharge. Only returned if return_components=True.

    '''    
    
    # =============== Ebb tide conditions ==================
    
    marineside_water_depth = marineside_water_elev - in_dict['geom']['invert_elev']
    landside_water_depth = landside_water_elev - in_dict['geom']['invert_elev']
    A_sluice_open = in_dict['geom']['open_height']*in_dict['geom']['open_width'] # area of sluice opening (for submerged flow comp)
    
    # If sea level is less than inland/landside water level -> Positive Flow (Ebb Tide)
    with np.errstate(invalid='ignore'):
        water_depth_ratio = landside_water_depth/marineside_water_depth
        
        ebb_flow_condition = landside_water_depth > 0.
        
        # Ebb Sluice Supercritical Weir Condition
        ebb_sluice_supcrit_weir_cond = (marineside_water_depth<landside_water_depth) &\
                                       (1./water_depth_ratio < (2./3.)) &\
                                       (landside_water_depth < in_dict['geom']['open_height'])
        
        # Ebb Sluice Free Sluice Condition
        ebb_sluice_free_cond = (marineside_water_depth<landside_water_depth) &\
                               (1./water_depth_ratio < (2./3.)) &\
                               (landside_water_depth > in_dict['geom']['open_height'])
        
        # Ebb Sluice Subcritical Weir Condition
        ebb_sluice_subcrit_weir_cond = (marineside_water_depth<landside_water_depth) &\
                                        (1./water_depth_ratio >= (2./3.)) &\
                                        (landside_water_depth < in_dict['geom']['open_height'])
        
        # Ebb Sluice Submerged Orifice Condition
        ebb_sluice_submer_or_cond = (marineside_water_depth<landside_water_depth) & \
                                    (1./water_depth_ratio >= (2./3.)) & \
                                    (landside_water_depth > in_dict['geom']['open_height']) &\
                                    (1./water_depth_ratio >= 0.8)
        
        # Ebb Sluice Transitional Condition
        ebb_sluice_transit_cond = (marineside_water_depth<landside_water_depth) &\
                                  (1./water_depth_ratio >= (2./3)) &\
                                  (landside_water_depth > in_dict['geom']['open_height']) &\
                                  (1./water_depth_ratio < 0.8)
    
        # Supercritical Broad-crested Weir Flow
        Q_ebb_supcrit_weir = ebb_flow_condition * ebb_sluice_supcrit_weir_cond * \
                             (in_dict['flow_coef']['ebb']['supercritical_weir']*(2./3.) * \
                              in_dict['geom']['open_width'] * landside_water_depth * \
                                  np.sqrt((2./3)*grav*landside_water_depth))
        
        # Free Sluice Flow
        # C_Swamee = ebb_flow_condition * ebb_sluice_free_cond*0.611*((landside_water_depth-marineside_water_depth)/(landside_water_depth+15*marineside_water_depth))**0.072 # Free Flow Sluice-Gate C_d by Prabhata K. Swamee, for comparison against C_d_ebb_free
        Q_ebb_free = ebb_flow_condition * ebb_sluice_free_cond * \
                     (in_dict['flow_coef']['ebb']['freeflow'] * \
                      A_sluice_open * np.sqrt(2.*grav*landside_water_depth))
        
        # Subcritical Broad-crested Weir Flow
        Q_ebb_subcrit_weir = ebb_flow_condition * ebb_sluice_subcrit_weir_cond *\
                             (in_dict['flow_coef']['ebb']['subcritical_weir'] *\
                              in_dict['geom']['open_width']*marineside_water_depth *\
                                  np.sqrt(2*grav*(landside_water_depth-marineside_water_depth)))
        
        # Submerged Orifice Flow
        Q_ebb_submer_or = ebb_flow_condition * ebb_sluice_submer_or_cond * \
                        (in_dict['flow_coef']['ebb']['submerged_orifice'] * \
                         A_sluice_open*np.sqrt(2*grav*(landside_water_depth-marineside_water_depth)))
        
        # Transitional Flow
        Q_ebb_transit = ebb_flow_condition * ebb_sluice_transit_cond * \
                        (in_dict['flow_coef']['ebb']['transitional'] * \
                         A_sluice_open*np.sqrt(2*grav*3.*(landside_water_depth-marineside_water_depth)))
        
        # Ebb Qtotal
        
        Q_dike_sluice_calc_ebb = np.nansum((Q_ebb_free,
                                            Q_ebb_transit,
                                            Q_ebb_submer_or,
                                            Q_ebb_subcrit_weir,
                                            Q_ebb_supcrit_weir),axis=0)
    
    if not return_components:
        return Q_dike_sluice_calc_ebb
    else:
        Qdict = {'free_ebb':Q_ebb_free,
                 'transitional_ebb':Q_ebb_transit,
                 'submerged_orifice_ebb':Q_ebb_submer_or,
                 'subcritical_ebb': Q_ebb_subcrit_weir,
                 'supercritical_ebb':Q_ebb_supcrit_weir}
        return Q_dike_sluice_calc_ebb, Qdict

def ebb_flap(water_elev_1=None,water_elev_2=None, 
                 in_dict=None,return_components=False,
                 grav=9.80665,angle_tol=1e-4):
    '''
    Calculate discharge through sluice gate for ebb.

    Parameters
    ----------
    water_elev_1 : float, optional
        Water level on outside of gate. For the diked system, this is seaward or downstream. The default is None.
    water_elev_2 : float, optional
        Water level on inside of gate. For the diked system, this is landward or upstream.
    in_dict : TYPE, optional
        Dictionary containing all information about the flow coefficients and
        construction of the dike. The default is None.
    return_components : bool, optional
        True returns the discharge calculated from each sluice equation. False
        only returns their sum. The default is False.
    grav : float, optional
        Gravitational acceleration in consistent units with the elevations. The default is 9.80665.
    angle_tol : float, optional
        Angle tolerance for the solution of the flap gate opening angle. The default is 1e-4.

    Returns
    -------
    Q_dike_flap_calc, float
        Sum of discharge through flap.
        
    Qdict, dict
        Dictionary with individual components of discharge. Only returned if return_components=True.

    '''

    if water_elev_1 >= water_elev_2: # flap gate is closed
        Q_dike_flap_calc = 0.
        Q_ebb_flap_subcrit_weir = 0.
        Q_ebb_flap_supcrit_weir = 0.
    else:

        # Center Flap Gate Calculations
        marineside_water_depth = water_elev_1 - in_dict['geom']['invert_elev']
        landside_water_depth = water_elev_2 - in_dict['geom']['invert_elev']
        A_center_flap_oceanside = marineside_water_depth*in_dict['geom']['width_out']
        
        # Using SciPy fsolve
        # Changing angle of flap gate based on moment
        theta_ebb_flap_deg = []
        if isinstance(marineside_water_depth,(float)): # make them arrays
            marineside_water_depth = np.array([marineside_water_depth])
            landside_water_depth = np.array([landside_water_depth])
            
        for i in range(len(marineside_water_depth)): # Optimize angle for all water level pairs.
            d_hinge_to_y_d = in_dict['geom']['open_el_hinge'] - landside_water_depth[i]
            # d_hinge_to_H = geom_dict['open_el_hinge'] - marineside_water_depth[i]
            # in_args = {'angle_init_flaps':geom_dict['flap_init_angle'],
            #            'w_flaps_out':geom_dict['flap_width_out'],
            #            'w_flaps_in':geom_dict['flap_width_in'], 'h_gate':geom_dict['flap_height'], 'd_hinge_to_H':marineside_water_depth[i],
            #            'd_hinge_to_y_d':d_hinge_to_y_d,'W_gateN':geom_dict['flap_weightN']}
            
            in_args = (in_dict['geom']['init_angle'],in_dict['geom']['width_out'],
                        in_dict['geom']['width_in'], in_dict['geom']['height'],
                        marineside_water_depth[i],
                        d_hinge_to_y_d,in_dict['geom']['weightN'])
            
            
            root = float(fsolve(flap_gate_discharge, 0,args=in_args,xtol=angle_tol)) # use root finder to find angle closest to zero
            theta_ebb_flap_deg.append(np.rad2deg(root))
        
        theta_ebb_flap_deg = np.array(theta_ebb_flap_deg)
        
        # Changing conditions for flaps during ebb tide.
        with np.errstate(invalid='ignore'):
            water_depth_ratio = landside_water_depth/marineside_water_depth
        
            # Ebb Flap Supercritical Weir Condition
            ebb_flap_supcrit_weir_cond = (marineside_water_depth < landside_water_depth) & \
                                            (1./water_depth_ratio < (2./3.)) & \
                                            (theta_ebb_flap_deg > 0.)
            # Ebb Flap Subcritical Weir Condition
            ebb_flap_subcrit_weir_cond = (marineside_water_depth < landside_water_depth) & \
                                         (1./water_depth_ratio > (2./3.)) & \
                                         (theta_ebb_flap_deg > 0.)
        
    
            # Actual head loss as function of maximum head loss
            HL = in_dict['flow_coef']['max_hloss'] * \
                 (1-0.5*(landside_water_depth+marineside_water_depth) / \
                 in_dict['flow_coef']['flap_param'])
            
            # Supercritical BC weir/free sluice
            Q_ebb_flap_supcrit_weir = ebb_flap_supcrit_weir_cond * \
                                    (in_dict['flow_coef']['ebb']['supercritical'] * (2./3.) * \
                                     (landside_water_depth + HL) * in_dict['geom']['width_in'] * \
                                     np.sqrt((2./3.) * grav * (landside_water_depth+HL)))
                    
            # Subcritical BC weir/submerged orifice
            # Use area of water surface on harbor side of flap or HR side?
            Q_ebb_flap_subcrit_weir = ebb_flap_subcrit_weir_cond * \
                (in_dict['flow_coef']['ebb']['subcritical'] * \
                 A_center_flap_oceanside * np.sqrt(2.*grav*(landside_water_depth+HL-marineside_water_depth)))
            
            Q_dike_flap_calc = np.nansum((Q_ebb_flap_subcrit_weir,Q_ebb_flap_supcrit_weir),axis=0)
    
    if not return_components:
        return Q_dike_flap_calc
    else:
        Qdict = {'subcritical_ebb': Q_ebb_flap_subcrit_weir,
                 'supercritical_ebb':Q_ebb_flap_supcrit_weir}
        return Q_dike_flap_calc,Qdict


def flap_gate_discharge(theta, angle_init_flaps=None, w_flaps_out=None,
                        w_flaps_in=None, h_gate=None, d_hinge_to_H=None, d_hinge_to_y_d=None,
                        W_gateN=None, dens_seawater=1024.,
                        grav=9.80665,): 
    """
    Equation for moment around hinge on flap gate.

    Parameters
    ----------
    theta : TYPE
        Angle of gate from initial.
        
    W_gateN: float
        Weight of flap gates (assume same for all - could use better estimate)
        HR gate = 2000 # Newtons -> see excel calculations using gate parts, volumes, and densities.
    
    angle_init_flaps: float
        Initial angle of opening for flap gate.
        
    w_flaps_out: float
        Width of tide gate opening/flapper.
    
    w_flaps_in: float
        Width of tide gate entrance on diked/upstream side.
    
    h_gate: float
        Height of flap gate. Meters from flap gate bottom to hinge. Assume weight is uniformly distributed.
    
    hinge_el_open: float
        Elevation of flap gate hinge [m].
    
    d_hinge_to_H: float
        Hinge open elevation minus the marine water level in meters of elevation.
    
    d_hinge_to_y_d: float
        Hinge open elevation minus the water level on inland side of the dike in meters of elevation.
    

    Returns
    -------
    TYPE
        Sum of moments around hinge.

    """
    
    # Vertical distances from flap gate hinge to water levels.
    # d_hinge_to_y_d = hinge_el_open - diked_wl
    # d_hinge_to_H = hinge_el_open - marine_wl
    
    return -W_gateN*np.sin(theta+angle_init_flaps)*h_gate/dens_seawater/grav - \
           (w_flaps_out*(h_gate**2*np.cos(theta+angle_init_flaps)**2 - 2*h_gate*d_hinge_to_H*np.cos(theta+angle_init_flaps) +\
                        d_hinge_to_H**2)/np.cos(theta+angle_init_flaps)) *\
                        (h_gate-(1/3)*(h_gate-d_hinge_to_H/np.cos(theta+angle_init_flaps))) +\
            w_flaps_in*((h_gate**2*np.cos(theta+angle_init_flaps)**2 -\
                        2*h_gate*d_hinge_to_y_d*np.cos(theta+angle_init_flaps) +\
                        d_hinge_to_y_d**2)/np.cos(theta+angle_init_flaps)) *\
                        (h_gate-(1/3)*(h_gate - d_hinge_to_y_d/np.cos(theta+angle_init_flaps)))

    

def calc_qmatrix(water_elev_1=None,
                 water_elev_2=None,
                 control_dict=None,qdf=None,
                 ndecimal=3, return_components=False):
    '''
    Calculate discharge matrix for water level elevations.

    Parameters
    ----------
    water_elev_1 : list, np.ndarray, optional
        Water level on outside of gate. For the diked system, this is seaward or downstream. The default is None.
    water_elev_2 : list, np.ndarray, optional
        Water level on inside of gate. For the diked system, this is landward or upstream. The default is None.
    control_dict : dict, optional
        Dictionary containing information on each flow control structure, including its
        geometry and flow coefficients for various flow conditions. For an example, see
        HRE_Model_to2100_currentinfrastructure.py. The default is None.

    Returns
    -------
    qdf : pandas.DataFrame
        Dataframe with rows set by water_elev_1 and columns by water_elev_2 with the 
        value of each cell the total discharge through all control structures for the
        water level conditions set by row, col.
        
    outQ_dict : dict
        Dictionary with individual components of discharge storing a matrix for each.

    '''
    
    melev = np.round(water_elev_1,decimals=ndecimal)
    lelev = np.round(water_elev_2,decimals=ndecimal)
    
    if qdf is None:
        # initialize dataframe, marineside rows by landside columns
        qdf = pd.DataFrame([],index=melev,columns=lelev,dtype=float)
        
    else:
        # Check if enough values are included in index and columns
        add_index = [i for i in melev if i not in qdf.index.values]
        add_col = [i for i in lelev if i not in qdf.columns.values]
        qdf[add_col] = np.nan # add columns
        qdf = qdf.reindex(qdf.index.union(add_index)).copy()
        qdf = qdf.reindex(sorted(qdf.columns),axis=1)
        qdf = qdf.reindex(sorted(qdf.index.values),axis=0)
    
    if return_components:
        save_Q = np.nan*np.zeros_like(qdf.values)
        outQ_dict = {}
    
    for ime, me in enumerate(melev):
        for ile, le in enumerate(lelev):
            if np.isnan(qdf.at[me,le]): # Only calculate Q for un-run combinations
                Q_init = 0 # add constant Q later to allow transient Q inputs
                for istructure in list(control_dict.keys()): # Loop through each structure
                    Q_flap = 0.
                    Q_sluice = 0.
                    Qdict1 = None

                    # Update input dictionary to describe active structure 
                    in_dict = {'water_elev_2': le,
                               'water_elev_1': me,
                               'return_components':return_components,
                        'in_dict':{'geom':control_dict[istructure]['geom'],
                                'flow_coef':control_dict[istructure]['flow_coef']}}
                    
                    if control_dict[istructure]['type'] == 'flap':
                        if return_components:
                            Qtemp, Qdict1 = ebb_flap(**in_dict)
                            Q_flap = control_dict[istructure]['multiplier'] * Qtemp
                        else:
                            Q_flap = control_dict[istructure]['multiplier'] * ebb_flap(**in_dict)
                        # Can add gate leakage (commented out below) here
                        
                    elif control_dict[istructure]['type'] == 'sluice':
                        if return_components:
                            Qetemp,Qdict1 = sluice_flow(**in_dict)
                            Q_sluice = control_dict[istructure]['multiplier'] * Qetemp

                        else:
                            Q_sluice = control_dict[istructure]['multiplier'] * sluice_flow(**in_dict)
                    
                    if return_components:
                        if istructure not in list(outQ_dict.keys()):
                            outQ_dict.update({istructure:{}}) # initialize structure in dictionary
                        
                        all_keys = list(Qdict1.keys())
                        
                        for ikey in all_keys:
                            if ikey not in list(outQ_dict[istructure].keys()):
                                outQ_dict[istructure].update({ikey:save_Q.copy()})# Initialize Q matrix
                            # Save Q for each condition
                            if isinstance(Qdict1[ikey],float):
                                outQ_dict[istructure][ikey][ime,ile] = Qdict1[ikey]
                            else:
                                outQ_dict[istructure][ikey][ime,ile] = Qdict1[ikey][0]
                            
              
                    # Calculate total discharge for all structures (+ is to marine, - is into landside waterbody)
                    Q_init += Q_flap + Q_sluice
                    # Update entry in qdf
                
                qdf.at[me,le] = Q_init
    

    
    if return_components:
        return qdf,outQ_dict
    else:
        return qdf
    
    

def diked_waterlevel_solver(marineside_water_elev,  dt, Q_const=0, h_init=0,
                            stage_storage_dict=None, control_dict=None,
                            verbose=False,print_nsteps=10000,saveQ=False):
    '''Solve dike flow using marine water level and stage-storage information.
    

    Parameters
    ----------
    marineside_water_elev : np.ndarray,  list
         Water elevations seaward or downstream of the dike.
    dt : float
        Model and marineside_water_elev time step in consistent units with output Q.
    Q_const : float, np.ndarray, list, optional
        Source/sink of water into the water balance from sources other than the dike.
        It can be negative to represent losses (i.e., pumping, evaporation). If Q_const is
        a list-type, then it must be the same length as marineside_water_elev, and each timestep
        in the water balance will use the corresponding Q_const value. The default is 0.
    h_init : float, optional
        Starting surface water head inside water control structures. The default is 0.
    stage_storage_dict : dict, optional
        Stage-storage dictionary with keys 'stage' and 'storage' that is used to update the 
        water level upstream of the dike at the end of each timestep. The default is None.
    control_dict : dict, optional
        Dictionary containing information on each flow control structure, including its
        geometry and flow coefficients for various flow conditions. For an example, see
        HRE_Model_*.py. The default is None.
    verbose : bool, optional
        True turns on verbose mode. The default is False.
    print_nsteps : int, optional
        With verbose=True, this sets how frequently the model progress is printed to the console.
        The default is 10000.
    saveQ : bool, optional
        Boolean for saving the discharge from each flow structure through time. The default is False.

    Returns
    -------
    landside_water_elev : np.ndarray
        Modeled water level on the upstream/inland side of the dike for each model step.
    volume_array : np.ndarray
        Modeled water volume stored behind the dike for each model step.
    Q_dict : dict
         Dictionary with individual components of discharge. Only returned if saveQ=True..

    '''
    landside_water_elev = np.zeros_like(marineside_water_elev)
    volume_array = np.zeros_like(marineside_water_elev) # array stores volume in diked waterbody for each time
    Volume_update = None
    
    nsteps = len(marineside_water_elev)
    
    if saveQ: # save discharge at each timestep
        Q_dict = {'background':np.zeros(nsteps)}
        for istructure in list(control_dict.keys()):
            Q_dict.update({istructure:{'flood':np.zeros(nsteps),
                                       'ebb':np.zeros(nsteps)}})
    
    for i in range(nsteps): # loop through time, one marineside water level at a time
        # Have y_out initial, y_in initial, synthetic curve to get initial volume from y_in initial
        landside_water_elev[i] = h_init # use head calculated from the last time step to initialize new Q calculations
        
        if i == 0:
            # Solve for volume at given diked water level, landside_water_elev
            volume_array[i] = np.interp(landside_water_elev[i],stage_storage_dict['stage'], stage_storage_dict['storage'])
        else:
            volume_array[i] = Volume_update # stored volume from last iteration
        
        in_dict = {'water_elev_2': np.array([landside_water_elev[i]]),
               'water_elev_1': np.array([marineside_water_elev[i]]),
               }
        
        # Inflow from upstream       
        if isinstance(Q_const,(list,tuple,np.ndarray)):
            Q_init = Q_const[i]
        else:
            Q_init = Q_const
            
        # Discharge calculations for each dike structure    
        if saveQ:
            Q_dict['background'][i] = Q_const
        for istructure in list(control_dict.keys()): # Loop through each structure
            Q_flap = 0.
            Q_sluice = 0.
            
            # Update input dictionary to describe active structure 
            in_dict.update({'in_dict':{'geom':control_dict[istructure]['geom'],
                                       'flow_coef':control_dict[istructure]['flow_coef']}})
            
            if control_dict[istructure]['type'] == 'flap':
                Q_flap = control_dict[istructure]['multiplier'] * ebb_flap(**in_dict)
                
            elif control_dict[istructure]['type'] == 'sluice':
                Q_sluice = control_dict[istructure]['multiplier'] * sluice_flow(**in_dict)
           
            if saveQ:
                if Q_flap > 0. or Q_sluice > 0.:
                    Q_dict[istructure]['ebb'][i] = Q_flap + Q_sluice
                elif Q_sluice < 0.:
                    Q_dict[istructure]['flood'][i] = Q_flap + Q_sluice
            
            # Calculate total discharge for all structures (+ is to marine, - is into landside waterbody)
            Q_init += Q_flap + Q_sluice
        

        # Calculate change in dike waterbody volume over the time step, dt
        V_flux_init = Q_init*dt # Volume passed is discharge multipled by time
    
        # Update volume of water stored behind dike - Calculate water stored behind dike for the next time step
        Volume_update = volume_array[i] - V_flux_init # "Add" V_flux to initial volume (subtract since discharge into dike area is negative)

        # Calculate water level behind dike (inland_water_elev) with new volume for next time step
        # using the stage-storage relationship.
        h_init = np.interp(Volume_update,stage_storage_dict['storage'],stage_storage_dict['stage'])

        if verbose:
            if np.mod(i,print_nsteps)==0:
                print("Elapsed model time = {0:3.2f} yr | {1}/{2} | {3:3.1f}%".format(i*dt/86400/365.25,i,nsteps,100.*float(i)/nsteps))
    if saveQ:
        return landside_water_elev,volume_array,Q_dict
    else:
        return landside_water_elev,volume_array

def total_Q(q_dict=None):
    '''
    Calculate total discharge from a component dictionary.

    Parameters
    ----------
    q_dict : dict, optional
        Dictionary with individual components of discharge. The default is None.

    Returns
    -------
    all_Q : float, np.ndarray
        Total discharge summed over components, keeping the same size as input components.

    '''
    all_Q = None
    for ikey in list(q_dict.keys()):
        if isinstance(q_dict[ikey],(list,np.ndarray)):
            if all_Q is None:
                all_Q = np.array(q_dict[ikey]).copy()
            else:
                all_Q += q_dict[ikey]
        elif isinstance(q_dict[ikey],dict):
            if all_Q is None:
                all_Q = q_dict[ikey]['flood'] + \
                        q_dict[ikey]['ebb']
            else:
                all_Q += q_dict[ikey]['flood'] + \
                         q_dict[ikey]['ebb']
                    
    return all_Q

def integrate_Q(dates, discharge):
    """
    Calculate net volume of flow over a time period.
    
    Function for extracting an estimated volume flux over a time period from 
    discharge through a dike using the trapezoid rule (e.g., water volume through
    dike each hour). Input initial discharge value and discharge at t + delta t.

    Parameters
    ----------
    dates : list-like, pandas.Series
        Time list.
    discharge : list-like
        Discharge rates to be integrated over time.

    Returns
    -------
    V_flux : np.ndarray
        Integrated volume over each timestep.

    """
    V_flux = np.zeros_like(discharge)
    for bin_index in range(len(dates)-1):
        disch_start = discharge[bin_index]
        disch_end = discharge[bin_index+1]
        date_interval = (dates[bin_index+1] - dates[bin_index]).seconds # date interval in seconds
        V_flux[bin_index+1] = integrate.trapz(np.array([disch_start,disch_end]),np.array([0,date_interval]))
    return V_flux

# ------------------ End Discharge functions ------------------

# -------------- Processing functions --------------------
def nse(simulations, evaluation, K_version=False):
    """Nash-Sutcliffe Efficiency (NSE) as per `Nash and Sutcliffe, 1970
    <https://doi.org/10.1016/0022-1694(70)90255-6>`_.
    
    After: Hallouin, T., (2019), HydroEval: Streamflow Simulations Evaluator 
    (Version 0.0.3). Zenodo. `doi.org/10.5281/zenodo.3402383 
    <https://doi.org/10.5281/zenodo.3402383>`_.
    
    :Calculation Details:
        .. math::
           E_{\\text{NSE}} = 1 - \\frac{\\sum_{i=1}^{N}[e_{i}-s_{i}]^2}
           {\\sum_{i=1}^{N}[e_{i}-\\mu(e)]^2}
        where *N* is the length of the *simulations* and *evaluation*
        periods, *e* is the *evaluation* series, *s* is (one of) the
        *simulations* series, and *μ* is the arithmetic mean.
        
    Logarithm option suggested for NSE,  using Eqn 7
    Krause, P., D. P. Boyle, and F. Bäse (2005), Comparison of different efficiency
    criteria for hydrological model assessment, Adv. Geosci., 5(89), 89–97,
    doi:10.5194/adgeo-5-89-2005.
    
    Parameters
    ----------
    simulations : np.ndarray, list
        Values simulated by model.
    evaluation : np.ndarray, list
        Observation values.
    log_version : TYPE, optional
        Logarithm of the values to reduce influence of extreme values. The default is False.

    Returns
    -------
    nse_ : float
        Nash-Sutcliffe Efficiency value.
    
    """
    
    if K_version:
        new_datum = np.nanmin([evaluation,simulations])-1
        
        ev1 = evaluation-new_datum
        s1 = simulations-new_datum
        mean_eval = np.nanmean(ev1)
        
        nse_ = 1 - (
                np.nansum(((ev1 - s1)/ev1) ** 2, dtype=np.float64) / \
             np.nansum(((ev1 - mean_eval)/mean_eval) ** 2, dtype=np.float64)
        )
    else:
    
        nse_ = 1 - (
                np.nansum((evaluation - simulations) ** 2, dtype=np.float64)
                / np.nansum((evaluation - np.nanmean(evaluation)) ** 2, dtype=np.float64)
        )

    return nse_

def rnp(sim,obs,parametric=False):
    '''
    ###################################################################
    # RNP: Efficiency with non-parametric components
    # RNP consists of three components: 
    #     - mean discharge (beta component),
    #     - normalized flow duration curve (alpha component)
    #     - Spearman rank correlation (r component)
    # RNP needs as input observed and simulated discharge time series 
    # A perfect fit results in a RNP value of 1                       
    # Created: 22-02-2018                                             
    ###################################################################
    
    Source
    -------
    
    Pool, S., M. Vis, and J. Seibert (2018), Evaluating model performance: 
        towards a non-parametric variant of the Kling-Gupta efficiency, 
        Hydrol. Sci. J., 63(13–14), 1941–1953, doi:10.1080/02626667.2018.1552002.
    
     https://www.tandfonline.com/doi/full/10.1080/02626667.2018.1552002

    Parameters
    ----------
    sim : np.ndarray
        Simulated value or list.
    obs : np.ndarray
        Observed value or list.

    Returns
    -------
    rnp_ : np.ndarray
        Calculated efficiency array.

    '''
    
    # Remove nan values
    not_nan = ~np.isnan(sim) & ~np.isnan(obs)
    sim = sim[not_nan]
    obs = obs[not_nan]
    
    # Calculate means
    mean_sim = np.nanmean(sim)
    mean_obs = np.nanmean(obs)
    
    # Calculate beta
    beta = mean_sim / mean_obs
    
    if parametric:
        # Calculate alpha
        alpha = np.nanstd(sim)/np.nanstd(obs)
        
        # Pearson correlation
        r,p = pearsonr(sim,obs)
        
    else:
    
        # Calculate normalized values
        norm_sim = np.sort(sim/(mean_sim*len(sim)))
        norm_obs = np.sort(obs/(mean_obs*len(obs)))
        
        # Calculate alpha
        alpha = 1 - 0.5 * np.nansum(np.abs(norm_sim - norm_obs))
        
        
        # Calculate two-sided Spearman r
        r,p = spearmanr(sim,obs)
        
    # Calculate parametric or non-parametric variant of Kling-Gupta efficiency
    rnp_ = 1 - np.sqrt((alpha-1)**2 + (beta-1)**2 + (r-1)**2)
    
    return rnp_

def prepare_water_elev(fname,dt='600s',start_datetime=None,end_datetime=None,
                       time_col='datetime',
                       interp_method='linear',interp_order=1, interp_limit=12):
    '''
    Load water elevations to new timestep.

    Parameters
    ----------
    fname : str
        Filename of input water levels.
    dt : str, optional
        Timestep in pandas-readable string format. The default is '600s'.
    start_datetime : str, datetime.datetime, optional
        Starting date for the output water level series. The default is None.
    end_datetime : str, datetime.datetime, optional
        Ending date to use for the output water level series. The default is None.
    time_col : str, optional
        Name of time column in fname. The default is 'datetime'.
    interp_method : str, optional
        Interpolation method for changing to new timestep. The default is 'linear'.
    interp_order : int, optional
        Interpolation order. The default is 1.
    interp_limit : int, optional
        Interpolation limit - number of points that would be interpolated/filled. The default is 12.

    Returns
    -------
    df_resampled : pandas.DataFrame
        Resampled water level time series data.

    '''
    df = pd.read_csv(fname)
    
    # Specify data types
    data_cols = df.columns.drop(time_col)
    df[data_cols] = df[data_cols].apply(pd.to_numeric,errors='coerce')
    df[time_col] = pd.to_datetime(df[time_col])
    
    df.drop_duplicates(subset=time_col,inplace=True)
    # Prepare for and apply resampling in time
    df = df.set_index(time_col)
    if not isinstance(dt,str):
        # Need to update dt to be a str, dt required to have units of seconds
        dt = '{0}s'.format(dt)
    
    
    
    df_resampled = df.resample(dt).asfreq()
    df_resampled = df_resampled.interpolate(method=interp_method,
                                            order=interp_order,
                                            limit=interp_limit)
   
    
    if start_datetime is not None:
        df_resampled = df_resampled[start_datetime:]
    
    if end_datetime is not None:
        df_resampled = df_resampled[:end_datetime]
    
    df_resampled.reset_index(inplace=True)
    
    return df_resampled
    
def unique_rows(a,sort=True,return_inverse=False):
    '''
    Find unique rows and return indexes of unique rows

    Parameters
    ----------
    a : np.ndarray
        Input two-dimensional array.
    sort : bool, optional
        Boolean to sort the unique indexes. The default is True.
    return_inverse : bool, optional
        Boolean to return the inverse array. The default is False.

    Returns
    -------
    unique_a, np.ndarray
        Unique row values. Only returned if return_inverse = True.
    uind : np.ndarray
        Indexes of the first index for each unique row in a. Only returned if return_inverse = True.
    univ: np.ndarray
        Inverse array of indexes to recreate a from unique_a and uind. Only returned if return_inverse = True.
    outorder:
        Unique row indexes, either sorted or not sorted, depending on sort variable.

    '''
    a = np.ascontiguousarray(a)
    unique_a,uind,uinv = np.unique(a.view([('', a.dtype)]*a.shape[1]),return_index=True,return_inverse=True)
    if sort:    
        uord = [(uind==utemp).nonzero()[0][0] for utemp in np.sort(uind)]
        outorder = uind[uord]
    else:
        outorder = uind
    if return_inverse:
        return unique_a,uind,uinv
    else:
        return outorder
# -------------- End Processing functions --------------------
                                          