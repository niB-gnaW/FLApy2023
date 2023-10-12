# -*- coding: utf-8 -*-
#---------------------------------------------------------------------#
#   FLApy: Forest Light Analyzer python package                       #
#   Developer: Wang Bin (Yunnan University, Kunming, China)           #
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as grd
from matplotlib.pyplot import MultipleLocator
import FLApy as fp
from scipy.optimize import curve_fit
import tqdm


from FLApy import DataManagement

def vis_3Dpoint(inPoints):    #xyz
    # The function is used to visualize the 3D point cloud (xyz, (n, 3))
    # inPoints: 3D point cloud, numpy array
    # return: 3D point cloud visualization

    pc = pv.PolyData(inPoints)

    value = inPoints[:, -1]

    pc['Elevation'] = value

    return pc.plot(render_points_as_spheres=False, show_grid=True)

def vis_Raster(inRaster, resolution = 1):
    # The function is used to visualize the raster data
    # inRaster: raster data, numpy array
    # return: raster data visualization

    if isinstance(inRaster, np.ndarray):
        dataP2M = DataManagement.StudyFieldLattice.p2m(inRaster, resolution)
    elif isinstance(inRaster, xr.DataArray):
        dataP2M = inRaster

    dataP2M.plot()
    return plt.show()

def vis_SFL(inSFL, field):
    # The function is used to visualize the study field lattice
    # inSFL: study field lattice, structured DataArray
    # return: study field lattice visualization

    dataSFL = inSFL
    dataSFL.active_scalars_name = field

    P = pv.Plotter()
    P.add_mesh(dataSFL, cmap='viridis', show_scalar_bar=True)
    P.show_grid()
    return P.show()

def vis_Figures(inSFL, field):
    # The function is used to visualize the figures
    # return: figures visualization

    dataSFL = inSFL
    dataSFL.active_scalars_name = field

    dataY = dataSFL[field]


    fig = plt.figure()
    gs = grd.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.4, wspace=0.3)

    ax1 = plt.subplot(gs[0, 0])
    sns.kdeplot(dataY, shade=True, common_norm = True)
    ax1.set_xlabel('LA')



    ax2 = plt.subplot(gs[0, 1])
    X_vertical = dataSFL['Z_normed']
    Y_vertical = dataY
    coords = dataSFL.cell_centers().points
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    v_coords = dataY
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    coordsWall = np.meshgrid(x_unique, y_unique)
    coordsXwall = coordsWall[0].flatten()
    coordsYwall = coordsWall[1].flatten()
    paramaters = np.zeros((len(coordsXwall), 2))

    x_range = np.linspace(0, np.max(z_coords), 100)


    for i in tqdm.tqdm(range(len(coordsXwall))):
        extract = np.logical_and(x_coords == coordsXwall[i], y_coords == coordsYwall[i])
        filterZ = z_coords[extract]
        filterV = v_coords[extract]

        sns.scatterplot(x=filterZ, y=filterV, color='black', alpha=0.1)
        _params, _ = curve_fit(fp.LAHanalysis.sigmoid_func, filterZ, filterV, maxfev=99999)
        sns.lineplot(x=x_range, y=fp.LAHanalysis.sigmoid_func(x_range, *_params), color='black')
        plt.show()


        _params, _ = curve_fit(fp.LAHanalysis.sigmoid_func, filterZ, filterV, maxfev=99999)
        sns.lineplot(x=x_range, y=fp.LAHanalysis.sigmoid_func(x_range, *_params), color='black', alpha=0.1)
        paramaters[i][0] = _params[0]
        paramaters[i][1] = _params[1]

    sns.lineplot(x = x_range, y = fp.LAHanalysis.sigmoid_func(x_range, np.mean(paramaters[:, 0]), np.mean(paramaters[:, 1])), color='red')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xlabel('Relative height (m)')
    ax2.set_ylabel('LA')



    ax3 = plt.subplot(gs[1, 0])
    Y_horizontal = dataY[np.abs(X_vertical - 1.5) <= 0.5]
    sns.kdeplot(Y_horizontal, shade=True, common_norm = True)
    ax3.set_xlabel('LA')



    ax4 = plt.subplot(gs[1, 1])
    hotspotAll = dataSFL.field_data['LAH_3Dcluster_Hot_SVF']
    coldspotAll = dataSFL.field_data['LAH_3Dcluster_Cold_SVF']
    hotspotLabel = np.full((len(hotspotAll)), 'Hotspot')
    coldspotLabel = np.full((len(coldspotAll)), 'Coldspot')
    hotArray = np.vstack((hotspotAll, hotspotLabel)).T
    coldArray = np.vstack((coldspotAll, coldspotLabel)).T
    pdata = np.vstack((hotArray, coldArray))
    pdata2 = pd.DataFrame({'SVF': np.array(pdata[:, 0], dtype=float), 'Type': np.array(pdata[:, 1], dtype=str)})
    sns.boxplot(data=pdata2, x='SVF', y='Type', hue='Type', palette='vlag')
    ax4.get_legend().remove()
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

    return plt.show()








