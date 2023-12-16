# -*- coding: utf-8 -*-

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.gridspec as grd
import FLApy as fp
import matplotlib.ticker as ticker
import os

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

    if inRaster is None:
        raise ValueError('Raster data is empty!')

    if isinstance(inRaster, np.ndarray):
        dataP2M = DataManagement.StudyFieldLattice.p2m(inRaster, resolution)
    elif isinstance(inRaster, xr.DataArray):
        dataP2M = inRaster

    dataP2M.plot()
    return plt.show()

def vis_SFL(inSFL, field = 'SVF_flat'):
    # The function is used to visualize the study field lattice
    # inSFL: study field lattice, structured DataArray
    # return: study field lattice visualization

    dataSFL = inSFL
    dataSFL.active_scalars_name = field
    dataSFL.set_active_scalars(field)
    P = pv.Plotter()
    P.add_mesh(dataSFL, cmap='viridis', show_scalar_bar=True)
    P.show_grid()
    return P.show()

def vis_Figures(inSFL, field = 'SVF_flat'):
    # The function is used to visualize the figures of LAH analysis
    # return: figures visualization

    if inSFL is None:
        raise ValueError('Study field lattice is empty!')

    else:

        if isinstance(inSFL, str) is True:
            if os.path.exists(inSFL) is False:
                raise ValueError('Study field lattice is empty!')
            else:
                dataSFL = DataManagement.dataInput(inSFL).read_VTK()

        else:
            if hasattr(inSFL, '_SFL') is True:
                dataSFL = inSFL._SFL
            elif hasattr(inSFL, '_DataContainer') is True:
                dataSFL = inSFL._DataContainer
            elif hasattr(inSFL, '_inGrid') is True:
                dataSFL = inSFL._inGrid
            elif hasattr(inSFL, '_SFL') is False and hasattr(inSFL, '_DataContainer') is False and hasattr(inSFL, '_inGrid') is False:
                dataSFL = inSFL

    dataY = dataSFL.cell_data[field]
    dataX = dataSFL.cell_data['Z_normed']


    plt.figure()
    gs = grd.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.1)

    ax1 = plt.subplot(gs[0, 0])
    sns.kdeplot((dataY * 100), shade=True, common_norm = True)
    ax1.set_xlabel('LA (%)')
    ax1.set_xlim(0, 100)

    ax2 = plt.subplot(gs[0, 1])
    X_vertical = dataSFL.field_data['Z_normed_full']
    Y_vertical = (dataSFL.field_data['SVF_flat_full']) * 100
    x_range = np.linspace(0, np.max(X_vertical), 100)
    _params = [dataSFL.field_data['LAR_Ver'], dataSFL.field_data['HIP_Ver']]
    sns.scatterplot(y = X_vertical, x = Y_vertical, alpha=0.1, edgecolor='none', s = 1)
    sns.lineplot(y = x_range, x = fp.LAHanalysis.sigmoid_func(x_range, *_params), color = 'red')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('Relative height (m)')
    ax2.set_xlabel('LA (%)')
    ax2.set_xlim(0, 100)

    ax3 = plt.subplot(gs[1, 0])
    Y_horizontal = (dataY[np.abs(dataX - 1.5) <= 0.5]) * 100
    sns.kdeplot(Y_horizontal, shade=True, common_norm = True)
    ax3.set_xlabel('LA (%)')
    ax3.set_xlim(0, 100)
    ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

    ax4 = plt.subplot(gs[1, 1])
    hotspotAll = dataSFL.field_data['LAH_3Dcluster_Hot_SVF'] * 100
    coldspotAll = dataSFL.field_data['LAH_3Dcluster_Cold_SVF'] * 100
    hotspotLabel = np.full((len(hotspotAll)), 'Hotspot')
    coldspotLabel = np.full((len(coldspotAll)), 'Coldspot')
    hotArray = np.vstack((hotspotAll, hotspotLabel)).T
    coldArray = np.vstack((coldspotAll, coldspotLabel)).T
    pdata = np.vstack((hotArray, coldArray))
    pdata2 = pd.DataFrame({'SVF': np.array(pdata[:, 0], dtype=float), 'Type': np.array(pdata[:, 1], dtype=str)})
    sns.boxplot(data=pdata2, x='SVF', y='Type')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_xlabel('LA (%)')
    ax4.set_ylabel('Zone')
    ax4.set_xlim(0, 100)


    return plt.show()








