# -*- coding: utf-8 -*-
#---------------------------------------------------------------------#
#   FLApy: Forest Light Analyzer python package                       #
#   Developer: Wang Bin (Yunnan University, Kunming, China)           #
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import xarray as xr

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








