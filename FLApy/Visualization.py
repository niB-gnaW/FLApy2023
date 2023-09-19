# -*- coding: utf-8 -*-
#---------------------------------------------------------------------#
#   FLApy: Forest Light Analyzer python package                       #
#   Developer: Wang Bin (Yunnan University, Kunming, China)           #

import pyvista as pv

def vis_3Dpoint(inPoints):    #xyz

    pc = pv.PolyData(inPoints)

    value = inPoints[:, -1]

    pc['Elevation'] = value

    return pc.plot(render_points_as_spheres=False, show_grid=True)



