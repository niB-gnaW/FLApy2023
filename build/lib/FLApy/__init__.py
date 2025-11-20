name = "FLApy"
import os
import vtk

try:
    import naturalneighbor
except ImportError:
    # Only require this package on Windows
    if os.name == 'nt': 
        raise ImportError(
            "Critical dependency 'naturalneighbor' is missing.\n"
            "Please run the initialization script once before using FLApy:\n"
            "   python -m FLApy.FLApy_ini"
        )

os.environ['USE_PYGEOS'] = '0'

from FLApy import DataManagement, LAcalculator, LAHanalysis, Visualization
vtk.vtkObject.GlobalWarningDisplayOff()

import pyvista

FLApy_theme = pyvista.themes.DocumentTheme()
FLApy_theme.background = 'white'
FLApy_theme.title = 'FLApy'
FLApy_theme.font.family = 'Times'
FLApy_theme.font.color = 'black'
FLApy_theme.axes.x_color = 'black'
FLApy_theme.axes.y_color = 'black'
FLApy_theme.axes.z_color = 'black'
FLApy_theme.outline_color = 'black'


pyvista.global_theme.load_theme(FLApy_theme)




