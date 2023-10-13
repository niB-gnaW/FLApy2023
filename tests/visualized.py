import FLApy as fp
import pyvista as pv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



site = fp.DataManagement.dataInput('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/SimForestStandardDis10Numtree100Sub05_FIN.vtk').read_VTK()
site.set_active_scalars('Gi_Value')
hotspot = site.threshold(value=2.576, invert = False)
coldspot = site.threshold(value=[-99998, -2.576])


PTS = site.field_data['PTS']
PTS = pv.PolyData(PTS)
PTS['z'] = PTS.points[:, -1]


P = pv.Plotter()
P.add_mesh(hotspot, color='red', opacity=0.5)
P.add_mesh(coldspot, color='blue', opacity=0.5)
#P.add_mesh(PTS, cmap='jet', show_scalar_bar=True)
P.show_grid()
P.show()

'''''
DSM = site.field_data['DSM_cliped']
DSM = pv.PolyData(DSM)

PTS = site.field_data['PTS']
PTS = pv.PolyData(PTS)
PTS['z'] = PTS.points[:, -1]


P = pv.Plotter()
P.add_mesh(site, cmap='jet', show_scalar_bar=True, opacity=1, scalars='SVF_flat')
#P.add_mesh(PTS, cmap='jet', show_scalar_bar=True)
P.show_grid()
P.show()


relativeHeight = site.field_data['Z_normed_full']
SVF = (site.field_data['SVF_flat']) * 100

sns.scatterplot(x=relativeHeight, y=SVF, color = 'black', alpha=0.1)
xrangge = np.linspace(0, np.max(relativeHeight), 100)
_params = [site.field_data['LAR_Ver'], site.field_data['HIP_Ver']]
sns.lineplot(x = xrangge, y = fp.LAHanalysis.sigmoid_func(xrangge, *_params), color = 'red')
plt.show()
'''''




