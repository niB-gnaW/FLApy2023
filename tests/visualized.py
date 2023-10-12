import FLApy as fp
import pyvista as pv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



site = fp.DataManagement.dataInput('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/FineCal.vtk').read_VTK()
site.active_scalars_name = 'Gi_Value'

DSM = site.field_data['DSM_cliped']
DSM = pv.PolyData(DSM)

PTS = site.field_data['PTS']
PTS = pv.PolyData(PTS)
PTS['z'] = PTS.points[:, -1]

'''''
P = pv.Plotter()
P.add_mesh(site, cmap='viridis', show_scalar_bar=True, opacity=0.9)
P.add_mesh(PTS, cmap='jet', show_scalar_bar=True)
P.show_grid()
P.show()
'''''

fp.Visualization.vis_Figures(site, 'SVF_flat')

'''''
hotspotAll = site.field_data['LAH_3Dcluster_Hot_SVF']
coldspotAll = site.field_data['LAH_3Dcluster_Cold_SVF']
hotspotLabel = np.full((len(hotspotAll)), 'Hotspot')
coldspotLabel = np.full((len(coldspotAll)), 'Coldspot')
hotArray = np.vstack((hotspotAll, hotspotLabel)).T
coldArray = np.vstack((coldspotAll, coldspotLabel)).T
pdata = np.vstack((hotArray, coldArray))
pdata2 = pd.DataFrame({'SVF': np.array(pdata[:, 0], dtype=float), 'Type': np.array(pdata[:, 1], dtype=str)})

example_data = pd.DataFrame({
    'Type': ['A', 'B', 'A', 'B'],
    'SVF': [1, 2, 3, 4]
})

sns.boxplot(data=pdata2, x='SVF', y='Type', hue='Type', palette='vlag')

plt.show()
'''''
