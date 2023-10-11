import FLApy as fp
import pyvista as pv

site = fp.DataManagement.dataInput('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/FineCal.vtk').read_VTK()
site.active_scalars_name = 'Gi_Value'

DSM = site.field_data['DSM_cliped']
DSM = pv.PolyData(DSM)

PTS = site.field_data['PTS']
PTS = pv.PolyData(PTS)
PTS['z'] = PTS.points[:, -1]


P = pv.Plotter()
P.add_mesh(site, cmap='terrain', show_scalar_bar=True, opacity=0.8)

#P.add_mesh(DSM, cmap='viridis')
#P.add_mesh(PTS)
P.show_grid()
P.show()