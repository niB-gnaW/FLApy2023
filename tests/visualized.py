import FLApy as fp

site = fp.DataManagement.dataInput('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/.temFile.vtk').read_VTK()

fp.Visualization.vis_3Dpoint(site.field_data['DSM'])

a = 1