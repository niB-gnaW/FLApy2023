import FLApy as fp
import pyvista as pv

site = fp.DataManagement.StudyFieldLattice()
site.read_LasData('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/SimForestStandardDis10Numtree100.las')
site.gen_SFL([100,200,100,200], 10)

siteLA = fp.LAcalculator.LAcalculator(site)
a = siteLA.computeBatch(multiPro = 'p_map', CPU_count=4)



a = 1