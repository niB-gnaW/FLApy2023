import FLApy as fp
import matplotlib.pyplot as plt
import pyvista as pv

site = fp.DataManagement.StudyFieldLattice()
site.read_LasData('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/SimForestStandardDis10Numtree300Sub05.las')
site.gen_SFL([100,200,100,200], 1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=1)
fp.Visualization.vis_3Dpoint(site._terPoints_buffered)

siteLA = fp.LAcalculator.LAcalculator(site)
siteLA.computeBatch(multiPro = 'p_map', CPU_count=6)

siteLAH = fp.LAHanalysis.LAH_analysis('siteLA')
siteLAH.computeLAH(save='/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/FineCal.vtk')

plotSite = fp.Visualization.vis_Figures()