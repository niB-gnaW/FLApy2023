import FLApy as fp

# Initialize the study field lattice and read the LiDAR data
site = fp.DataManagement.StudyFieldLattice()
site.read_LasData('/Users/wangbin/PythonSpace/PythonEX/FLApy/ForestIApy/FLApy/Datapool/demoLiDAR2021.las')
site.gen_SFL(bbox=[100, 200, 100, 200], resolution=1, obsType=3, udXSpacing=10, udYSpacing=10, udZNum=5)

# Calculate the LA
siteLA = fp.LAcalculator.LAcalculator(site)
siteLA.computeBatch(multiPro = 'p_map', CPU_count=6)

# Calculate the LAH
siteLAH = fp.LAHanalysis.LAH_analysis(siteLA)
result = siteLAH.com_allLAH()
print(result)
