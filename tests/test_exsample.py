import FLApy as fp


# Initialize the study field lattice and read the LiDAR data
site = fp.DataManagement.StudyFieldLattice()
site.read_LasData('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/demo_Data/demoData.las')
site.gen_SFL(bbox=[100, 200, 100, 200], resolution=1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=2)

# Calculate the LA
siteLA = fp.LAcalculator.LAcalculator(site)
siteLA.computeBatch(multiPro = 'joblib', CPU_count=4)

# Calculate the LAH
siteLAH = fp.LAHanalysis.LAH_analysis(siteLA)
result = siteLAH.com_allLAH()
print(result)

# Visualize the LAH
fp.Visualization.vis_Figures(siteLAH)