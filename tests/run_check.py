import FLApy as fp

site = fp.DataManagement.StudyFieldLattice()
site.read_LasData(r'D:\CodeSpace\FLApy2023\demo_Data\demoData.las')
site.gen_SFL(bbox=[100, 200, 100, 200], resolution=1, obsType=3, udXSpacing=50, udYSpacing=50, udZNum=2)

siteLA = fp.LAcalculator.LAcalculator(site)
siteLA.computeBatch(CPU_count=30)

siteLAH = fp.LAHanalysis.LAH_analysis(siteLA)
result = siteLAH.com_allLAH()
print(result)

fp.Visualization.vis_Figures(siteLAH)