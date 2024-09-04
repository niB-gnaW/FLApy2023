import FLApy as fp

site = fp.DataManagement.StudyFieldLattice()
site.read_LasData(r'D:\CodeSpace\FLApy2023\demo_Data\demoData.las')
fp.Visualization.vis_3Dpoint(site.point_cloud)