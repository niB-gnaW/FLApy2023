import FLApy as fp
import pyvista as pv
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.gridspec as grd
import matplotlib.ticker as ticker

site1 = fp.DataManagement.dataInput('/Volumes/WangBinSSD/FLApy2023/FLApy2023/studyCase1/sub05_0499_LAHfin.vtk').read_VTK()
site2 = fp.DataManagement.dataInput('/Volumes/WangBinSSD/FLApy2023/FLApy2023/studyCase1/sub05_0703_LAHfin.vtk').read_VTK()
site3 = fp.DataManagement.dataInput('/Volumes/WangBinSSD/FLApy2023/FLApy2023/studyCase1/sub05_0902_LAHfin.vtk').read_VTK()

dataYsite1 = site1['SVF_flat']
dataXsite1 = site1['Z_normed']

dataYsite2 = site2['SVF_flat']
dataXsite2 = site2['Z_normed']

dataYsite3 = site3['SVF_flat']
dataXsite3 = site3['Z_normed']

def prepare_data(site, label, color):
    hotspotAll = site.field_data['LAH_3Dcluster_Hot_SVF'] * 100
    coldspotAll = site.field_data['LAH_3Dcluster_Cold_SVF'] * 100
    hotspotLabel = np.full((len(hotspotAll)), 'Hotspot')
    coldspotLabel = np.full((len(coldspotAll)), 'Coldspot')

    hotArray = np.column_stack(
        (hotspotAll, hotspotLabel, np.full(hotspotAll.shape, label), np.full(hotspotAll.shape, color)))
    coldArray = np.column_stack(
        (coldspotAll, coldspotLabel, np.full(coldspotAll.shape, label), np.full(coldspotAll.shape, color)))

    pdata = np.vstack((hotArray, coldArray))
    return pd.DataFrame({'SVF': pdata[:, 0].astype(float),
                         'Type': pdata[:, 1],
                         'Site': pdata[:, 2],
                         'Color': pdata[:, 3]})


fig = plt.figure()
gs = grd.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1], hspace=0.3, wspace=0.1)
site1Color = '#00BEFF'
site2Color = '#F45581'
site3Color = '#569426'

ax1 = plt.subplot(gs[0, 0])
sns.kdeplot((dataYsite1 * 100), shade = True, common_norm = True, color = site1Color, lw=2)
sns.kdeplot((dataYsite2 * 100), shade = True, common_norm = True, color = site2Color, lw=2)
sns.kdeplot((dataYsite3 * 100), shade = True, common_norm = True, color = site3Color, lw=2)
ax1.set_xlabel('LA (%)')
ax1.set_xlim(0, 100)


ax2 = plt.subplot(gs[0, 1])
X_vertical_site1 = site1.field_data['Z_normed_full']
Y_vertical_site1 = site1.field_data['SVF_flat_full'] * 100
x_range_site1 = np.linspace(0, np.max(X_vertical_site1), 100)
_params_site1 = [site1.field_data['LAR_Ver'], site1.field_data['HIP_Ver']]
X_vertical_site2 = site2.field_data['Z_normed_full']
Y_vertical_site2 = site2.field_data['SVF_flat_full'] * 100
x_range_site2 = np.linspace(0, np.max(X_vertical_site2), 100)
_params_site2 = [site2.field_data['LAR_Ver'], site2.field_data['HIP_Ver']]
X_vertical_site3 = site3.field_data['Z_normed_full']
Y_vertical_site3 = site3.field_data['SVF_flat_full'] * 100
x_range_site3 = np.linspace(0, np.max(X_vertical_site3), 100)
_params_site3 = [site3.field_data['LAR_Ver'], site3.field_data['HIP_Ver']]

#sns.scatterplot(y = X_vertical_site1, x = Y_vertical_site1, alpha=0.01, edgecolor='none', s = 1, color = site1Color)
sns.lineplot(y = x_range_site1, x = fp.LAHanalysis.sigmoid_func(x_range_site1, *_params_site1), color = site1Color, lw=2)
#sns.scatterplot(y = X_vertical_site2, x = Y_vertical_site2, alpha=0.01, edgecolor='none', s = 1, color = site2Color)
sns.lineplot(y = x_range_site2, x = fp.LAHanalysis.sigmoid_func(x_range_site2, *_params_site2), color = site2Color, lw=2)
#sns.scatterplot(y = X_vertical_site3, x = Y_vertical_site3, alpha=0.01, edgecolor='none', s = 1, color = site3Color)
sns.lineplot(y = x_range_site3, x = fp.LAHanalysis.sigmoid_func(x_range_site3, *_params_site3), color = site3Color, lw=2)

ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")
ax2.set_ylabel('Relative height (m)')
ax2.set_xlabel('LA (%)')
ax2.set_xlim(0, 100)

ax3 = plt.subplot(gs[1, 0])
Y_horizontal_site1 = (dataYsite1[np.abs(dataXsite1 - 1.5) <= 0.5]) * 100
Y_horizontal_site2 = (dataYsite2[np.abs(dataXsite2 - 1.5) <= 0.5]) * 100
Y_horizontal_site3 = (dataYsite3[np.abs(dataXsite3 - 1.5) <= 0.5]) * 100

sns.kdeplot(Y_horizontal_site1, shade=True, common_norm = True, color = site1Color, lw=2)
sns.kdeplot(Y_horizontal_site2, shade=True, common_norm = True, color = site2Color, lw=2)
sns.kdeplot(Y_horizontal_site3, shade=True, common_norm = True, color = site3Color, lw=2)
ax3.set_xlabel('LA (%)')
ax3.set_xlim(0, 30)
ax3.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))

ax4 = plt.subplot(gs[1, 1])

'''''
hotspotAll_site1 = site1.field_data['LAH_3Dcluster_Hot_SVF'] * 100
coldspotAll_site1 = site1.field_data['LAH_3Dcluster_Cold_SVF'] * 100
hotspotLabel_site1 = np.full((len(hotspotAll_site1)), 'Hotspot')
coldspotLabel_site1 = np.full((len(coldspotAll_site1)), 'Coldspot')
hotArray_site1 = np.vstack((hotspotAll_site1, hotspotLabel_site1)).T
coldArray_site1 = np.vstack((coldspotAll_site1, coldspotLabel_site1)).T
pdata_site1 = np.vstack((hotArray_site1, coldArray_site1))
pdata2_site1 = pd.DataFrame({'SVF': np.array(pdata_site1[:, 0], dtype=float), 'Type': np.array(pdata_site1[:, 1], dtype=str)})
sns.boxplot(data=pdata2_site1, x='SVF', y='Type', color = site1Color)

hotspotAll_site2 = site2.field_data['LAH_3Dcluster_Hot_SVF'] * 100
coldspotAll_site2 = site2.field_data['LAH_3Dcluster_Cold_SVF'] * 100
hotspotLabel_site2 = np.full((len(hotspotAll_site2)), 'Hotspot')
coldspotLabel_site2 = np.full((len(coldspotAll_site2)), 'Coldspot')
hotArray_site2 = np.vstack((hotspotAll_site2, hotspotLabel_site2)).T
coldArray_site2 = np.vstack((coldspotAll_site2, coldspotLabel_site2)).T
pdata_site2 = np.vstack((hotArray_site2, coldArray_site2))
pdata2_site2 = pd.DataFrame({'SVF': np.array(pdata_site2[:, 0], dtype=float), 'Type': np.array(pdata_site2[:, 1], dtype=str)})
sns.boxplot(data=pdata2_site2, x='SVF', y='Type', color = site2Color)

hotspotAll_site3 = site3.field_data['LAH_3Dcluster_Hot_SVF'] * 100
coldspotAll_site3 = site3.field_data['LAH_3Dcluster_Cold_SVF'] * 100
hotspotLabel_site3 = np.full((len(hotspotAll_site3)), 'Hotspot')
coldspotLabel_site3 = np.full((len(coldspotAll_site3)), 'Coldspot')
hotArray_site3 = np.vstack((hotspotAll_site3, hotspotLabel_site3)).T
coldArray_site3 = np.vstack((coldspotAll_site3, coldspotLabel_site3)).T
pdata_site3 = np.vstack((hotArray_site3, coldArray_site3))
pdata2_site3 = pd.DataFrame({'SVF': np.array(pdata_site3[:, 0], dtype=float), 'Type': np.array(pdata_site3[:, 1], dtype=str)})
sns.boxplot(data=pdata2_site3, x='SVF', y='Type', color = site3Color)
'''''

df = pd.concat([prepare_data(site1, 'Site1', site1Color),
                prepare_data(site2, 'Site2', site2Color),
                prepare_data(site3, 'Site3', site3Color)])

sns.boxplot(data=df, x='SVF', y='Type', hue='Site', palette=[site1Color, site2Color, site3Color])
#sns.violinplot(data=df, x='SVF', y='Type', hue='Site', palette=[site1Color, site2Color, site3Color], inner=None, scale='count', width=0.5, alpha=0.3)

ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.set_xlabel('LA (%)')
ax4.set_ylabel('Zone')
ax4.set_xlim(0, 100)
ax4.get_legend().remove()


plt.show()
fig.savefig('/Users/wangbin/Documents/文章/准备中的文章/FLApy/Data/2023/Pics/Figure4sub1.png', dpi=300, bbox_inches='tight')



'''''
site = fp.DataManagement.dataInput('/Users/wangbin/PythonSpace/PythonEX/FLApy/FLApy2023/tests/SimForestStandardDis10Numtree100Sub05_FIN.vtk').read_VTK()
fp.Visualization.vis_Figures(site, field='SVF_flat')
'''''


'''''
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






