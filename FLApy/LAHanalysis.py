# -*- coding: utf-8 -*-
#---------------------------------------------------------------------#
#   FLApy: Forest Light Analyzer python package                       #
#   Developer: Wang Bin (Yunnan University, Kunming, China)           #

import numpy as np
import naturalneighbor
import pyvista as pv
import pandas as pd
import xarray as xr
import os
import miniball
import time
import open3d as o3d

from scipy.optimize import curve_fit
from scipy.spatial import ConvexHull
from scipy.stats import entropy
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from PVGeo.grids import ExtractTopography
from joblib import Parallel, delayed
from tqdm import tqdm




class LAH_analysis(object):
    # The class is used to calculate LAH index
    # Parameters:
    #   inGrid: the input grid data, that is the SFL data.
    #   fieldName: the field name of LA needed to be computed, default is 'SVF_flat'

    def __init__(self, inGrid = None, fieldName = 'SVF_flat'):

        if inGrid is None:
            raise ValueError('Please input a SFL data container!')

        elif inGrid is not None and os.path.isfile(inGrid) is False:
            self._inGrid = inGrid._DataContainer
            self.tempSFL = str(self._inGrid.field_data['temPath'][0])

        elif inGrid is not None and isinstance(inGrid, str) is True:
            self._inGrid = pv.read(inGrid)
            self.tempSFL = str(self._inGrid.field_data['temPath'][0])


        self.__obsType = self._inGrid.field_data['OBS_Type'][0]

        if self.__obsType == 0 or self.__obsType == 1 or self.__obsType == 3:
            self._valueImport = np.array(self._inGrid.field_data[fieldName])
            self._OBScoords = np.array(self._inGrid.field_data['OBS_SFL'])

        elif self.__obsType == 2:
            self._valueImport = np.array(self._inGrid.field_data['Given_Value'][0])
            self._OBScoords = np.array(self._inGrid.field_data['Given_Obs'])

        self._inGridWraped = self.interpolation_3D(self._inGrid, fieldName = fieldName)


        self._value = np.array(self._inGridWraped.cell_data[fieldName])
        self.allGridPoints = np.array(self._inGridWraped.cell_centers().points)

        self._x_bar = np.sum(self._value) / len(self._value)
        self._s_Star = np.sqrt((np.sum(self._value ** 2) / len(self._value)) - (self._x_bar ** 2))

        self.kdtree = KDTree(self.allGridPoints)
        self.gridSpacing =  self._inGridWraped.field_data['SFLset_resolution'][0]
        self.voxelArea = self.gridSpacing ** 2
        self.voxelVolume = self.gridSpacing ** 3

        self.fieldName = fieldName

        self._inGrid = self._inGridWraped

    def interpolation_3D(self, inGrid, fieldName = None):

        print('\033[34m' + 'Wraping to the 3D SFL...' + '\033[0m')

        pts_SFL = inGrid.points


        Xmin, Xmax = min(pts_SFL[:, 0]), max(pts_SFL[:, 0])
        Ymin, Ymax = min(pts_SFL[:, 1]), max(pts_SFL[:, 1])
        Zmin, Zmax = min(pts_SFL[:, 2]), max(pts_SFL[:, 2])

        resolutionGrid = int(inGrid.field_data['SFLset_resolution'])

        Granges = [[Xmin, Xmax, resolutionGrid], [Ymin, Ymax, resolutionGrid], [Zmin, Zmax, resolutionGrid]]
        inter = naturalneighbor.griddata(self._OBScoords, self._valueImport, Granges)

        tensorGrid = pv.UniformGrid()
        tensorGrid.dimensions = np.array(inter.shape) + 1
        tensorGrid.origin = (Xmin, Ymin, Zmin)
        tensorGrid.spacing = (resolutionGrid, resolutionGrid, resolutionGrid)
        tensorGrid.cell_data[fieldName] = inter.flatten(order='F')

        tensorGrid.field_data['PTS'] = inGrid.field_data['PTS']
        tensorGrid.field_data['OBS_SFL'] = inGrid.field_data['OBS_SFL']
        tensorGrid.field_data['DSM'] = inGrid.field_data['DSM']
        tensorGrid.field_data['DTM'] = inGrid.field_data['DTM']
        tensorGrid.field_data['DEM'] = inGrid.field_data['DEM']
        tensorGrid.field_data['SFLset_resolution'] = inGrid.field_data['SFLset_resolution']

        tensorGrid.field_data['PTS_cliped'] = inGrid.field_data['PTS_cliped']
        tensorGrid.field_data['DTM_cliped'] = inGrid.field_data['DTM_cliped']
        tensorGrid.field_data['DSM_cliped'] = inGrid.field_data['DSM_cliped']

        tensorGrid.field_data['temPath'] = inGrid.field_data['temPath']
        tensorGrid.field_data['OBS_Type'] = inGrid.field_data['OBS_Type']

        tensorGrid.add_field_data([str(inGrid.field_data['temPath'][0])], 'temPath')

        _dsm = np.array(tensorGrid.field_data['DSM'])
        dsm = _dsm[~np.isnan(_dsm[:, 2])]
        dsm = pv.PolyData(dsm)

        _dtm = np.array(tensorGrid.field_data['DTM'])
        dtm = _dtm[~np.isnan(_dtm[:, 2])]
        dtm = pv.PolyData(dtm)

        ext_dtm = ExtractTopography(invert=True).apply(tensorGrid, dtm)
        ext_dsm = ExtractTopography(invert=False).apply(tensorGrid, dsm)
        extMer = ext_dtm.cell_data['Extracted'] * ext_dsm.cell_data['Extracted']
        tensorGrid.cell_data['Classification'] = extMer
        tensorGrid.cell_data[fieldName] = extMer * tensorGrid.cell_data[fieldName]

        da = pd.DataFrame(_dtm, columns=["x", "y", "value"])
        pivoted_df = da.pivot(index="y", columns="x", values="value")
        da = xr.DataArray(pivoted_df)

        tensorGridCenterPts = np.array(tensorGrid.cell_centers().points)
        tgt_x = xr.DataArray(tensorGridCenterPts[:, 0], dims='points')
        tgt_y = xr.DataArray(tensorGridCenterPts[:, 1], dims='points')
        daquery = da.sel(x=tgt_x, y=tgt_y, method='nearest')
        zNormed = tensorGridCenterPts[:, 2] - daquery.data
        zNormed[zNormed < 0] = 0
        tensorGrid.cell_data['Z_normed'] = zNormed
        tensorGrid = tensorGrid.extract_cells(np.where(tensorGrid.cell_data['Classification'] == 1)[0])

        print('\033[34m' + 'Wraping to the 3D SFL is done!' + '\033[0m')

        return tensorGrid



    def voxel_SummarySta(self, thinning = 3):
        self._inGrid.LAH_Vox_average, self._inGrid.LAH_Vox_std, self._inGrid.LAH_Vox_CV, self._inGrid.LAH_Vox_Range = self.summarySta(self._value)

        coords = self.allGridPoints
        values = self._value
        self._inGrid.LAH_Vox_SAC_local, self._inGrid.LAH_Vox_SAC = self.cal_Moran(coords, values, ds=thinning)
        self._inGrid.LAH_Vox_Diversity = self.cal_Diversity(values)
        self._inGrid.LAH_Vox_Gini = self.cal_Gini(values)

    def cal_Diversity(self, values):
        time1 = time.time()
        print('Calculating Diversity Index ...')
        proportions = np.array(values) / np.sum(values)
        sdi = entropy(proportions, base=2)
        print('Diversity Index is calculated in {} seconds'.format(time.time() - time1))
        return sdi

    def cal_Gini(self, values):
        time1 = time.time()
        print('Calculating Gini Index ...')
        sorted_light_availability = sorted(values)
        n = len(values)
        _gini = (np.sum([(i + 1) * sorted_light_availability[i] for i in range(n)]) / (n * np.sum(sorted_light_availability))) - ((n + 1) / (n * 2))
        print('Gini Index is calculated in {} seconds'.format(time.time() - time1))
        return _gini

    def summarySta(self, values, per95 = True):
        values = np.array(values)
        _average = values.mean()
        _std = values.std()
        _CV = (_std / _average) * 100

        if per95:
            _range = np.quantile(values, 0.95) - np.quantile(values, 0.05)
        else:
            _range = np.max(values) - np.min(values)

        return _average, _std, _CV, _range
    def calculate_spatial_weights_matrix_idw(self, coords):
        n_points = len(coords)
        time1 = time.time()
        print('Number of observers: {}'.format(n_points) + '|Calculating Spatial Automatically Correlationship ...|' + ' Construct the KDTree ...')
        kdtree = KDTree(coords)
        distances, indices = kdtree.query(coords, k=n_points)
        print('KDTree was constructed in {} seconds'.format(time.time() - time1))
        row_indices = np.arange(n_points)[:, None]
        sorted_distances = distances[row_indices, indices.argsort()]
        sorted_distances = np.maximum(sorted_distances, 1e-6)
        W = 1 / sorted_distances
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1)
        W_normalized = W / row_sums[:, np.newaxis]
        print('Spatial Automatically Correlationship was calculated in {} seconds'.format(time.time() - time1))
        return W_normalized

    def calculate_spatial_weights_matrix_idw_LHS(self, coords, n_neighbors=None):
        n_points = len(coords)
        time1 = time.time()

        print('Number of observers: {}'.format(
            n_points) + '|Calculating Spatial Automatically Correlationship ...|' + ' Construct the NN ...')


        n_neighbors = n_points if n_neighbors is None else n_neighbors


        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
        nn.fit(coords)


        distances, indices = nn.kneighbors(coords)

        print('NN was constructed in {} seconds'.format(time.time() - time1))

        row_indices = np.arange(n_points)[:, None]
        sorted_distances = distances[row_indices, indices.argsort()]
        sorted_distances = np.maximum(sorted_distances, 1e-6)

        W = 1 / sorted_distances
        np.fill_diagonal(W, 0)
        row_sums = W.sum(axis=1)
        W_normalized = W / row_sums[:, np.newaxis]

        print('Spatial Automatically Correlationship was calculated in {} seconds'.format(time.time() - time1))
        return W_normalized


    def cal_Moran(self, coords, values, ds = 1):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        downsampled_pcd = pcd.voxel_down_sample(ds)
        kdtree = KDTree(coords)

        coordsd = np.asarray(downsampled_pcd.points)
        nn = kdtree.query(coordsd)
        nni = nn[1]
        valuesd = values[nni]

        W = self.calculate_spatial_weights_matrix_idw_LHS(coordsd)
        n = len(valuesd)
        mean_value = np.mean(valuesd)
        deviation = valuesd - mean_value
        num = np.sum(deviation * (W @ deviation))
        denom = np.sum(deviation ** 2)
        moran_i = n * num / (np.sum(W) * denom)
        l_moran_i = W @ deviation
        return l_moran_i, moran_i



    def normalize_array(self, arr):
        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val) * 100
        return normalized_arr


    def vertical_Summary(self):
        time1 = time.time()
        print('Calculating Vertical Light Attenuation ...')
        relativeHeight = np.array(self._inGrid.cell_data['Z_normed'])

        relativeHeight[relativeHeight < 0] = 0
        relativeHeight = relativeHeight
        _SVF = self.normalize_array(self._value)

        #_p0 = [np.max(_SVF), np.median(relativeHeight), 1, np.min(_SVF)]
        #_params, _params_covariance = curve_fit(sigmoid_func2, relativeHeight, _SVF, _p0, maxfev=99999)

        _params, _params_covariance = curve_fit(sigmoid_func, relativeHeight, _SVF, maxfev=99999)
        # Light attenuation rate
        self._inGrid.LAH_Ver_LAR = _params[0]
        # Height of the inflection point
        self._inGrid.LAH_Ver_HIP = _params[1]
        self._inGrid.LAH_Ver_HIPr = self._inGrid.LAH_Ver_HIP / np.max(relativeHeight)

        if _params[1] < 0:
            _p0 = [np.max(_SVF), np.median(relativeHeight), 1, np.min(_SVF)]
            _params, _params_covariance = curve_fit(sigmoid_func2, relativeHeight, _SVF, _p0, maxfev=99999)

            # Light attenuation rate
            self._inGrid.LAH_Ver_LAR = _params[2]
            # Height of the inflection point
            self._inGrid.LAH_Ver_HIP = _params[1]
            self._inGrid.LAH_Ver_HIPr = self._inGrid.LAH_Ver_HIP / np.max(relativeHeight)


        relativeHeight = self.normalize_array(relativeHeight)

        __points = np.vstack((relativeHeight, _SVF)).transpose()
        _hull = ConvexHull(__points)
        self._inGrid.LAH_Ver_ACH = _hull.volume
        print('Vertical Light Attenuation was calculated in {} seconds'.format(time.time() - time1))

    def vertical_ACH(self):

        pass

    def horizontal_Summary(self, givenHeight = 1.5):
        relativeHeight = np.array(self._inGrid.cell_data['Z_normed'])

        relativeHeight[relativeHeight < 0] = 0

        _SVF = self._value
        _OBS_SFL = self.allGridPoints
        tolerance = self.gridSpacing * 0.5
        _SVF_filterSet = _SVF[np.abs(relativeHeight - givenHeight) <= tolerance]
        _OBS_SFL_filterSet = _OBS_SFL[np.abs(relativeHeight - givenHeight) <= tolerance]

        self._inGrid.LAH_Hor_average, self._inGrid.LAH_Hor_std, self._inGrid.LAH_Hor_CV, self._inGrid.LAH_Hor_Range = self.summarySta(_SVF_filterSet)
        self._inGrid.LAH_Hor_SAC_local, self._inGrid.LAH_Hor_SAC = self.cal_Moran(_OBS_SFL_filterSet, _SVF_filterSet, ds = 1)
        self._inGrid.LAH_Hor_Diversity = self.cal_Diversity(_SVF_filterSet)
        self._inGrid.LAH_Hor_Gini = self.cal_Gini(_SVF_filterSet)


    def cluster3D_Summary(self, limiterMin = 27):
        self._inGrid.set_active_scalars('Gi_Value')
        self._inGrid.compute_cell_sizes()

        self.NumLandscape = len(self._value)

        _threshed_hotspot = self._inGrid.threshold(value=2.576, invert = False)
        _threshed_coldspot = self._inGrid.threshold(value=[-99998, -2.576])

        _bodiesHot = _threshed_hotspot.split_bodies()
        _bodiesCold = _threshed_coldspot.split_bodies()

        sumStaHot = []
        sumStaSVFHot = []


        for hot_key in tqdm(_bodiesHot.keys(), desc = 'Calculating Hotspot Statistics', ncols=100,total=len(_bodiesHot.keys())):
            oneBody_h = _bodiesHot[hot_key]
            if oneBody_h.n_cells > limiterMin:

                oneBody_h_volume = oneBody_h.n_cells * self.voxelVolume
                oneBody_h_Surface = oneBody_h.extract_geometry()
                oneBody_h_area = oneBody_h_Surface.n_cells * self.voxelArea
                oneHull_h = ConvexHull(oneBody_h.points)
                oneHull_h_pts = oneBody_h.points[oneHull_h.vertices]
                Chot, r2hot = miniball.get_bounding_ball(np.array(oneHull_h_pts))
                oneBody_h_SVF = oneBody_h.cell_data[self.fieldName]
                sumStaHot.append([oneBody_h_volume, oneBody_h_area, r2hot])
                sumStaSVFHot.extend(oneBody_h_SVF)

        sumStaCold = []
        sumStaSVFCold = []
        for cold_key in tqdm(_bodiesCold.keys(), desc = 'Calculating Coldspot Statistics', ncols=100,total=len(_bodiesCold.keys())):
            oneBody_c = _bodiesCold[cold_key]
            if oneBody_c.n_cells > limiterMin:

                oneBody_c_volume = oneBody_c.n_cells * self.voxelVolume
                oneBody_c_Surface = oneBody_c.extract_geometry()
                oneBody_c_area = oneBody_c_Surface.n_cells * self.voxelArea
                oneHull_c = ConvexHull(oneBody_c.points)
                oneHull_c_pts = oneBody_c.points[oneHull_c.vertices]
                Ccold, r2cold = miniball.get_bounding_ball(np.array(oneHull_c_pts))
                oneBody_c_SVF = oneBody_c.cell_data[self.fieldName]
                sumStaCold.append([oneBody_c_volume, oneBody_c_area, r2cold])
                sumStaSVFCold.extend(oneBody_c_SVF)

        self._inGrid.field_data['LAH_3Dcluster_Hot_SVF'] = np.array(sumStaSVFHot)
        self._inGrid.field_data['LAH_3Dcluster_Cold_SVF'] = np.array(sumStaSVFCold)
        sumStaHot = np.array(sumStaHot)
        sumStaCold = np.array(sumStaCold)

        self._inGrid.LAH_3Dcluster_Hot_Volume = np.sum(sumStaHot[:, 0])
        self._inGrid.LAH_3Dcluster_Cold_Volume = np.sum(sumStaCold[:, 0])

        self._inGrid.LAH_3Dcluster_Hot_Volume_relative = self._inGrid.LAH_3Dcluster_Hot_Volume / (self.NumLandscape * self.voxelVolume)
        self._inGrid.LAH_3Dcluster_Cold_Volume_relative = self._inGrid.LAH_3Dcluster_Cold_Volume / (self.NumLandscape * self.voxelVolume)

        self._inGrid.LAH_3Dcluster_VolumeRatio_Hot2Cold = self._inGrid.LAH_3Dcluster_Hot_Volume / self._inGrid.LAH_3Dcluster_Cold_Volume

        self._inGrid.LAH_3Dcluster_Hot_Largest_Volume = np.max(sumStaHot[:, 0])
        self._inGrid.LAH_3Dcluster_Cold_Largest_Volume = np.max(sumStaCold[:, 0])


        self._inGrid.LAH_3Dcluster_Hot_Abundance = len(sumStaHot[:, 0])
        self._inGrid.LAH_3Dcluster_Cold_Abundance = len(sumStaCold[:, 0])

        self._inGrid.LAH_3Dcluster_Hot_Volume_Numweight = self._inGrid.LAH_3Dcluster_Hot_Volume / self._inGrid.LAH_3Dcluster_Hot_Abundance
        self._inGrid.LAH_3Dcluster_Cold_Volume_Numweight = self._inGrid.LAH_3Dcluster_Cold_Volume / self._inGrid.LAH_3Dcluster_Cold_Abundance

        self._inGrid.LAH_3Dcluster_Hot_Cohesion = np.mean(self.cal_Cohesion(N = self.NumLandscape, P = sumStaHot[:, 1], A = sumStaHot[:, 0]))
        self._inGrid.LAH_3Dcluster_Cold_Cohesion = np.mean(self.cal_Cohesion(N = self.NumLandscape, P = sumStaCold[:, 1], A = sumStaCold[:, 0]))

        self._inGrid.LAH_3Dcluster_Hot_ShapeFactor = self.cal_shape_factor(sumStaHot[:, 0], sumStaHot[:, 1])
        self._inGrid.LAH_3Dcluster_Cold_ShapeFactor = self.cal_shape_factor(sumStaCold[:, 0], sumStaCold[:, 1])

        miniballHotVolume = self.cal_ShphericalVolume(sumStaHot[:, 2])
        miniballColdVolume = self.cal_ShphericalVolume(sumStaCold[:, 2])
        self._inGrid.LAH_3Dcluster_Hot_ShapeIndex = np.mean(sumStaHot[:, 0] / miniballHotVolume)
        self._inGrid.LAH_3Dcluster_Cold_ShapeIndex = np.mean(sumStaCold[:, 0] / miniballColdVolume)



    def cal_Cohesion(self,N, P, A):
        p = P / self.voxelArea
        a = A / self.voxelVolume

        sum_p = np.sum(p)
        sum_pa = np.sum(p * np.sqrt(a))
        pc = (1 - sum_p / sum_pa) * (1 - (1 / np.sqrt(N - 1))) ** -1
        return pc
    def cal_ShphericalVolume(self, r2):
        r = np.sqrt(r2)
        return (4/3) * np.pi * r**3
    def cal_shape_index(self, V):

        return

    def cal_shape_factor(self, V, A):
        return np.mean((36 * np.pi * V) / (A ** 2))

    def cluster3D_Connectivity(self):

        return



    def hotspotAnalysis(self, CPU_count = None):

        if CPU_count is None:
            numCPU = os.cpu_count() - 1
        else:
            numCPU = int(CPU_count)

        _gi_star_list = np.ones(self._inGrid.n_cells)
        _gi_star_list = _gi_star_list * -99999
        obsIDX = np.arange(len(self.obsIdxWithin))

        #gi_star_valuesSet = p_map(self.cal_Gix, obsIDX, num_cpus = numCPU, desc='Hot(cold) spot identifying', ncols=100)
        gi_star_valuesSet = Parallel(n_jobs=numCPU, verbose=10)(delayed(self.cal_Gix)(argLAH) for argLAH in obsIDX)

        gi_star_valuesSet = np.array(gi_star_valuesSet)
        _gi_star_list[self.obsIdxWithin] = gi_star_valuesSet

        self._inGrid.cell_data['Gi_Value'] = _gi_star_list
        #self._inGrid.cell_data['Z_Value'] = self.calculate_z_values(_gi_star_list)
        #self._inGrid.save(self.tempSFL)

    def calculate_z_values(self, G):
        mean_G = np.mean(G)
        std_G = np.std(G)
        Z = (G - mean_G) / std_G
        return Z
    def cal_Gix(self, obsIdx):
        onepLocation = self.allGridPoints[obsIdx]
        GIXdistances, GIXindices = self.kdtree.query(onepLocation, k=27)
        GIXdistances = np.where(GIXdistances == 0, 1, GIXdistances)
        _omega = 1. / GIXdistances

        ind_v = self._value[GIXindices]

        _A = np.sum(ind_v * _omega)
        _B = self._x_bar * np.sum(_omega)
        _C = (len(self._value) - 1) * (np.sum(_omega ** 2))
        _D = (np.sum(_omega)) ** 2
        _E = ((_C - _D) / (len(self._value) - 1)) ** 0.5
        _Gi_x = (_A - _B) / (self._s_Star * _E)
        return _Gi_x

    def cal_Gi_fast(self, coords, value, k=27):
        xyz = np.array(coords)
        values = np.array(value)

        kdtree = KDTree(xyz)
        time1 = time.time()
        print('Gi_KDTree build time: ')
        distances, nearest_neighbors_indices = kdtree.query(xyz, k=k)
        print('Gi_KDTree query time: ', time.time() - time1)
        inverse_distances = 1 / distances[:, 1:]

        # Getis-Ord Gi*
        n = len(xyz)
        mean_X = np.mean(values)
        S = np.std(values)


        numerator = np.array([np.sum(inverse_distances[i] * values[neighbors[1:]]) - mean_X * np.sum(inverse_distances[i]) for i, neighbors in tqdm(enumerate(nearest_neighbors_indices),
                                                                                                                                                    total=len(xyz),
                                                                                                                                                    desc='Calculating Gi*',
                                                                                                                                                    ncols=100)])


        w_square_sum = np.sum(inverse_distances ** 2, axis=1)
        denominator = S * np.sqrt((n * w_square_sum - np.sum(inverse_distances, axis=1) ** 2) / (n - 1))

        Gi_star = numerator / denominator
        return Gi_star

    def hotspotAnalysis_fast(self):
        coords = self.allGridPoints
        value = self._value
        k = 27
        time1 = time.time()
        gi_star_valuesSet = self.cal_Gi_fast(coords, value, k)
        print('Time for Gi_Value calculation: ', time.time() - time1, 's')
        self._inGrid.cell_data['Gi_Value'] = np.array(gi_star_valuesSet)
        #self._inGrid.cell_data['Z_Value'] = self.calculate_z_values(np.array(self._inGrid.cell_data['Gi_Value']))
        #self._inGrid.save(self.tempSFL)
        print('Time for Gi_Value saving: ', time.time() - time1, 's')


    def computeLAH(self, Voxel = True, Vertical = True, Horizontal = True, Cluster3D = True, save = None, thinning = 3, limit = 27):

        self._inGrid.LAH_Vox_average = 0
        self._inGrid.LAH_Vox_std = 0
        self._inGrid.LAH_Vox_CV = 0
        self._inGrid.LAH_Vox_Range = 0
        self._inGrid.LAH_Vox_SAC = 0
        self._inGrid.LAH_Vox_Diversity = 0
        self._inGrid.LAH_Vox_Gini = 0
        self._inGrid.LAH_Ver_LAR = 0
        self._inGrid.LAH_Ver_HIP = 0
        self._inGrid.LAH_Ver_HIPr = 0
        self._inGrid.LAH_Ver_ACH = 0
        self._inGrid.LAH_Hor_average = 0
        self._inGrid.LAH_Hor_std = 0
        self._inGrid.LAH_Hor_CV = 0
        self._inGrid.LAH_Hor_Range = 0
        self._inGrid.LAH_Hor_SAC = 0
        self._inGrid.LAH_Hor_Diversity = 0
        self._inGrid.LAH_Hor_Gini = 0
        self._inGrid.LAH_3Dcluster_Hot_Volume = 0
        self._inGrid.LAH_3Dcluster_Cold_Volume = 0
        self._inGrid.LAH_3Dcluster_Hot_Volume_relative = 0
        self._inGrid.LAH_3Dcluster_Cold_Volume_relative = 0
        self._inGrid.LAH_3Dcluster_VolumeRatio_Hot2Cold = 0
        self._inGrid.LAH_3Dcluster_Hot_Largest_Volume = 0
        self._inGrid.LAH_3Dcluster_Cold_Largest_Volume = 0
        self._inGrid.LAH_3Dcluster_Hot_Abundance = 0
        self._inGrid.LAH_3Dcluster_Cold_Abundance = 0
        self._inGrid.LAH_3Dcluster_Hot_Volume_Numweight = 0
        self._inGrid.LAH_3Dcluster_Cold_Volume_Numweight = 0
        self._inGrid.LAH_3Dcluster_Hot_Cohesion = 0
        self._inGrid.LAH_3Dcluster_Cold_Cohesion = 0
        self._inGrid.LAH_3Dcluster_Hot_ShapeFactor = 0
        self._inGrid.LAH_3Dcluster_Cold_ShapeFactor = 0
        self._inGrid.LAH_3Dcluster_Hot_ShapeIndex = 0
        self._inGrid.LAH_3Dcluster_Cold_ShapeIndex = 0

        if Voxel is True:
            print('\033[34m' + '----Calculating Voxel-scale Light Heterogeneity ...----' + '\033[0m')
            self.voxel_SummarySta(thinning=thinning)

        if Vertical is True:
            print('\033[34m' + '----Calculating Vertical-scale Light Heterogeneity ...----' + '\033[0m')
            self.vertical_Summary()

        if Horizontal is True:
            print('\033[34m' + '----Calculating Horizontal-scale Light Heterogeneity ...----' + '\033[0m')
            self.horizontal_Summary()

        if Cluster3D is True:
            print('\033[34m' + '----Calculating 3D-Cluster Light Heterogeneity ...----' + '\033[0m')
            self.hotspotAnalysis_fast()
            self.cluster3D_Summary(limiterMin=limit)

        dataSet = {'Indicators': ['Average', 'Standard_deviation', 'Coefficient_of_variation', 'Range',
                               'Spatial_autocorrelation', 'Diversity', 'Gini_coefficient', 'Light_attenuation_rate',
                               'Height_of_inflection_point', 'Relative_height_of_inflection_point', 'Convex_hull_area',
                               'Average', 'Standard_deviation', 'Coefficient_of_variation', 'Range',
                               'Spatial_autocorrelation', 'Diversity', 'Gini_coefficient', 'Hot_volume', 'Cold_volume',
                               'Relative_hot_volume', 'Relative_cold_volume', 'Volume_ratio_of_hot_to_cold',
                               'Largest_hot_volume', 'Largest_cold_volume', 'Hot_abundance', 'Cold_abundance',
                               'Hot_volume_average', 'Cold_volume_average', 'Hot_cohesion', 'Cold_cohesion',
                               'Hot_shape_factor', 'Cold_shape_factor', 'Hot_shape_index', 'Cold_shape_index'],
                'Scale': ['Voxel', 'Voxel', 'Voxel', 'Voxel', 'Voxel', 'Voxel', 'Voxel', 'Vertical', 'Vertical',
                          'Vertical', 'Vertical', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal',
                          'Horizontal', 'Horizontal', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster',
                          '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster',
                          '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster',
                          '3D_Cluster'],
                'Value': [self._inGrid.LAH_Vox_average,
                          self._inGrid.LAH_Vox_std,
                          self._inGrid.LAH_Vox_CV,
                          self._inGrid.LAH_Vox_Range,
                          self._inGrid.LAH_Vox_SAC,
                          self._inGrid.LAH_Vox_Diversity,
                          self._inGrid.LAH_Vox_Gini,
                          self._inGrid.LAH_Ver_LAR,
                          self._inGrid.LAH_Ver_HIP,
                          self._inGrid.LAH_Ver_HIPr,
                          self._inGrid.LAH_Ver_ACH,
                          self._inGrid.LAH_Hor_average,
                          self._inGrid.LAH_Hor_std,
                          self._inGrid.LAH_Hor_CV,
                          self._inGrid.LAH_Hor_Range,
                          self._inGrid.LAH_Hor_SAC,
                          self._inGrid.LAH_Hor_Diversity,
                          self._inGrid.LAH_Hor_Gini,
                          self._inGrid.LAH_3Dcluster_Hot_Volume,
                          self._inGrid.LAH_3Dcluster_Cold_Volume,
                          self._inGrid.LAH_3Dcluster_Hot_Volume_relative,
                          self._inGrid.LAH_3Dcluster_Cold_Volume_relative,
                          self._inGrid.LAH_3Dcluster_VolumeRatio_Hot2Cold,
                          self._inGrid.LAH_3Dcluster_Hot_Largest_Volume,
                          self._inGrid.LAH_3Dcluster_Cold_Largest_Volume,
                          self._inGrid.LAH_3Dcluster_Hot_Abundance,
                          self._inGrid.LAH_3Dcluster_Cold_Abundance,
                          self._inGrid.LAH_3Dcluster_Hot_Volume_Numweight,
                          self._inGrid.LAH_3Dcluster_Cold_Volume_Numweight,
                          self._inGrid.LAH_3Dcluster_Hot_Cohesion,
                          self._inGrid.LAH_3Dcluster_Cold_Cohesion,
                          self._inGrid.LAH_3Dcluster_Hot_ShapeFactor,
                          self._inGrid.LAH_3Dcluster_Cold_ShapeFactor,
                          self._inGrid.LAH_3Dcluster_Hot_ShapeIndex,
                          self._inGrid.LAH_3Dcluster_Cold_ShapeIndex
                          ],
                'Abbreviation': ['AVE_Vox', 'STD_Vox', 'CV_Vox', 'RAN_Vox', 'SAC_Vox', 'DIV_Vox', 'GINI_Vox', 'LAR_Ver',
                                 'HIP_Ver', 'HIPr_Ver', 'ACH_Ver', 'AVE_Hor', 'STD_Hor', 'CV_Hor', 'RAN_Hor', 'SAC_Hor',
                                 'DIV_Hor', 'GINI_Hor', 'HVOL_3D', 'CVOL_3D', 'HVOLr_3D', 'CVOLr_3D', 'VRH2C_3D',
                                 'LHV_3D', 'LCV_3D', 'HAB_3D', 'CAB_3D', 'HVA_3D', 'CVA_3D', 'HCO_3D', 'CCO_3D',
                                 'HSF_3D', 'CSF_3D', 'HSI_3D', 'CSI_3D']
                }

        self.indicatorCatalog = pd.DataFrame(dataSet)



        self._inGrid.field_data['AVE_Vox'] = self._inGrid.LAH_Vox_average
        self._inGrid.field_data['STD_Vox'] = self._inGrid.LAH_Vox_std
        self._inGrid.field_data['CV_Vox'] = self._inGrid.LAH_Vox_CV
        self._inGrid.field_data['RAN_Vox'] = self._inGrid.LAH_Vox_Range
        self._inGrid.field_data['SAC_Vox'] = self._inGrid.LAH_Vox_SAC
        self._inGrid.field_data['DIV_Vox'] = self._inGrid.LAH_Vox_Diversity
        self._inGrid.field_data['GINI_Vox'] = self._inGrid.LAH_Vox_Gini
        self._inGrid.field_data['LAR_Ver'] = self._inGrid.LAH_Ver_LAR
        self._inGrid.field_data['HIP_Ver'] = self._inGrid.LAH_Ver_HIP
        self._inGrid.field_data['HIPr_Ver'] = self._inGrid.LAH_Ver_HIPr
        self._inGrid.field_data['ACH_Ver'] = self._inGrid.LAH_Ver_ACH
        self._inGrid.field_data['AVE_Hor'] = self._inGrid.LAH_Hor_average
        self._inGrid.field_data['STD_Hor'] = self._inGrid.LAH_Hor_std
        self._inGrid.field_data['CV_Hor'] = self._inGrid.LAH_Hor_CV
        self._inGrid.field_data['RAN_Hor'] = self._inGrid.LAH_Hor_Range
        self._inGrid.field_data['SAC_Hor'] = self._inGrid.LAH_Hor_SAC
        self._inGrid.field_data['DIV_Hor'] = self._inGrid.LAH_Hor_Diversity
        self._inGrid.field_data['GINI_Hor'] = self._inGrid.LAH_Hor_Gini
        self._inGrid.field_data['HVOL_3D'] = self._inGrid.LAH_3Dcluster_Hot_Volume
        self._inGrid.field_data['CVOL_3D'] = self._inGrid.LAH_3Dcluster_Cold_Volume
        self._inGrid.field_data['HVOLr_3D'] = self._inGrid.LAH_3Dcluster_Hot_Volume_relative
        self._inGrid.field_data['CVOLr_3D'] = self._inGrid.LAH_3Dcluster_Cold_Volume_relative
        self._inGrid.field_data['VRH2C_3D'] = self._inGrid.LAH_3Dcluster_VolumeRatio_Hot2Cold
        self._inGrid.field_data['LHV_3D'] = self._inGrid.LAH_3Dcluster_Hot_Largest_Volume
        self._inGrid.field_data['LCV_3D'] = self._inGrid.LAH_3Dcluster_Cold_Largest_Volume
        self._inGrid.field_data['HAB_3D'] = self._inGrid.LAH_3Dcluster_Hot_Abundance
        self._inGrid.field_data['CAB_3D'] = self._inGrid.LAH_3Dcluster_Cold_Abundance
        self._inGrid.field_data['HVA_3D'] = self._inGrid.LAH_3Dcluster_Hot_Volume_Numweight
        self._inGrid.field_data['CVA_3D'] = self._inGrid.LAH_3Dcluster_Cold_Volume_Numweight
        self._inGrid.field_data['HCO_3D'] = self._inGrid.LAH_3Dcluster_Hot_Cohesion
        self._inGrid.field_data['CCO_3D'] = self._inGrid.LAH_3Dcluster_Cold_Cohesion
        self._inGrid.field_data['HSF_3D'] = self._inGrid.LAH_3Dcluster_Hot_ShapeFactor
        self._inGrid.field_data['CSF_3D'] = self._inGrid.LAH_3Dcluster_Cold_ShapeFactor
        self._inGrid.field_data['HSI_3D'] = self._inGrid.LAH_3Dcluster_Hot_ShapeIndex
        self._inGrid.field_data['CSI_3D'] = self._inGrid.LAH_3Dcluster_Cold_ShapeIndex


        if save is None:
            self._inGrid.save(self.tempSFL)
        else:
            self._inGrid.save(save)

        print('\033[42;97m' + '!!!LAH calculation finished!!!' + '\033[0m')
        return self.indicatorCatalog



def sigmoid_func(x, a, b):
    #
    return 100 / (1 + np.exp(-a * (x - b)))

def sigmoid_func2(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y












