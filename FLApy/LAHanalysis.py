# -*- coding: utf-8 -*-


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
            a = hasattr(inGrid, '_SFL')
            if hasattr(inGrid, '_SFL') is True:
                self._inGrid = inGrid._SFL
            if hasattr(inGrid, '_DataContainer') is True:
                self._inGrid = inGrid._DataContainer
            if hasattr(inGrid, '_SFL') is False and hasattr(inGrid, '_DataContainer') is False:
                self._inGrid = inGrid

            self.tempSFL = str(self._inGrid.field_data['temPath'][0])

        elif inGrid is not None and isinstance(inGrid, str) is True:
            self._inGrid = pv.read(inGrid)
            self.tempSFL = str(self._inGrid.field_data['temPath'][0])


        self.__obsType = self._inGrid.field_data['OBS_Type'][0]


        print('\033[34m' + 'The calculation of LAH is started! The working directory is: ' + os.getcwd() + '\033[0m')


        if self.__obsType == 0 or self.__obsType == 1 or self.__obsType == 3:
            self._valueImport = np.array(self._inGrid.field_data[fieldName])
            self._OBScoords = np.array(self._inGrid.field_data['OBS_SFL'])

            #self._valueImport = np.array(self._inGrid.cell_data[fieldName])
            #self._OBScoords = np.array(self._inGrid.cell_centers().points)

        elif self.__obsType == 2:
            self._valueImport = np.array(self._inGrid.field_data['Given_Value'])
            self._OBScoords = np.array(self._inGrid.field_data['OBS_SFL'])

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
        self._inGrid_copy = self._inGridWraped.copy()

    def interpolation_3D(self, inGrid, fieldName = None):
        # This function is used to interpolate to 3D SFL. The interpolation method is 3D natural neighbor interpolation.
        # Parameters:
        #   inGrid: the input grid data, that is the SFL data.
        #   fieldName: the field name of LA needed to be computed, default is 'SVF_flat'
        # Return:
        #   tensorGrid: the 3D SFL data

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
        tensorGrid.field_data[str(fieldName + '_full')] = inter.flatten(order='F')
        tensorGrid.field_data[fieldName] = inter.flatten(order='F')
        tensorGrid.field_data['PTS'] = inGrid.field_data['PTS']
        tensorGrid.field_data['OBS_SFL'] = inGrid.field_data['OBS_SFL']
        tensorGrid.field_data['DSM'] = inGrid.field_data['DSM']
        tensorGrid.field_data['DTM'] = inGrid.field_data['DTM']
        tensorGrid.field_data['DEM'] = inGrid.field_data['DEM']
        tensorGrid.field_data['SFLset_resolution'] = inGrid.field_data['SFLset_resolution']
        tensorGrid.field_data['coords_full'] = np.array(tensorGrid.cell_centers().points)
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
        self.da = xr.DataArray(pivoted_df)

        tensorGridCenterPts = np.array(tensorGrid.cell_centers().points)
        tgt_x = xr.DataArray(tensorGridCenterPts[:, 0], dims='points')
        tgt_y = xr.DataArray(tensorGridCenterPts[:, 1], dims='points')
        daquery = self.da.sel(x=tgt_x, y=tgt_y, method='nearest')
        zNormed = tensorGridCenterPts[:, 2] - daquery.data
        zNormed[zNormed < 0] = 0
        tensorGrid.cell_data['Z_normed'] = zNormed
        tensorGrid.field_data['Z_normed_full'] = zNormed

        tensorGrid = tensorGrid.extract_cells(np.where(tensorGrid.cell_data['Classification'] == 1)[0])

        print('\033[34m' + 'Wraping to the 3D SFL is done!' + '\033[0m')

        return tensorGrid



    def voxel_SummarySta(self, thinning = 3, per_95 = True):
        # This function is used to calculate the summary statistics of the given values.
        # Parameters:
        #   thinning: the thinning factor, default is 3.
        #   per_95: whether to calculate the 95% range of the given values, default is True.

        coords = self._allGridPoints_clip
        values = self._value_clip

        self.LAH_Vox_average, self.LAH_Vox_std, self.LAH_Vox_CV, self.LAH_Vox_Range = self.summarySta(values, per95 = per_95)
        self.LAH_Vox_SAC_local, self.LAH_Vox_SAC = self.cal_Moran(coords, values, ds=thinning)
        self.LAH_Vox_Diversity = self.cal_Diversity(values)
        self.LAH_Vox_Gini = self.cal_Gini(values)

    @staticmethod
    def cal_Diversity(values, bins = 10):
        # This function is used to calculate the diversity index of the given values. All values will be divided into
        # several bins, and the diversity index will be calculated based on the proportion of each bin.
        # Parameters:
        #   values: the given values
        #   bins: the number of bins, default is 10.
        # Return:
        #   sdi: the diversity index of the given values


        hist, bin_edges = np.histogram(values, bins=bins, density=True)
        proportions = hist * np.diff(bin_edges)
        sdi = entropy(proportions, base=2)

        return sdi

    @staticmethod
    def cal_Gini(values):
        # This function is used to calculate the Gini index of the given values.
        # Parameters:
        #   values: the given values
        # Return:
        #   _gini: the Gini index of the given values


        sorted_light_availability = sorted(values)
        n = len(values)
        _gini = (np.sum([(i + 1) * sorted_light_availability[i] for i in range(n)]) / (n * np.sum(sorted_light_availability))) - ((n + 1) / (n * 2))

        return _gini

    @staticmethod
    def summarySta(values, per95 = True):
        # This function is used to calculate the summary statistics of the given values.
        # Parameters:
        #   values: the given values
        #   per95: whether to calculate the 95% range of the given values, default is True.
        # Return:
        #   _average: the average of the given values
        #   _std: the standard deviation of the given values
        #   _CV: the coefficient of variation of the given values
        #   _range: the range of the given values

        values = np.array(values)
        _average = values.mean()
        _std = values.std()
        _CV = (_std / _average) * 100

        if per95:
            _range = np.quantile(values, 0.95) - np.quantile(values, 0.05)
        else:
            _range = np.max(values) - np.min(values)

        return _average, _std, _CV, _range

    @staticmethod
    def calculate_spatial_weights_matrix_idw_LHS(coords, n_neighbors=None):
        # This function is used to calculate the spatial weights matrix based on the inverse distance weighting method.
        # Parameters:
        #   coords: the coordinates of the given values
        #   n_neighbors: the number of neighbors, default is None.
        # Return:
        #   W_normalized: the spatial weights matrix

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
        # This function is used to calculate the spatial autocorrelation of the given values.
        # Parameters:
        #   coords: the coordinates of the given values
        #   values: the given values
        #   ds: the downsample rate, default is 1.
        # Return:
        #   l_moran_i: the local spatial autocorrelation of the given values
        #   moran_i: the global spatial autocorrelation of the given values

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


    @staticmethod
    def normalize_array(arr):
        # This function is used to normalize the given values.
        # Parameters:
        #   arr: the given values
        # Return:
        #   normalized_arr: the normalized values

        arr = np.array(arr)
        min_val = np.min(arr)
        max_val = np.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val) * 100
        return normalized_arr


    def vertical_Summary(self, mode = 'c'):
        # This function is used to calculate the vertical LAH. All the parameters will be calculated and stored in the
        # input grid data.
        # Parameters:
        #   mode: the mode of the calculation, 'v' means voxel sampling, 'c' means column sampling.
        # Return:
        #   None

        time1 = time.time()
        print('Calculating Light Availability Heterogeneity using vertical sampling ...')

        relativeHeight = self._relativeHeight_Full_clip
        relativeHeight[relativeHeight < 0] = 0

        _SVF = self.normalize_array(self._value_Full_clip)

        if mode == 'v':
            _params, _params_covariance = curve_fit(sigmoid_func, relativeHeight, _SVF, maxfev=99999)

            # Light attenuation rate
            self.LAH_Ver_LAR = _params[0]
            # Height of the inflection point
            self.LAH_Ver_HIP = _params[1]
            self.LAH_Ver_HIPr = self.LAH_Ver_HIP / np.max(relativeHeight)

            if _params[1] < 0:
                _p0 = [np.max(_SVF), np.median(relativeHeight), 1, np.min(_SVF)]
                _params, _params_covariance = curve_fit(sigmoid_func2, relativeHeight, _SVF, _p0, maxfev=99999)

                # Light attenuation rate
                self.LAH_Ver_LAR = _params[2]
                # Height of the inflection point
                self.LAH_Ver_HIP = _params[1]
                self.LAH_Ver_HIPr = self.LAH_Ver_HIP / np.max(relativeHeight)


        elif mode == 'c':
            _coords = self.allGridPoints_Full_clip

            XYmap = np.meshgrid(np.unique(_coords[:, 0]), np.unique(_coords[:, 1]))
            _params_Pool = []
            _Xflat = XYmap[0].flatten()
            _Yflat = XYmap[1].flatten()

            for pixel_wise in tqdm(range(len(_Xflat)), desc = 'Fitting Sigmoid Function...', ncols=100,total=len(_Xflat)):
                _coords_index = np.where((_coords[:, 0] == _Xflat[pixel_wise]) & (_coords[:, 1] == _Yflat[pixel_wise]))
                _SVF_selected = _SVF[_coords_index]
                _relativeHeight_selected = relativeHeight[_coords_index]
                _params, _params_covariance = curve_fit(sigmoid_func, _relativeHeight_selected, _SVF_selected, maxfev=99999)
                _params_Pool.append(_params)

            _params_Pool = np.array(_params_Pool)
            self.LAH_Ver_LAR = np.mean(_params_Pool[:, 0])
            self.LAH_Ver_HIP = np.mean(_params_Pool[:, 1])
            self.LAH_Ver_HIPr = self.LAH_Ver_HIP / np.max(relativeHeight)

        relativeHeight = self.normalize_array(relativeHeight)
        __points = np.vstack((relativeHeight, _SVF)).transpose()
        _hull = ConvexHull(__points)
        self.LAH_Ver_ACH = _hull.volume
        print('Vertical LAH was calculated in {} seconds'.format(time.time() - time1))



    def horizontal_Summary(self, givenHeight = None):
        # This function is used to calculate the horizontal LAH. All the parameters will be calculated and stored in the
        # input grid data.
        # Parameters:
        #   givenHeight: the given height, default is np.array([1, 10, 20, 30]).

        time1 = time.time()
        print('Calculating Light Availability using horizontal sampling ...')
        relativeHeight = self._relativeHeight_Full_clip
        relativeHeight[relativeHeight < 0] = 0

        _SVF = self._value_Full_clip
        _OBS_SFL = self.allGridPoints_Full_clip


        _X = _OBS_SFL[:, 0]
        _Y = _OBS_SFL[:, 1]
        _Z = relativeHeight
        _coordsNorm = np.vstack((_X, _Y, _Z)).transpose()
        _kdTreeNorm = KDTree(_coordsNorm)

        XYmap = np.meshgrid(np.unique(_X), np.unique(_Y))
        _Xflat = XYmap[0].flatten()
        _Yflat = XYmap[1].flatten()
        _ZflatAll = np.arange(np.min(relativeHeight), np.max(relativeHeight), self.gridSpacing)


        if givenHeight is None:

            LHA = []
            LHS = []
            LHC = []
            LHR = []
            LHSAC = []
            LHDiv = []
            LHGini = []


            for tensorHeight in tqdm(_ZflatAll, desc = 'Calculating Horizontal LAH...', ncols=100,total=len(_ZflatAll)):
                _Zflat = np.full(len(_Xflat), tensorHeight)
                _queryPoints = np.vstack((_Xflat, _Yflat, _Zflat)).transpose()
                _, idx = _kdTreeNorm.query(_queryPoints, k=1)
                extracted = _SVF[idx]
                _tgt_X = xr.DataArray(_Xflat, dims='points')
                _tgt_Y = xr.DataArray(_Yflat, dims='points')
                daQuery = self.da.sel(x=_tgt_X, y=_tgt_Y, method='nearest')
                _Z_fixed = _Zflat + daQuery.data
                extracted_loc = np.vstack((_Xflat, _Yflat, _Z_fixed)).transpose()

                _average, _std, _CV, _range = self.summarySta(extracted)
                LHA.append(_average)
                LHS.append(_std)
                LHC.append(_CV)
                LHR.append(_range)
                _, _lhSAC = self.cal_Moran(extracted_loc, extracted, ds = 1)
                LHSAC.append(_lhSAC)
                _lhDiv = self.cal_Diversity(extracted)
                LHDiv.append(_lhDiv)
                _lhGini = self.cal_Gini(extracted)
                LHGini.append(_lhGini)

            self.LAH_Hor_average = np.mean(np.array(LHA))
            self.LAH_Hor_average_maxHLH_height = _ZflatAll[np.argmax(np.array(LHA))]
            self.LAH_Hor_std = np.mean(np.array(LHS))
            self.LAH_Hor_std_maxHLH_height = _ZflatAll[np.argmax(np.array(LHS))]
            self.LAH_Hor_std_maxHLH_rheight = self.LAH_Hor_std_maxHLH_height / np.max(relativeHeight)
            self.LAH_Hor_CV = np.mean(np.array(LHC))
            self.LAH_Hor_CV_maxHLH_height = _ZflatAll[np.argmax(np.array(LHC))]
            self.LAH_Hor_Range = np.mean(np.array(LHR))
            self.LAH_Hor_Range_maxHLH_height = _ZflatAll[np.argmax(np.array(LHR))]
            self.LAH_Hor_SAC = np.mean(np.array(LHSAC))
            self.LAH_Hor_SAC_maxHLH_height = _ZflatAll[np.argmax(np.array(LHSAC))]
            self.LAH_Hor_Diversity = np.mean(np.array(LHDiv))
            self.LAH_Hor_Diversity_maxHLH_height = _ZflatAll[np.argmax(np.array(LHDiv))]
            self.LAH_Hor_Gini = np.mean(np.array(LHGini))
            self.LAH_Hor_Gini_maxHLH_height = _ZflatAll[np.argmax(np.array(LHGini))]

            self._inGrid.field_data['givenHeight'] = _ZflatAll
            self._inGrid.field_data['LHS_multiHeight'] = np.array(LHS)



        if givenHeight is not None and isinstance(givenHeight, float):
            _Zflat = np.full(len(_Xflat), givenHeight)
            _queryPoints = np.vstack((_Xflat, _Yflat, _Zflat)).transpose()
            _, idx = _kdTreeNorm.query(_queryPoints, k=1)
            extracted = _SVF[idx]
            _tgt_X = xr.DataArray(_Xflat, dims='points')
            _tgt_Y = xr.DataArray(_Yflat, dims='points')
            daQuery = self.da.sel(x=_tgt_X, y=_tgt_Y, method='nearest')
            _Z_fixed = _Zflat + daQuery.data
            extracted_loc = np.vstack((_Xflat, _Yflat, _Z_fixed)).transpose()

            self.LAH_Hor_average, self.LAH_Hor_std, self.LAH_Hor_CV, self.LAH_Hor_Range = self.summarySta(extracted)
            self.LAH_Hor_SAC_local, self.LAH_Hor_SAC = self.cal_Moran(extracted_loc, extracted, ds = 1)
            self.LAH_Hor_Diversity = self.cal_Diversity(extracted)
            self.LAH_Hor_Gini = self.cal_Gini(extracted)




        if givenHeight is not None and isinstance(givenHeight, list):
            LHA = []
            LHS = []
            LHC = []
            LHR = []
            LHSAC = []
            LHDiv = []
            LHGini = []
            for tensorHeight in tqdm(givenHeight, desc = 'Calculating Horizontal LAH...', ncols=100,total=len(givenHeight)):
                _Zflat = np.full(len(_Xflat), tensorHeight)
                _queryPoints = np.vstack((_Xflat, _Yflat, _Zflat)).transpose()
                _, idx = _kdTreeNorm.query(_queryPoints, k=1)
                extracted = _SVF[idx]
                _tgt_X = xr.DataArray(_Xflat, dims='points')
                _tgt_Y = xr.DataArray(_Yflat, dims='points')
                daQuery = self.da.sel(x=_tgt_X, y=_tgt_Y, method='nearest')
                _Z_fixed = _Zflat + daQuery.data
                extracted_loc = np.vstack((_Xflat, _Yflat, _Z_fixed)).transpose()

                _average, _std, _CV, _range = self.summarySta(extracted)
                LHA.append(_average)
                LHS.append(_std)
                LHC.append(_CV)
                LHR.append(_range)
                _, _lhSAC = self.cal_Moran(extracted_loc, extracted, ds = 1)
                LHSAC.append(_lhSAC)
                _lhDiv = self.cal_Diversity(extracted)
                LHDiv.append(_lhDiv)
                _lhGini = self.cal_Gini(extracted)
                LHGini.append(_lhGini)

            self.LAH_Hor_average = np.mean(np.array(LHA))
            self.LAH_Hor_average_maxHLH_height = givenHeight[np.argmax(np.array(LHA))]
            self.LAH_Hor_std = np.mean(np.array(LHS))
            self.LAH_Hor_std_maxHLH_height = givenHeight[np.argmax(np.array(LHS))]
            self.LAH_Hor_std_maxHLH_rheight = self.LAH_Hor_std_maxHLH_height / np.max(relativeHeight)
            self.LAH_Hor_CV = np.mean(np.array(LHC))
            self.LAH_Hor_CV_maxHLH_height = givenHeight[np.argmax(np.array(LHC))]
            self.LAH_Hor_Range = np.mean(np.array(LHR))
            self.LAH_Hor_Range_maxHLH_height = givenHeight[np.argmax(np.array(LHR))]
            self.LAH_Hor_SAC = np.mean(np.array(LHSAC))
            self.LAH_Hor_SAC_maxHLH_height = givenHeight[np.argmax(np.array(LHSAC))]
            self.LAH_Hor_Diversity = np.mean(np.array(LHDiv))
            self.LAH_Hor_Diversity_maxHLH_height = givenHeight[np.argmax(np.array(LHDiv))]
            self.LAH_Hor_Gini = np.mean(np.array(LHGini))
            self.LAH_Hor_Gini_maxHLH_height = givenHeight[np.argmax(np.array(LHGini))]

            self._inGrid.field_data['givenHeight'] = givenHeight
            self._inGrid.field_data['LHS_multiHeight'] = np.array(LHS)





        print('Horizontal LAH was calculated in {} seconds'.format(time.time() - time1))


    def cluster3D_Summary(self, limiterMin = 27):
        # This function is used to calculate the 3D cluster LAH. All the parameters will be calculated and stored in the
        # input grid data.
        # Parameters:
        #   limiterMin: the minimum number of voxels in a cluster, default is 27.
        # Return:
        #   None

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

        self.LAH_3Dcluster_Hot_Volume = np.sum(sumStaHot[:, 0])
        self.LAH_3Dcluster_Cold_Volume = np.sum(sumStaCold[:, 0])

        self.LAH_3Dcluster_Hot_Volume_relative = self.LAH_3Dcluster_Hot_Volume / (self.NumLandscape * self.voxelVolume)
        self.LAH_3Dcluster_Cold_Volume_relative = self.LAH_3Dcluster_Cold_Volume / (self.NumLandscape * self.voxelVolume)

        self.LAH_3Dcluster_VolumeRatio_Hot2Cold = self.LAH_3Dcluster_Hot_Volume / self.LAH_3Dcluster_Cold_Volume

        self.LAH_3Dcluster_Hot_Largest_Volume = np.max(sumStaHot[:, 0])
        self.LAH_3Dcluster_Cold_Largest_Volume = np.max(sumStaCold[:, 0])
        self.LAH_3Dcluster_Hot_Largest_Volume_index = (self.LAH_3Dcluster_Hot_Largest_Volume / (self.NumLandscape * self.voxelVolume)) * 100
        self.LAH_3Dcluster_Cold_Largest_Volume_index = (self.LAH_3Dcluster_Cold_Largest_Volume / (self.NumLandscape * self.voxelVolume)) * 100

        self.LAH_3Dcluster_Hot_Abundance = len(sumStaHot[:, 0])
        self.LAH_3Dcluster_Cold_Abundance = len(sumStaCold[:, 0])

        self.LAH_3Dcluster_Hot_Volume_Numweight = self.LAH_3Dcluster_Hot_Volume / self.LAH_3Dcluster_Hot_Abundance
        self.LAH_3Dcluster_Cold_Volume_Numweight = self.LAH_3Dcluster_Cold_Volume / self.LAH_3Dcluster_Cold_Abundance

        self.LAH_3Dcluster_Hot_Cohesion = np.mean(self.cal_Cohesion(N = self.NumLandscape, P = sumStaHot[:, 1], A = sumStaHot[:, 0]))
        self.LAH_3Dcluster_Cold_Cohesion = np.mean(self.cal_Cohesion(N = self.NumLandscape, P = sumStaCold[:, 1], A = sumStaCold[:, 0]))

        self.LAH_3Dcluster_Hot_ShapeIndex = self.cal_shape_index(sumStaHot[:, 0], sumStaHot[:, 1])
        self.LAH_3Dcluster_Cold_ShapeIndex = self.cal_shape_index(sumStaCold[:, 0], sumStaCold[:, 1])

        miniballHotVolume = self.cal_ShphericalVolume(sumStaHot[:, 2])
        miniballColdVolume = self.cal_ShphericalVolume(sumStaCold[:, 2])
        self.LAH_3Dcluster_Hot_Circle = np.mean(sumStaHot[:, 0] / miniballHotVolume)
        self.LAH_3Dcluster_Cold_Circle = np.mean(sumStaCold[:, 0] / miniballColdVolume)


    def cal_Cohesion(self, N, P, A):
        # This function is used to calculate the cohesion of the given values.
        # Parameters:
        #   N: the number of voxels in the landscape
        #   P: the perimeter of the cluster (In FLApy, the perimeter is surface area of the cluster)
        #   A: the area of the cluster (In FLApy, the area is volume of the cluster)
        # Return:
        #   pc: the cohesion of the given values

        p = P / self.voxelArea
        a = A / self.voxelVolume

        sum_p = np.sum(p)
        sum_pa = np.sum(p * np.sqrt(a))
        pc = (1 - (sum_p / sum_pa)) * (1 - (1 / np.sqrt(N))) ** -1 * 100
        return pc

    def cal_ShphericalVolume(self, r2):
        # This function is used to calculate the spherical volume of the given values.
        # Parameters:
        #   r2: the radius of the sphere
        # Return:
        #   The spherical volume of the given values
        r = np.sqrt(r2)
        return (4/3) * np.pi * r**3

    def cal_shape_index(self, V, A):
        # This function is used to calculate the shape index of the given values.
        # Parameters:
        #   V: the volume of the cluster
        #   A: the surface area of the cluster
        # Return:
        #   The shape index of the given values

        return np.mean(A / (6 * V ** (2/3)))

    @staticmethod
    def cal_Gi_fast(coords, value, k=27):
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
        coords = self._allGridPoints_clip
        value = self._value_clip
        k = 27
        time1 = time.time()
        gi_star_valuesSet = self.cal_Gi_fast(coords, value, k)
        print('Time for Gi_Value calculation: ', time.time() - time1, 's')
        if self.bboxSet is False:
            self._inGrid.cell_data['Gi_Value'] = np.array(gi_star_valuesSet)
        elif self.bboxSet is True:
            self._inGrid = self._inGrid_copy.extract_cells(self.indices_clip)
            self._inGrid.cell_data['Gi_Value'] = np.array(gi_star_valuesSet)
        print('Time for Gi_Value saving: ', time.time() - time1, 's')


    def com_allLAH(self, Voxel = True, Vertical = True, Horizontal = True, Cluster3D = True, save = None,
                   bbox = None,
                   thinning = 3, per_95 = True,
                   mode = 'c',
                   givenHeight = None,
                   limit = 27):

        # This function is used to calculate all the LAH indicators. All the parameters will be calculated and stored in the
        # input SFL data.
        # Parameters:
        #   Voxel: whether to calculate the voxel-scale LAH, default is True.
        #   Vertical: whether to calculate the vertical LAH, default is True.
        #   Horizontal: whether to calculate the horizontal LAH, default is True.
        #   Cluster3D: whether to calculate the 3D cluster LAH, default is True.
        #   save: whether to save the results, default is None.
        #   bbox: the bounding box, default is None.
        #   thinning: the thinning rate, default is 3.
        #   per_95: whether to calculate the 95% range of the given values, default is True.
        #   mode: the mode of the calculation, 'v' means voxel sampling, 'c' means column sampling.
        #   givenHeight: the given height, default is None.
        #   limit: the minimum number of voxels in a cluster, default is 27.
        # Return:
        #   A table of all the LAH indicators.

        time_all = time.time()


        if bbox is not None:
            self.bboxSet = True
            self._allGridPoints_clip, self.indices_clip = self.clip_bbox(self.allGridPoints, bbox)
            self._value_clip = self._value[self.indices_clip]
            self.allGridPoints_Full_clip, indices_full = self.clip_bbox(np.array(self._inGrid.field_data['coords_full']), bbox)
            self._value_Full_clip = np.array(self._inGrid.field_data[str(self.fieldName + '_full')])[indices_full]
            self._relativeHeight_Full_clip = np.array(self._inGrid.field_data['Z_normed_full'])[indices_full]

        else:
            self.bboxSet = False
            self._allGridPoints_clip = self.allGridPoints
            self._value_clip = self._value
            self.allGridPoints_Full_clip = np.array(self._inGrid.field_data['coords_full'])
            self._value_Full_clip = np.array(self._inGrid.field_data[str(self.fieldName + '_full')])
            self._relativeHeight_Full_clip = np.array(self._inGrid.field_data['Z_normed_full'])

        self.LAH_Vox_average = 0
        self.LAH_Vox_std = 0
        self.LAH_Vox_CV = 0
        self.LAH_Vox_Range = 0
        self.LAH_Vox_SAC = 0
        self.LAH_Vox_Diversity = 0
        self.LAH_Vox_Gini = 0
        self.LAH_Ver_LAR = 0
        self.LAH_Ver_HIP = 0
        self.LAH_Ver_HIPr = 0
        self.LAH_Ver_ACH = 0
        self.LAH_Hor_average = 0
        self.LAH_Hor_std = 0
        self.LAH_Hor_std_maxHLH_height = 0
        self.LAH_Hor_std_maxHLH_rheight = 0
        self.LAH_Hor_CV = 0
        self.LAH_Hor_Range = 0
        self.LAH_Hor_SAC = 0
        self.LAH_Hor_Diversity = 0
        self.LAH_Hor_Gini = 0
        self.LAH_3Dcluster_Hot_Volume = 0
        self.LAH_3Dcluster_Cold_Volume = 0
        self.LAH_3Dcluster_Hot_Volume_relative = 0
        self.LAH_3Dcluster_Cold_Volume_relative = 0
        self.LAH_3Dcluster_Hot_Largest_Volume_index = 0
        self.LAH_3Dcluster_Cold_Largest_Volume_index = 0
        self.LAH_3Dcluster_VolumeRatio_Hot2Cold = 0
        self.LAH_3Dcluster_Hot_Largest_Volume = 0
        self.LAH_3Dcluster_Cold_Largest_Volume = 0
        self.LAH_3Dcluster_Hot_Abundance = 0
        self.LAH_3Dcluster_Cold_Abundance = 0
        self.LAH_3Dcluster_Hot_Volume_Numweight = 0
        self.LAH_3Dcluster_Cold_Volume_Numweight = 0
        self.LAH_3Dcluster_Hot_Cohesion = 0
        self.LAH_3Dcluster_Cold_Cohesion = 0
        self.LAH_3Dcluster_Hot_Circle = 0
        self.LAH_3Dcluster_Cold_Circle = 0
        self.LAH_3Dcluster_Hot_ShapeIndex = 0
        self.LAH_3Dcluster_Cold_ShapeIndex = 0

        if Voxel is True:
            print('\033[34m' + '----Calculating Voxel-scale (Voxel sampling) Light Heterogeneity ...----' + '\033[0m')
            self.voxel_SummarySta(thinning=thinning, per_95 = per_95)

        if Vertical is True:
            print('\033[34m' + '----Calculating Voxel-scale (Vertical sampling) Light Heterogeneity ...----' + '\033[0m')
            self.vertical_Summary(mode = mode)

        if Horizontal is True:
            print('\033[34m' + '----Calculating Voxel-scale (Horizantal sampling) Light Heterogeneity ...----' + '\033[0m')
            self.horizontal_Summary(givenHeight = givenHeight)

        if Cluster3D is True:
            print('\033[34m' + '----Calculating 3D-Cluster-scale Light Heterogeneity ...----' + '\033[0m')
            self.hotspotAnalysis_fast()
            self.cluster3D_Summary(limiterMin=limit)

        dataSet = {'Indicators': ['Average', 'Standard_deviation', 'Coefficient_of_variation', 'Range',
                               'Spatial_autocorrelation', 'Diversity', 'Gini_coefficient','Light_attenuation_rate',
                               'Height_of_inflection_point', 'Relative_height_of_inflection_point', 'Convex_hull_area',
                               'Average', 'Standard_deviation','Std_MaxHLH_height', 'Std_MaxHLH_relative_height', 'Coefficient_of_variation', 'Range',
                               'Spatial_autocorrelation', 'Diversity', 'Gini_coefficient', 'Hot_volume', 'Cold_volume',
                               'Relative_hot_volume', 'Relative_cold_volume', 'Largest_hot_volume_index', 'Largest_cold_volume_index','Volume_ratio_of_hot_to_cold',
                               'Largest_hot_volume', 'Largest_cold_volume', 'Hot_abundance', 'Cold_abundance',
                               'Hot_volume_average', 'Cold_volume_average', 'Hot_cohesion', 'Cold_cohesion',
                               'Hot_related_circumscribing_sphere', 'Cold_related_circumscribing_sphere', 'Hot_shape_index', 'Cold_shape_index'],
                'Scale': ['Voxel', 'Voxel', 'Voxel', 'Voxel', 'Voxel', 'Voxel', 'Voxel',
                          'Vertical', 'Vertical', 'Vertical', 'Vertical',
                          'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal','Horizontal', 'Horizontal',
                          '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster',
                          '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster',
                          '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster', '3D_Cluster',
                          '3D_Cluster', '3D_Cluster', '3D_Cluster'],
                'Value': [self.LAH_Vox_average,
                          self.LAH_Vox_std,
                          self.LAH_Vox_CV,
                          self.LAH_Vox_Range,
                          self.LAH_Vox_SAC,
                          self.LAH_Vox_Diversity,
                          self.LAH_Vox_Gini,
                          self.LAH_Ver_LAR,
                          self.LAH_Ver_HIP,
                          self.LAH_Ver_HIPr,
                          self.LAH_Ver_ACH,
                          self.LAH_Hor_average,
                          self.LAH_Hor_std,
                          self.LAH_Hor_std_maxHLH_height,
                          self.LAH_Hor_std_maxHLH_rheight,
                          self.LAH_Hor_CV,
                          self.LAH_Hor_Range,
                          self.LAH_Hor_SAC,
                          self.LAH_Hor_Diversity,
                          self.LAH_Hor_Gini,
                          self.LAH_3Dcluster_Hot_Volume,
                          self.LAH_3Dcluster_Cold_Volume,
                          self.LAH_3Dcluster_Hot_Volume_relative,
                          self.LAH_3Dcluster_Cold_Volume_relative,
                          self.LAH_3Dcluster_Hot_Largest_Volume_index,
                          self.LAH_3Dcluster_Cold_Largest_Volume_index,
                          self.LAH_3Dcluster_VolumeRatio_Hot2Cold,
                          self.LAH_3Dcluster_Hot_Largest_Volume,
                          self.LAH_3Dcluster_Cold_Largest_Volume,
                          self.LAH_3Dcluster_Hot_Abundance,
                          self.LAH_3Dcluster_Cold_Abundance,
                          self.LAH_3Dcluster_Hot_Volume_Numweight,
                          self.LAH_3Dcluster_Cold_Volume_Numweight,
                          self.LAH_3Dcluster_Hot_Cohesion,
                          self.LAH_3Dcluster_Cold_Cohesion,
                          self.LAH_3Dcluster_Hot_Circle,
                          self.LAH_3Dcluster_Cold_Circle,
                          self.LAH_3Dcluster_Hot_ShapeIndex,
                          self.LAH_3Dcluster_Cold_ShapeIndex
                          ],
                'Abbreviation': ['AVE_Vox', 'STD_Vox', 'CV_Vox', 'RAN_Vox', 'SAC_Vox', 'DIV_Vox', 'GINI_Vox', 'LAR_Ver',
                                 'HIP_Ver', 'HIPr_Ver', 'ACH_Ver', 'AVE_Hor', 'STD_Hor','STDmh_Hor','STDmrh_Hor', 'CV_Hor', 'RAN_Hor', 'SAC_Hor',
                                 'DIV_Hor', 'GINI_Hor', 'HVOL_3D', 'CVOL_3D', 'HVOLr_3D', 'CVOLr_3D', 'LHI_3D', 'LCI_3D','VRH2C_3D',
                                 'LHV_3D', 'LCV_3D', 'HAB_3D', 'CAB_3D', 'HVA_3D', 'CVA_3D', 'HCO_3D', 'CCO_3D',
                                 'HCC_3D', 'CCC_3D', 'HSI_3D', 'CSI_3D']
                }

        self.indicatorCatalog = pd.DataFrame(dataSet)



        self._inGrid.field_data['AVE_Vox']  = self.LAH_Vox_average
        self._inGrid.field_data['STD_Vox']  = self.LAH_Vox_std
        self._inGrid.field_data['CV_Vox']   = self.LAH_Vox_CV
        self._inGrid.field_data['RAN_Vox']  = self.LAH_Vox_Range
        self._inGrid.field_data['SAC_Vox']  = self.LAH_Vox_SAC
        self._inGrid.field_data['DIV_Vox']  = self.LAH_Vox_Diversity
        self._inGrid.field_data['GINI_Vox'] = self.LAH_Vox_Gini
        self._inGrid.field_data['LAR_Ver']  = self.LAH_Ver_LAR
        self._inGrid.field_data['HIP_Ver']  = self.LAH_Ver_HIP
        self._inGrid.field_data['HIPr_Ver'] = self.LAH_Ver_HIPr
        self._inGrid.field_data['ACH_Ver']  = self.LAH_Ver_ACH
        self._inGrid.field_data['AVE_Hor']  = self.LAH_Hor_average
        self._inGrid.field_data['STD_Hor']  = self.LAH_Hor_std
        self._inGrid.field_data['STDmh_Hor'] = self.LAH_Hor_std_maxHLH_height
        self._inGrid.field_data['STDmrh_Hor'] = self.LAH_Hor_std_maxHLH_rheight
        self._inGrid.field_data['CV_Hor']   = self.LAH_Hor_CV
        self._inGrid.field_data['RAN_Hor']  = self.LAH_Hor_Range
        self._inGrid.field_data['SAC_Hor']  = self.LAH_Hor_SAC
        self._inGrid.field_data['DIV_Hor']  = self.LAH_Hor_Diversity
        self._inGrid.field_data['GINI_Hor'] = self.LAH_Hor_Gini
        self._inGrid.field_data['HVOL_3D']  = self.LAH_3Dcluster_Hot_Volume
        self._inGrid.field_data['CVOL_3D']  = self.LAH_3Dcluster_Cold_Volume
        self._inGrid.field_data['HVOLr_3D'] = self.LAH_3Dcluster_Hot_Volume_relative
        self._inGrid.field_data['CVOLr_3D'] = self.LAH_3Dcluster_Cold_Volume_relative
        self._inGrid.field_data['LHI_3D']   = self.LAH_3Dcluster_Hot_Largest_Volume_index
        self._inGrid.field_data['LCI_3D']   = self.LAH_3Dcluster_Cold_Largest_Volume_index
        self._inGrid.field_data['VRH2C_3D'] = self.LAH_3Dcluster_VolumeRatio_Hot2Cold
        self._inGrid.field_data['LHV_3D']   = self.LAH_3Dcluster_Hot_Largest_Volume
        self._inGrid.field_data['LCV_3D']   = self.LAH_3Dcluster_Cold_Largest_Volume
        self._inGrid.field_data['HAB_3D']   = self.LAH_3Dcluster_Hot_Abundance
        self._inGrid.field_data['CAB_3D']   = self.LAH_3Dcluster_Cold_Abundance
        self._inGrid.field_data['HVA_3D']   = self.LAH_3Dcluster_Hot_Volume_Numweight
        self._inGrid.field_data['CVA_3D']   = self.LAH_3Dcluster_Cold_Volume_Numweight
        self._inGrid.field_data['HCO_3D']   = self.LAH_3Dcluster_Hot_Cohesion
        self._inGrid.field_data['CCO_3D']   = self.LAH_3Dcluster_Cold_Cohesion
        self._inGrid.field_data['HCC_3D']   = self.LAH_3Dcluster_Hot_Circle
        self._inGrid.field_data['CCC_3D']   = self.LAH_3Dcluster_Cold_Circle
        self._inGrid.field_data['HSI_3D']   = self.LAH_3Dcluster_Hot_ShapeIndex
        self._inGrid.field_data['CSI_3D']   = self.LAH_3Dcluster_Cold_ShapeIndex


        if save is None:
            self._inGrid.save(self.tempSFL)
        else:
            self._inGrid.save(save)

        print('\033[42;97m' + '!!!LAH calculation finished!!!' + '\033[0m')
        print('Time for all LAH calculation: ', (time.time() - time_all) / 60, 'min')
        return self.indicatorCatalog

    def com_VoxelScale(self, sampling = 'Voxel', bbox = None, thinning = 1, per_95 = True, mode = 'c', givenHeight = 1.5, save = None):

        if bbox is not None:
            self.allGridPoints, indices = self.clip_bbox(self.allGridPoints, bbox)
            self._value = self._value[indices]

        if sampling == 'Voxel':
            if thinning is not None:
                self.com_VoxelScale_Vox(thinning=thinning, per_95=per_95, save = save)
            else:
                raise ValueError('Please input the thinning factor')

        elif sampling == 'Vertical':
            if bbox.shape == 6:
                raise ValueError('Please input the 4-dim bbox')
            if bbox.shape == 4:
                self.com_VoxelScale_Ver(mode = mode, save = save)


        elif sampling == 'Horizontal':
            self.com_VoxelScale_Hor(height = givenHeight, save = save)

        elif sampling == 'All':
            self.com_VoxelScale_Vox(thinning=thinning, per_95=per_95, save = save)
            self.com_VoxelScale_Ver(mode = mode, save = save)
            self.com_VoxelScale_Hor(height = givenHeight, save = save)

    def com_VoxelScale_Vox(self, thinning = 3, per_95 = True, save = None):
        # This function is used to calculate the voxel-scale LAH. All the parameters will be calculated and stored in the
        # input grid data. The calculation is based on the voxel data.
        # Parameters:
        #   thinning: the thinning factor, default is 3.
        #   per_95: whether to calculate the 95% percentile, default is True.


        self.LAH_Vox_average = 0
        self.LAH_Vox_std = 0
        self.LAH_Vox_CV = 0
        self.LAH_Vox_Range = 0
        self.LAH_Vox_SAC = 0
        self.LAH_Vox_Diversity = 0
        self.LAH_Vox_Gini = 0

        print('\033[34m' + '----Calculating Voxel-scale Light Heterogeneity ...----' + '\033[0m')
        self.voxel_SummarySta(thinning=thinning, per_95=per_95)

        dataSet = {'Indicators': ['Average',
                                  'Standard_deviation',
                                  'Coefficient_of_variation',
                                  'Range',
                                  'spatial_autocorrelation',
                                  'Diversity',
                                  'Gini_coefficient'],
                   'Scale': ['Voxel',
                             'Voxel',
                             'Voxel',
                             'Voxel',
                             'Voxel',
                             'Voxel',
                             'Voxel'],
                   'Value': [self.LAH_Vox_average,
                             self.LAH_Vox_std,
                             self.LAH_Vox_CV,
                             self.LAH_Vox_Range,
                             self.LAH_Vox_SAC,
                             self.LAH_Vox_Diversity,
                             self.LAH_Vox_Gini],
                   'Abbreviation': ['AVE_Vox',
                                    'STD_Vox',
                                    'CV_Vox',
                                    'RAN_Vox',
                                    'SAC_Vox',
                                    'DIV_Vox',
                                    'GINI_Vox']}

        self.indicatorCatalog_VOXELvox = pd.DataFrame(dataSet)


        self._inGrid.field_data['AVE_Vox'] = self.LAH_Vox_average
        self._inGrid.field_data['STD_Vox'] = self.LAH_Vox_std
        self._inGrid.field_data['CV_Vox'] = self.LAH_Vox_CV
        self._inGrid.field_data['RAN_Vox'] = self.LAH_Vox_Range
        self._inGrid.field_data['SAC_Vox'] = self.LAH_Vox_SAC
        self._inGrid.field_data['DIV_Vox'] = self.LAH_Vox_Diversity
        self._inGrid.field_data['GINI_Vox'] = self.LAH_Vox_Gini

        if save is None:
            self._inGrid.save(self.tempSFL)
        else:
            self._inGrid.save(save)
        print('\033[42;97m' + '!!!LAH at Voxel scale (voxel sampling) calculation finished!!!' + '\033[0m')
        return self.indicatorCatalog_VOXELvox

    def com_VoxelScale_Ver(self, mode = 'c', save = None):
        self.LAH_Ver_LAR = 0
        self.LAH_Ver_HIP = 0
        self.LAH_Ver_HIPr = 0
        self.LAH_Ver_ACH = 0

        print('\033[34m' + '----Calculating Vertical-scale Light Heterogeneity ...----' + '\033[0m')

        self.vertical_Summary(mode = mode, save = None)

        dataSet = {'Indicators': ['Light_attenuation_rate',
                                  'Height_of_inflection_point',
                                  'Relative_height_of_inflection_point',
                                  'Convex_hull_area'],
                   'Scale': ['Vertical',
                             'Vertical',
                             'Vertical',
                             'Vertical'],
                   'Value': [self.LAH_Ver_LAR,
                             self.LAH_Ver_HIP,
                             self.LAH_Ver_HIPr,
                             self.LAH_Ver_ACH],
                   'Abbreviation': ['LAR_Ver',
                                    'HIP_Ver',
                                    'HIPr_Ver',
                                    'ACH_Ver']}

        self.indicatorCatalog_VOXELver = pd.DataFrame(dataSet)

        self._inGrid.field_data['LAR_Ver'] = self.LAH_Ver_LAR
        self._inGrid.field_data['HIP_Ver'] = self.LAH_Ver_HIP
        self._inGrid.field_data['HIPr_Ver'] = self.LAH_Ver_HIPr
        self._inGrid.field_data['ACH_Ver'] = self.LAH_Ver_ACH

        if save is None:
            self._inGrid.save(self.tempSFL)
        else:
            self._inGrid.save(save)
        print('\033[42;97m' + '!!!LAH at Voxel scale (vertical sampling) calculation finished!!!' + '\033[0m')
        return self.indicatorCatalog_VOXELver


    def com_VoxelScale_Hor(self, height = None, save = None):
        self.LAH_Hor_average = 0
        self.LAH_Hor_std = 0
        self.LAH_Hor_std_maxHLH_height = 0
        self.LAH_Hor_std_maxHLH_rheight = 0
        self.LAH_Hor_CV = 0
        self.LAH_Hor_Range = 0
        self.LAH_Hor_SAC = 0
        self.LAH_Hor_Diversity = 0
        self.LAH_Hor_Gini = 0

        print('\033[34m' + '----Calculating Horizontal-scale Light Heterogeneity ...----' + '\033[0m')
        self.horizontal_Summary(givenHeight = height)

        dataSet = {'Indicators': ['Average', 'Standard_deviation', 'Std_MaxHLH_height', 'Std_MaxHLH_relative_height',
                                  'Coefficient_of_variation', 'Range',
                                  'spatial_autocorrelation', 'Diversity', 'Gini_coefficient'],
                   'Scale': ['Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal', 'Horizontal'],
                   'Value': [self.LAH_Hor_average, self.LAH_Hor_std, self.LAH_Hor_std_maxHLH_height,
                             self.LAH_Hor_std_maxHLH_rheight, self.LAH_Hor_CV, self.LAH_Hor_Range,
                             self.LAH_Hor_SAC, self.LAH_Hor_Diversity, self.LAH_Hor_Gini],
                   'Abbreviation': ['AVE_Hor', 'STD_Hor','STDmh_Hor',
                                    'STDmrh_Hor', 'CV_Hor', 'RAN_Hor',
                                    'SAC_Hor', 'DIV_Hor', 'GINI_Hor']}

        self.indicatorCatalog_VOXELhor = pd.DataFrame(dataSet)

        self._inGrid.field_data['AVE_Hor'] = self.LAH_Hor_average
        self._inGrid.field_data['STD_Hor'] = self.LAH_Hor_std
        self._inGrid.field_data['STDmh_Hor'] = self.LAH_Hor_std_maxHLH_height
        self._inGrid.field_data['STDmrh_Hor'] = self.LAH_Hor_std_maxHLH_rheight
        self._inGrid.field_data['CV_Hor'] = self.LAH_Hor_CV
        self._inGrid.field_data['RAN_Hor'] = self.LAH_Hor_Range
        self._inGrid.field_data['SAC_Hor'] = self.LAH_Hor_SAC
        self._inGrid.field_data['DIV_Hor'] = self.LAH_Hor_Diversity
        self._inGrid.field_data['GINI_Hor'] = self.LAH_Hor_Gini


        if save is None:
            self._inGrid.save(self.tempSFL)
        else:
            self._inGrid.save(save)

        print('\033[42;97m' + '!!!LAH calculation finished!!!' + '\033[0m')
        return self.indicatorCatalog_VOXELhor



    def com_3DClusterScale(self, limited = 27,bbox = None, save = None):
        if bbox is not None:
            self.allGridPoints, indices = self.clip_bbox(self.allGridPoints, bbox)
            self._value = self._value[indices]

        self.LAH_3Dcluster_Hot_Volume = 0
        self.LAH_3Dcluster_Cold_Volume = 0
        self.LAH_3Dcluster_Hot_Volume_relative = 0
        self.LAH_3Dcluster_Cold_Volume_relative = 0
        self.LAH_3Dcluster_Hot_Largest_Volume_index = 0
        self.LAH_3Dcluster_Cold_Largest_Volume_index = 0
        self.LAH_3Dcluster_VolumeRatio_Hot2Cold = 0
        self.LAH_3Dcluster_Hot_Largest_Volume = 0
        self.LAH_3Dcluster_Cold_Largest_Volume = 0
        self.LAH_3Dcluster_Hot_Abundance = 0
        self.LAH_3Dcluster_Cold_Abundance = 0
        self.LAH_3Dcluster_Hot_Volume_Numweight = 0
        self.LAH_3Dcluster_Cold_Volume_Numweight = 0
        self.LAH_3Dcluster_Hot_Cohesion = 0
        self.LAH_3Dcluster_Cold_Cohesion = 0
        self.LAH_3Dcluster_Hot_Circle = 0
        self.LAH_3Dcluster_Cold_Circle = 0
        self.LAH_3Dcluster_Hot_ShapeIndex = 0
        self.LAH_3Dcluster_Cold_ShapeIndex = 0

        print('\033[34m' + '----Calculating 3D-Cluster Light Heterogeneity ...----' + '\033[0m')
        self.hotspotAnalysis_fast()
        self.cluster3D_Summary(limiterMin=limited)

        dataSet = {'Indicators': ['Hot_volume',
                                  'Cold_volume',
                                  'Relative_hot_volume',
                                  'Relative_cold_volume',
                                  'Largest_hot_volume_index',
                                  'Largest_cold_volume_index',
                                  'Volume_ratio_of_hot_to_cold',
                                  'Largest_hot_volume',
                                  'Largest_cold_volume',
                                  'Hot_abundance',
                                  'Cold_abundance',
                                  'Hot_volume_average',
                                  'Cold_volume_average',
                                  'Hot_cohesion',
                                  'Cold_cohesion',
                                  'Hot_related_circumscribing_sphere',
                                  'Cold_related_circumscribing_sphere',
                                  'Hot_shape_index',
                                  'Cold_shape_index'],
                     'Scale': ['3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster',
                               '3D_Cluster'],
                        'Value': [self.LAH_3Dcluster_Hot_Volume,
                          self.LAH_3Dcluster_Cold_Volume,
                          self.LAH_3Dcluster_Hot_Volume_relative,
                          self.LAH_3Dcluster_Cold_Volume_relative,
                          self.LAH_3Dcluster_Hot_Largest_Volume_index,
                          self.LAH_3Dcluster_Cold_Largest_Volume_index,
                          self.LAH_3Dcluster_VolumeRatio_Hot2Cold,
                          self.LAH_3Dcluster_Hot_Largest_Volume,
                          self.LAH_3Dcluster_Cold_Largest_Volume,
                          self.LAH_3Dcluster_Hot_Abundance,
                          self.LAH_3Dcluster_Cold_Abundance,
                          self.LAH_3Dcluster_Hot_Volume_Numweight,
                          self.LAH_3Dcluster_Cold_Volume_Numweight,
                          self.LAH_3Dcluster_Hot_Cohesion,
                          self.LAH_3Dcluster_Cold_Cohesion,
                          self.LAH_3Dcluster_Hot_Circle,
                          self.LAH_3Dcluster_Cold_Circle,
                          self.LAH_3Dcluster_Hot_ShapeIndex,
                          self.LAH_3Dcluster_Cold_ShapeIndex],
                        'Abbreviation': ['HVOL_3D', 'CVOL_3D', 'HVOLr_3D', 'CVOLr_3D', 'LHI_3D', 'LCI_3D','VRH2C_3D',
                                 'LHV_3D', 'LCV_3D', 'HAB_3D', 'CAB_3D', 'HVA_3D', 'CVA_3D', 'HCO_3D', 'CCO_3D',
                                 'HCC_3D', 'CCC_3D', 'HSI_3D', 'CSI_3D']
                        }

        self.indicatorCatalog_3DCluster = pd.DataFrame(dataSet)

        self._inGrid.field_data['HVOL_3D'] = self.LAH_3Dcluster_Hot_Volume
        self._inGrid.field_data['CVOL_3D'] = self.LAH_3Dcluster_Cold_Volume
        self._inGrid.field_data['HVOLr_3D'] = self.LAH_3Dcluster_Hot_Volume_relative
        self._inGrid.field_data['CVOLr_3D'] = self.LAH_3Dcluster_Cold_Volume_relative
        self._inGrid.field_data['LHI_3D'] = self.LAH_3Dcluster_Hot_Largest_Volume_index
        self._inGrid.field_data['LCI_3D'] = self.LAH_3Dcluster_Cold_Largest_Volume_index
        self._inGrid.field_data['VRH2C_3D'] = self.LAH_3Dcluster_VolumeRatio_Hot2Cold
        self._inGrid.field_data['LHV_3D'] = self.LAH_3Dcluster_Hot_Largest_Volume
        self._inGrid.field_data['LCV_3D'] = self.LAH_3Dcluster_Cold_Largest_Volume
        self._inGrid.field_data['HAB_3D'] = self.LAH_3Dcluster_Hot_Abundance
        self._inGrid.field_data['CAB_3D'] = self.LAH_3Dcluster_Cold_Abundance
        self._inGrid.field_data['HVA_3D'] = self.LAH_3Dcluster_Hot_Volume_Numweight
        self._inGrid.field_data['CVA_3D'] = self.LAH_3Dcluster_Cold_Volume_Numweight
        self._inGrid.field_data['HCO_3D'] = self.LAH_3Dcluster_Hot_Cohesion
        self._inGrid.field_data['CCO_3D'] = self.LAH_3Dcluster_Cold_Cohesion
        self._inGrid.field_data['HCC_3D'] = self.LAH_3Dcluster_Hot_Circle
        self._inGrid.field_data['CCC_3D'] = self.LAH_3Dcluster_Cold_Circle
        self._inGrid.field_data['HSI_3D'] = self.LAH_3Dcluster_Hot_ShapeIndex
        self._inGrid.field_data['CSI_3D'] = self.LAH_3Dcluster_Cold_ShapeIndex

        if save is None:
            self._inGrid.save(self.tempSFL)
        else:
            self._inGrid.save(save)

        print('\033[42;97m' + '!!!LAH at 3D-Cluster scale calculation finished!!!' + '\033[0m')
        return self.indicatorCatalog_3DCluster



    @staticmethod
    def clip_bbox(inCoords, bbox, indices = True):

        bbox = np.array(bbox)

        if bbox.shape[0] == 4:
            xmin, xmax, ymin, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

            xInvalid = np.logical_and((inCoords[:, 0] >= xmin), (inCoords[:, 0] <= xmax))
            yInvalid = np.logical_and((inCoords[:, 1] >= ymin), (inCoords[:, 1] <= ymax))
            xyInvalid = np.where(np.logical_and(xInvalid, yInvalid))

            keptPoints = inCoords[xyInvalid]

            if indices is True:
                return keptPoints, xyInvalid

            return keptPoints


        elif bbox.shape[0] == 6:
            xmin, xmax, ymin, ymax, zmin, zmax = bbox

            xInvalid = np.logical_and((inCoords[:, 0] >= xmin), (inCoords[:, 0] <= xmax))
            yInvalid = np.logical_and((inCoords[:, 1] >= ymin), (inCoords[:, 1] <= ymax))
            zInvalid = np.logical_and((inCoords[:, 2] >= zmin), (inCoords[:, 2] <= zmax))
            xyzInvalid = np.where(np.logical_and(xInvalid, np.logical_and(yInvalid, zInvalid)))

            keptPoints = inCoords[xyzInvalid]

            if indices is True:
                return keptPoints, xyzInvalid

            return keptPoints

    @staticmethod
    def subBbox(inbbox, subNum):
        # This function is used to generate sub-bbox from the given bbox.
        # Parameters:
        #   inbbox: the given bbox, a 4-dim array (Xmin, Xmax, Ymin, Ymax).
        #   subNum: the number of sub-bbox in each dimension.
        # Return:
        #   subBboxs: the sub-bboxs, a (subNum, 4) array. Each row is a sub-bbox including (Xmin, Xmax, Ymin, Ymax).

        subNum = int(np.sqrt(subNum))
        xmin, xmax, ymin, ymax = inbbox[0], inbbox[1], inbbox[2], inbbox[3]

        x_step = (xmax - xmin) / subNum
        y_step = (ymax - ymin) / subNum

        subBboxs = []
        for i in range(subNum):
            for j in range(subNum):
                sub_xmin = xmin + i * x_step
                sub_xmax = xmin + (i + 1) * x_step
                sub_ymin = ymin + j * y_step
                sub_ymax = ymin + (j + 1) * y_step

                subBboxs.append([sub_xmin, sub_xmax, sub_ymin, sub_ymax])

        subBboxs = np.array(subBboxs)
        return subBboxs






    def com_allLAH_subplot(self, Voxel = True, Vertical = True, Horizontal = True, Cluster3D = True, save = None,
                   bbox = None, subNum = 4,
                   thinning = 3, per_95 = True,
                   mode = 'c',
                   givenHeight = None,
                   limit = 27):
        subBbox = self.subBbox(bbox, subNum)


        allResult = pd.DataFrame()
        for i in tqdm(range(subBbox.shape[0]), desc = 'Calculating LAH ...', ncols = 100, colour = '#00AFBB'):
            pdTemp = self.com_allLAH(Voxel = Voxel, Vertical = Vertical, Horizontal = Horizontal, Cluster3D = Cluster3D, save = save,
                   bbox = subBbox[i],
                   thinning = thinning, per_95 = per_95,
                   mode = mode,
                   givenHeight = givenHeight,
                   limit = limit)

            allResult = pd.concat([allResult, pdTemp], axis = 0)

        return allResult


def sigmoid_func(x, a, b):
    #
    return 100 / (1 + np.exp(-a * (x - b)))

def sigmoid_func2(x, L ,x0, k, b):

    y = L / (1 + np.exp(-k*(x-x0))) + b
    return y

