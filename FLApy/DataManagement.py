# -*- coding: utf-8 -*-
#---------------------------------------------------------------------#
#   FLApy: Forest Light Analyzer python package                       #
#   Developer: Wang Bin (Yunnan University, Kunming, China)           #

import pyvista as pv
import os
import numpy as np
import xarray as xr
import open3d as o3d
import skimage.morphology as sm

from laspy.file import File
from pyvista.core.grid import UniformGrid
from PVGeo.model_build import CreateUniformGrid
from PVGeo.grids import ExtractTopography
from scipy import interpolate
from scipy.ndimage import distance_transform_edt
from collections import deque


class StudyFieldLattice(UniformGrid):
    # This class is used to create a SFL object for the study field. All the data will be stored in this object.
    # Parameters:
    #   workspace: the workspace of the project, default is the current working directory. If the workspace is specified, the temFile.vtk will be stored in the workspace.
    #
    # Return:
    #   SFL: a SFL object

    def __init__(self, workspace = None, *args, **kwargs):
        super().__init__(*args, **kwargs)


        if workspace is None:
            self._workspace = os.path.abspath(os.getcwd())
            print('\033[35mThe workspace is not specified, the temFile.vtk will be stored in the default working directory: ' + self._workspace+'\033[0m')
        else:
            self._workspace = workspace

        self.temPath = str(self._workspace + '/.temFile.vtk')

        self._point_cloud = None
        self._DSM = None
        self._DEM = None
        self._DTM = None
        self._obs = None
        self._DataContainer = None
        self._obsExternalLabel = None


    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    @property
    def point_cloud(self) -> np.ndarray:
        return self._point_cloud
    @property
    def DSM(self) -> xr.DataArray:
        return self._DSM

    @property
    def DTM(self) -> xr.DataArray:
        return self._DTM

    @property
    def DEM(self) -> xr.DataArray:
        return self._DEM

    @property
    def OBS(self) -> np.ndarray:
        return self._obs

    @point_cloud.setter
    def point_cloud(self, filePath):
        self.read_LasData(filePath)

    @DEM.setter
    def DEM(self, filePath):
        infile_DEM = self.read_RasterData(filePath, readAs='DEM')
        setattr(self, 'dem_data', infile_DEM)

    @DSM.setter
    def DSM(self, filePath):
        infile_DSM = self.read_RasterData(filePath, readAs='DSM')
        setattr(self, 'dsm_data', infile_DSM)

    @DTM.setter
    def DTM(self, filePath):
        infile_DTM = self.read_RasterData(filePath, readAs='DTM')
        setattr(self, 'dtm_data', infile_DTM)

    @OBS.setter
    def OBS(self, filePath):
        infile_OBS = self.read_CsvData(filePath)
        setattr(self, 'OBS_data', infile_OBS)

    def read_LasData(self, filePath):
        # This function is used to read the point cloud data from a las file.
        # Parameters:
        #   filePath: the path of the las file.

        lasRead = File(filePath, mode='r')

        x_dimension = lasRead.X
        scale_X = lasRead.header.scale[0]
        offset_X = lasRead.header.offset[0]
        X_Pcfin = x_dimension * scale_X + offset_X

        y_dimension = lasRead.Y
        scale_Y = lasRead.header.scale[1]
        offset_Y = lasRead.header.offset[1]
        Y_Pcfin = y_dimension * scale_Y + offset_Y

        z_dimension = lasRead.Z
        scale_Z = lasRead.header.scale[2]
        offset_Z = lasRead.header.offset[2]
        Z_Pcfin = z_dimension * scale_Z + offset_Z

        self._point_cloud = np.vstack((X_Pcfin, Y_Pcfin, Z_Pcfin)).transpose()
        classificationTP = lasRead.Classification == 2
        self._terrainPoints = self._point_cloud[classificationTP]
        self._vegPoints = self._point_cloud[~classificationTP]


    def read_RasterData(self, filePath, readAs = 'DSM'):
        # This function is used to read the raster data from a raster file.
        # Parameters:
        #   filePath: the path of the raster file.
        #   readAs: the type of the raster file, default is DSM.
        #       DSM: Digital Surface Model
        #       DTM: Digital Terrain Model
        #       DEM: Digital Elevation Model

        data = xr.open_rasterio(filePath)
        values = np.asarray(data)
        nans = values == data.nodatavals
        if np.any(nans):
            values[nans] = np.nan
        xx, yy = np.meshgrid(data['x'], data['y'])
        zz = values.reshape(xx.shape)
        _mesh = pv.StructuredGrid(xx, yy, zz)
        _mesh['data'] = values.ravel(order='F')

        if readAs == 'DSM':
            self._DSM = _mesh
            self._DSMp = self.m2p(self._DSM)
            self._DSMmesh = data

            return self._DSM
        elif readAs == 'DTM':
            self._DTM = _mesh
            self._DTMp = self.m2p(self._DTM)
            self._DTMmesh = data

            return self._DTM
        elif readAs == 'DEM':
            self._DEM = _mesh
            self._DEMp = self.m2p(self._DEM)
            self._DEMmesh = data


    @classmethod
    def m2p(cls, inMesh): #mesh to points
        # This function is used to convert the mesh data to point data.
        # Parameters:
        #   inMesh: the mesh data.
        # Return:
        #   _point: the point data.

        _point = np.array(inMesh.points)
        _point = _point[~np.isnan(_point[:, 2])]
        _point = pv.PolyData(_point)
        return _point

    @classmethod
    def p2m(cls, inPoints): #points to mesh
        # This function is used to convert the point data to mesh data.
        # Parameters:
        #   inPoints: the point data.
        # Return:
        #   _mesh: the mesh data.
        pointsArr = np.array(inPoints.points)
        unique_x, x_indices = np.unique(pointsArr[:, 0], return_inverse=True)
        unique_y, y_indices = np.unique(pointsArr[:, 1], return_inverse=True)
        array_2d = np.full((len(unique_x), len(unique_y)), np.nan)
        array_2d[x_indices, y_indices] = pointsArr[:, 2]
        da = xr.DataArray(array_2d, coords={'x': unique_x, 'y': unique_y}, dims=['x', 'y'])
        return da


    def read_csvData(self, filePath, skiprows = 1, readAs = 'OBS'):
        # This function is used to read the csv data. (x,4)or(x,3)
        # Parameters:
        #   filePath: the path of the csv file.
        #   skiprows: the number of rows to skip at the beginning of the file.
        #   readAs: the type of the csv file, default is OBS. OBS denotes the csv file is the observation points (x, 3).
        #           OBS is used to set the observation points manually. exList denotes the csv file is the external list (x, 4).
        #           exList is used to implement the sensitivity analysis or uncertainty analysis.
        #
        #
        # Return:
        #   obs: the csv data.

        if readAs == 'OBS':
            obs = np.loadtxt(filePath, dtype=np.float, delimiter=',', skiprows=skiprows)
            self._obs = obs
            self._obsExternalLabel = 1

        if readAs == 'exList':
            obs = np.loadtxt(filePath, dtype=np.float, delimiter=',', skiprows=skiprows)
            self._obs = obs
            self._obsExternalLabel = 2




    def gen_SFL(self, bbox, resolution, bufferSize = 100):
        # This function is used to generate the SFL. If users don't provide the DSM, DTM and DEM, the function will generate automatically.
        # Parameters:
        #   bbox: the bounding box of the SFL.
        #   resolution: the resolution of the SFL.
        #   bufferSize: the buffer size of the SFL.
        #   generatedOBS: whether to generate the observation points.
        #   genAuto: whether to generate all data required automatically.



        if self._point_cloud is None:
            raise OSError('Point cloud data has not been read.')


        keptPoints = self.clip_Points(self._point_cloud, bbox)

        xMin = bbox[0]
        yMin = bbox[2]
        zMin = min(keptPoints[:, 2])

        xComponent = np.ceil((np.ptp(keptPoints[:, 0])) / resolution) + 1
        yComponent = np.ceil((np.ptp(keptPoints[:, 1])) / resolution) + 1
        zComponent = np.ceil((np.ptp(keptPoints[:, 2])) / resolution) + 1

        self.origin = (xMin, yMin, zMin)
        self.spacing = (int(resolution), int(resolution), int(resolution))
        self.dimensions = (int(xComponent), int(yComponent), int(zComponent))

        self.bufferSize = bufferSize
        bboxBuffered = np.array(bbox)
        bboxBuffered[0] = bboxBuffered[0] - bufferSize
        bboxBuffered[1] = bboxBuffered[1] + bufferSize
        bboxBuffered[2] = bboxBuffered[2] - bufferSize
        bboxBuffered[3] = bboxBuffered[3] + bufferSize


        self.point_cloud_buffered = self.clip_Points(self._vegPoints, bboxBuffered)
        self._vegPoints_buffered = self.clip_Points(self._vegPoints, bboxBuffered)
        self._terPoints_buffered = self.clip_Points(self._terrainPoints, bboxBuffered)
        self._point_cloud_ = keptPoints



        if self._DSM is None:
            print('\033[35mDSM has not been read or constructed. FLApy will generate automatically\033[0m')
            self.surfacePoint = self.pc2raster(self._vegPoints_buffered, bboxBuffered[:2], bboxBuffered[2:], resolution)
            self.get_DSM(self.surfacePoint, voxelDownSampling=False, resolution=resolution)
            self._DSMp = self.m2p(self._DSM)
            self._DSMmesh = self.p2m(self._DSMp)

        if self._DTM is None:
            print('\033[35mDTM has not been read or constructed. FLApy will generate automatically\033[0m')
            self.get_DTM(self._terPoints_buffered, voxelDownSampling=True, resolution=resolution)
            self._DTMp = self.m2p(self._DTM)
            self._DTMmesh = self.p2m(self._DTMp)

        if self._DEM is None:
            print('\033[35mDEM has not been read or constructed. FLApy will generate automatically. The DEM will be used as DEM due to no DEM detected.\033[0m')
            self._DEM = self._DTM
            self._DEMp = self.m2p(self._DEM)
            self._DEMmesh = self.p2m(self._DEMp)

        self._SFL = CreateUniformGrid(origin=self.origin, spacing=self.spacing, extent=self.dimensions).apply()

        ext_dtm = ExtractTopography(invert=True).apply(self._SFL, self._DTMp)
        ext_dsm = ExtractTopography(invert=False).apply(self._SFL, self._DSMp)
        ext_merge = ext_dsm.cell_data['Extracted'] * ext_dtm.cell_data['Extracted']
        self._SFL.cell_data['Classification'] = ext_merge
        self._vegPoints_TerrainNormalization = self.normlization_height(self.point_cloud_buffered)
        cellCenters = self._SFL.cell_centers()

        if self._obsExternalLabel is None:
            self._obs = np.array(cellCenters.points)
            self._cellPoints_TerrainNormalization = self.normlization_height(self._obs)
            self._SFL.field_data['OBS_SFL'] = self._obs
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]

        elif self._obsExternalLabel == 1:
            self._obsSet = self._obs
            self._cellPoints_TerrainNormalization = self.normlization_height(self._obsSet)
            self._SFL.field_data['OBS_SFL'] = self._obsSet
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]

        elif self._obsExternalLabel == 2:
            self._obsSet = self._obs
            self._cellPoints_TerrainNormalization = self.normlization_height(self._obsSet)
            self._SFL.field_data['OBS_SFL'] = self._obsSet
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]


        Cdsm = self.m2p(self.DEM.clip_box(self._DTMp.bounds, invert=True))

        self._SFL.field_data['PTS'] = self.point_cloud_buffered
        self._SFL.field_data['DEM'] = Cdsm.points
        self._SFL.field_data['DTM'] = self._DTMp.points
        self._SFL.field_data['DSM'] = self._DSMp.points

        self._SFL.add_field_data([self._obsExternalLabel], 'obsExternalLabel')
        self._SFL.add_field_data([self.temPath], 'temPath')

        self._SFL.save(self.temPath)
        print('\033[35mSFL has been generated!' + '\033[0m')



    def normlization_height(self, inPoints):
        # This function is used to normalize the height of the points.
        # Parameters:
        #   inPoints: the points need to be normalized.
        # Return:
        #   zNormedCoords: the normalized points.

        xs = inPoints[:, 0]
        ys = inPoints[:, 1]
        zs = inPoints[:, 2]
        grabed = self.get_ValueByGivenPointsOnRasterMatrix(xs, ys, self._DTMmesh)

        zNorm = zs - grabed
        zNorm[zNorm < 0] = 0

        zNormedCoords = np.vstack((xs, ys, zNorm)).transpose()

        return zNormedCoords

    @classmethod
    def clip_Points(cls, inPoints, bbox):
        # This function is used to clip the points.
        # Parameters:
        #   inPoints: the points need to be clipped.
        #   bbox: the bounding box of the points.
        # Return:
        #   keptPoints: the clipped points.

        bbox_Xmin, bbox_Xmax = bbox[0], bbox[1]
        bbox_Ymin, bbox_Ymax = bbox[2], bbox[3]

        xInvalid = np.logical_and((inPoints[:, 0] >= bbox_Xmin), (inPoints[:, 0] <= bbox_Xmax))
        yInvalid = np.logical_and((inPoints[:, 1] >= bbox_Ymin), (inPoints[:, 1] <= bbox_Ymax))

        keptIndices = np.where(np.logical_and(xInvalid, yInvalid))

        keptPoints = inPoints[keptIndices]

        return keptPoints

    @classmethod
    def voxelDownsampling(cls, inPoints, resolution=0.5):
        # This function is used to downsample the points.
        # Parameters:
        #   inPoints: the points need to be downsampled.
        #   resolution: the resolution of the downsampled points.
        # Return:
        #   vd: the downsampled points.
        pointCloudInit = o3d.geometry.PointCloud()
        pointCloudInit.points = o3d.utility.Vector3dVector(inPoints)
        vd = pointCloudInit.voxel_down_sample(voxel_size=resolution)
        return np.array(vd.points)

    def interp_ByPoints(self, input_points = None, voxelDownSampling = False, resolution = 1):
        # This function is used to interpolate the points.
        # Parameters:
        #   input_points: the points need to be interpolated.
        #   voxelDownSampling: whether to downsample the points.
        #   resolution: the resolution of the matrix interpolated.
        # Return:
        #   interpMatrix: the interpolated matrix.

        if input_points is None:
            _inAP = self._point_cloud
        else:
            _inAP = input_points

        if voxelDownSampling:
            TP_vd = self.voxelDownsampling(_inAP, resolution)
        else:
            TP_vd = _inAP

        X = TP_vd[:, 0]
        Y = TP_vd[:, 1]
        Z = TP_vd[:, 2]

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)

        points = np.concatenate([X, Y], axis=1)

        X_min, X_max = np.min(X), np.max(X)
        Y_min, Y_max = np.min(Y), np.max(Y)
        Z_min, Z_max = np.min(Z), np.max(Z)

        cell_size = resolution

        X_grid, Y_grid, Z_grid = np.meshgrid(np.arange(X_min, X_max, cell_size), np.arange(Y_min, Y_max, cell_size), np.arange(Z_min, Z_max, cell_size))
        grid_data = interpolate.griddata(points, Z, (X_grid, Y_grid), method='linear')

        _mesh = pv.StructuredGrid(X_grid, Y_grid, Z_grid)
        _mesh['data'] = grid_data.ravel(order='F')

        return _mesh




    def pc2raster(self, inPoints, x_bounds, y_bounds, resolution = 1):
        # This function is used to convert the points to raster.
        # Parameters:
        #   inPoints: the points need to be converted.
        #   resolution: the resolution of the raster.
        # Return:
        #   raster: the raster data.

        x = inPoints[:, 0]
        y = inPoints[:, 1]
        z = inPoints[:, 2]

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        col_indices = np.digitize(x, np.arange(x_min, x_max, resolution)) - 1
        row_indices = np.digitize(y, np.arange(y_max, y_min, -resolution)) - 1


        _raster = np.full((int((y_max - y_min) / resolution), int((x_max - x_min) / resolution)), -np.inf)
        np.maximum.at(_raster, (row_indices, col_indices), z)
        _raster[_raster == -np.inf] = np.nan

        mask_nan = np.isnan(_raster)
        distances, indices = distance_transform_edt(mask_nan, return_indices=True)
        nearest_values = _raster[tuple(indices)]
        _raster[mask_nan] = nearest_values[mask_nan]


        yy, xx = np.mgrid[y_max:y_min:-resolution, x_min:x_max:resolution]
        out_array = np.vstack((xx.ravel(), yy.ravel(), _raster.ravel())).T

        return out_array





    def get_DSM(self, input_points = None, voxelDownSampling = False, resolution = 1):
        # This function can do the interpolation based on given points

        # Parameters:
        # |-input_points: array, the input points' format is XYZ array. The function will run using the data derived from the parant class if its None.
        # |-voxelDownSampling: bool, if True, all points loaded will be down-sample as voxel method.
        # |-resolution: define the resolution of the raster.

        # Return: export an interpolated Digital Surface Model.
        if input_points is None:
            _inAP = self._point_cloud
        else:
            _inAP = input_points

        if voxelDownSampling:
            TP_vd = self.voxelDownsampling(_inAP, resolution)
        else:
            TP_vd = _inAP


        self._DSM = self.interp_ByPoints(TP_vd, voxelDownSampling=voxelDownSampling, resolution=resolution)



    def get_DTM(self, input_points=None, voxelDownSampling=False, resolution=1):
        # This function can do the interpolation based on given points
        # Parameters:
        # |-input_points: array, the input points' format is XYZ array. The function will run using the data derived from the parant class if its None.
        # |-voxelDownSampling: bool, if True, all points loaded will be down-sample as voxel method.
        # |-resolution: define the resolution of the raster.

        # Return: export an interpolated Digital Terrain Model.
        if input_points is None:
            _inTP = self._terrainPoints
        else:
            _inTP = input_points

        if voxelDownSampling:
            _TP_vd = self.voxelDownsampling(_inTP, resolution)
        else:
            _TP_vd = _inTP

        self._DTM = self.interp_ByPoints(_TP_vd, voxelDownSampling=voxelDownSampling, resolution=resolution)
        #return DTM



    @staticmethod
    def get_ValueByGivenPointsOnRasterMatrix(x, y, raster, method = "nearest"):
        # This function is used to get a series of value by given points xy coordinate.
        # parameter:
        # -x: Import an array of x coordinate of given points
        # -y: Import an array of y coordinate of given points
        # -raster: Import the raster which want to query

        # Return: A series of value in the raster where the given points are.

        tgt_x = xr.DataArray(x, dims='points')
        tgt_y = xr.DataArray(y, dims='points')

        da = raster.sel(x=tgt_x, y=tgt_y, method=method)

        return da.data



class dataInput(object):

    '''''
    d point cloud data and returns the PyVista object of the DTM. This method uses geostatistics, using the semivariogram function to fit the spatial variability of the point cloud data to generate the DTM. Optional parameters include: whether to plot the DTM, etc.
    '''''

    def __init__(self, filePath = None):
        self.version = 2.0
        self.rootI = os.getcwd()
        self.filePath = filePath


    def read_VTK(self):
        _vtkR = pv.read(self.filePath)
        return _vtkR



