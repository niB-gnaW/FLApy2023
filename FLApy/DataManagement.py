# -*- coding: utf-8 -*-



import math
import pyvista as pv
import os
import numpy as np
import xarray as xr
import open3d as o3d
import laspy
import pdal
import json

from pyvista.core.grid import UniformGrid
from PVGeo.model_build import CreateUniformGrid
from PVGeo.grids import ExtractTopography
from scipy import interpolate
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation, label, binary_closing, generic_filter
from scipy.ndimage.morphology import generate_binary_structure
from tqdm import tqdm

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



        #string
        self.temPath = str(self._workspace + '/.temFile.vtk')
        #XYZ
        self._point_cloud = None
        self._DSM = None
        self._DEM = None
        self._DTM = None
        self._obs = None
        #bool
        self._obsType = 0



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

    def read_LasData(self, filePath, srs = None):
        # This function is used to read the point cloud data from a las file.
        # If the las file contains the ground points, the function will automatically classify the ground points.
        # Parameters:
        #   filePath: the path of the las file.
        #   srs: the spatial reference system of the las file, default is None.

        lasRead = laspy.read(filePath)
        ground_points_exist  = np.any(lasRead.classification == 2)

        if ground_points_exist:
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
        else:
            print(
                '\033[35mNo points classified as ground in the las file. FLApy will automatically classify the imported data for ground points.  \033[0m')
            self.classify_groundPoints(filePath, srs)

        classificationTP = lasRead.classification == 2
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

        if readAs == 'DSM':
            self._DSM = data

        elif readAs == 'DTM':
            self._DTM = data

        elif readAs == 'DEM':
            self._DEM = data

    @staticmethod
    def m2p(inMesh):
        #mesh to points
        # This function is used to convert the mesh data (xarray) to point data(xyv).
        # Parameters:
        #   inMesh: the mesh data.
        # Return:
        #   _point: the point data.

        if not all(dim in inMesh.dims for dim in ['x', 'y']):
            raise ValueError("The input mesh must have 'x' and 'y' dimensions.")

        values = np.asarray(inMesh)
        nodatavals = inMesh.attrs.get('nodatavals')

        if nodatavals is not None:
            nans = values == nodatavals[0]
            if np.any(nans):
                values[nans] = np.nan

        xx, yy = np.meshgrid(inMesh['x'], inMesh['y'])
        result_array = np.column_stack((xx.ravel(), yy.ravel(), values.ravel()))

        return result_array

    @staticmethod
    def p2m(inPoints, resolutuon):
        #points to mesh
        # This function is used to convert the point data(xyv) to mesh data(xarray).
        # Parameters:
        #   inPoints: the point data.
        # Return:
        #   _mesh: the mesh data.
        x = inPoints[:, 0]
        y = inPoints[:, 1]
        z = inPoints[:, 2]

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        rol_indices = np.digitize(x, np.arange(x_min, x_max, 1)) - 1
        col_indices = np.digitize(y, np.arange(y_max, y_min, -1)) - 1

        _rasteration = np.full((int((y_max - y_min) / resolutuon), int((x_max - x_min) / resolutuon)), -np.inf)
        np.maximum.at(_rasteration, (col_indices, rol_indices), z)
        _rasteration[_rasteration == -np.inf] = np.nan

        coords_nan = np.argwhere(np.isnan(_rasteration))
        coords_not_nan = np.argwhere(~np.isnan(_rasteration))

        values_not_nan = _rasteration[~np.isnan(_rasteration)]
        filledNan = interpolate.griddata(coords_not_nan, values_not_nan, coords_nan, method='nearest')
        _rasteration[np.isnan(_rasteration)] = filledNan

        _mesh = xr.DataArray(data = _rasteration,
                             dims = ['y', 'x'],
                             coords = {'y': np.arange(y_max, y_min, -resolutuon), 'x': np.arange(x_min, x_max, resolutuon)},
        )

        return _mesh

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
            self._obsType = 1

        if readAs == 'exList':
            obs = np.loadtxt(filePath, dtype=np.float, delimiter=',', skiprows=skiprows)
            self._obs = obs
            if obs.shape[1] == 4:
                self._obsType = 2
            elif obs.shape[1] == 3:
                self._obsType = 1


    def gen_SFL(self, bbox, resolution,
                bufferSize = 100,
                obsType = None, udXSpacing = None, udYSpacing = None, udZNum = None,
                eDSM_threshold = 10, dilationTimes = 1, specificHeight = None):
        # This function is used to generate the SFL. If users don't provide the DSM, DTM and DEM, the function will generate automatically.
        # Parameters:
        #   bbox: the bounding box of the SFL.
        #   resolution: the resolution of the SFL.
        #   bufferSize: the buffer size of the SFL.
        #   udXSpacing: the user-defined x spacing of the OBS if the traverse method forbidden.
        #   udYSpacing: the user-defined y spacing of the OBS if the traverse method forbidden.
        #   udZSpacing: the user-defined z spacing of the OBS if the traverse method forbidden.
        #

        if self._point_cloud is None:
            raise OSError('Point cloud data has not been read.')
        self._obsType = obsType

        keptPoints = self.clip_Points(self._point_cloud, bbox)

        xMin, xMax = bbox[:2]
        yMin, yMax = bbox[2:]
        zMin = min(keptPoints[:, 2])
        zMax = max(keptPoints[:, 2])

        xComponent = np.ceil((np.ptp(keptPoints[:, 0])) / resolution) + 1
        yComponent = np.ceil((np.ptp(keptPoints[:, 1])) / resolution) + 1
        zComponent = np.ceil((np.ptp(keptPoints[:, 2])) / resolution) + 1

        self.origin = (xMin, yMin, zMin)
        self.endP = (xMax, yMax, zMax)
        self.spacing = (int(resolution), int(resolution), int(resolution))
        self.dimensions = (int(xComponent), int(yComponent), int(zComponent))

        self.bufferSize = bufferSize
        bboxBuffered = np.array(bbox)
        bboxBuffered[0] = bboxBuffered[0] - bufferSize
        bboxBuffered[1] = bboxBuffered[1] + bufferSize
        bboxBuffered[2] = bboxBuffered[2] - bufferSize
        bboxBuffered[3] = bboxBuffered[3] + bufferSize


        self.point_cloud_buffered = self.clip_Points(self._point_cloud, bboxBuffered)
        self._vegPoints_buffered = self.clip_Points(self._vegPoints, bboxBuffered)
        self._terPoints_buffered = self.clip_Points(self._terrainPoints, bboxBuffered)


        if self._DTM is None:
            print('\033[35mDTM has not been read or constructed. FLApy will generate automatically\033[0m')
            self.get_DTM(input_points = self._terPoints_buffered,
                         x_bounds = bboxBuffered[:2],
                         y_bounds = bboxBuffered[2:],
                         resolution=resolution)

        self._vegPoints_buffered_norm = self.normlization_height(self._vegPoints_buffered)
        self._point_cloud_clipedByBbox = keptPoints


        if self._DSM is None:
            print('\033[35meDSM has not been read or constructed. FLApy will generate automatically\033[0m')



            if specificHeight is None:

                X = []
                Y = []
                for index_minSize in tqdm(range(10, 1000, 10), desc='Generating the best optimal eDSM', ncols=100):
                    test_DSM = self.get_DSM(input_points=self._vegPoints_buffered_norm,
                                            x_bounds=bboxBuffered[:2],
                                            y_bounds=bboxBuffered[2:],
                                            resolution=resolution,
                                            threshold=eDSM_threshold,
                                            min_size=index_minSize,
                                            dilationTimes=dilationTimes
                                            )
                    X.append(index_minSize)
                    Y.append(np.mean(test_DSM))

                XY = np.array([X, Y]).transpose()
                limitedV = 0.01
                Y_diff = np.abs(np.diff(XY[:, 1]))

                start_idx = 0
                best_avg = -np.inf
                best_range = (None, None)

                for i in range(len(Y_diff)):
                    if Y_diff[i] > limitedV or i == len(Y_diff) - 1:
                        if start_idx < i:
                            avg_y = np.mean(XY[start_idx:i + 1, 1])
                            if avg_y > best_avg:
                                best_avg = avg_y
                                best_range = (XY[start_idx, 0], XY[i, 0])
                        start_idx = i + 1
                best_X = np.mean(best_range)

                self.get_DSM(input_points=self._vegPoints_buffered_norm,
                             x_bounds=bboxBuffered[:2],
                             y_bounds=bboxBuffered[2:],
                             resolution=resolution,
                             threshold=eDSM_threshold,
                             min_size=best_X,
                             dilationTimes=dilationTimes,
                             specificHeight = np.max(self.points[:, -1])
                             )

            else:
                self.get_DSM(input_points=self._vegPoints_buffered_norm,
                             x_bounds=bboxBuffered[:2],
                             y_bounds=bboxBuffered[2:],
                             resolution=resolution,
                             threshold=eDSM_threshold,
                             dilationTimes=dilationTimes,
                             specificHeight=specificHeight
                             )


        if self._DEM is None:
            print('\033[35mDEM has not been read or constructed. FLApy will generate automatically. The DEM will be used as DEM due to no DEM detected.\033[0m')
            self._DEM = self._DTM

        self._DSMp = self.m2p(self._DSM)
        self._DSMp = self._DSMp[~np.isnan(self._DSMp[:, 2])]
        self._DTMp = self.m2p(self._DTM)
        self._DTMp = self._DTMp[~np.isnan(self._DTMp[:, 2])]

        self._SFL = CreateUniformGrid(origin=self.origin, spacing=self.spacing, extent=self.dimensions).apply()
        ext_dtm = ExtractTopography(invert=True).apply(self._SFL, pv.PolyData(self._DTMp))
        ext_dsm = ExtractTopography(invert=False).apply(self._SFL, pv.PolyData(self._DSMp))
        ext_merge = ext_dsm.cell_data['Extracted'] * ext_dtm.cell_data['Extracted']
        self._SFL.cell_data['Classification'] = ext_merge

        '''
        #FLApy provides four forms of observation points:
        1. (CODE is 0) The observation points are generated by FLApy automatically. The observation points are the voxel centers of the SFL.
        2. (CODE is 1) The observation points are the points provided by users. The points are stored in the field_data['OBS_SFL'].
           The shape of the points is (x, 3), that is containing the x, y, z coordinates.
        3. (CODE is 2) The observation points are the points provided by users. The points are stored in the field_data['OBS_SFL'].
           The shape of the points is (x, 4), that is containing the x, y, z, v coordinates. This Type is uesd to implement the sensitivity analysis for calibrating.
        4. (CODE is 3) The observation points are generated by class method gen_OBSbyUserDefined. The observation points are stored in the field_data['OBS_SFL'].
           The shape of the points is (x, 3), that is containing the x, y, z coordinates.
        '''

        #NONE
        if self._obsType == 0:
            cellCenters = self._SFL.cell_centers()
            self._obs = np.array(cellCenters.points)
            self._obs = self._obs[self._SFL.cell_data['Classification'] == 1]
            self._SFL.field_data['OBS_SFL'] = self._obs

        #xyz
        elif self._obsType == 1:
            self._SFL.field_data['OBS_SFL'] = self.clip_Points(self._obs, bbox)


        #xyzv
        elif self._obsType == 2:

            self._SFL.field_data['OBS_SFL'] = self._obs[:, :3]
            self._SFL.field_data['Given_Value'] = self._obs[:, 3]

        #User-defined
        elif self._obsType == 3:
            if udXSpacing is None or udYSpacing is None or udZNum is None:
                raise ValueError('The user-defined spacing is not specified.')

            self.gen_OBSbyUserDefined(udXSpacing, udYSpacing, udZNum)
            self._SFL.field_data['OBS_SFL'] = self._obs

        dembbox = np.array([bbox[0] - 1000, bbox[1] + 1000, bbox[2] - 1000, bbox[3] + 1000])
        Cdem = self.m2p(self._DEM.sel(x = slice(dembbox[0], dembbox[1]), y = slice(dembbox[2], dembbox[3])))

        self._SFL.field_data['PTS'] = self.point_cloud_buffered
        self._SFL.field_data['DEM'] = Cdem
        self._SFL.field_data['DTM'] = self.m2p(self._DTM)
        self._SFL.field_data['DSM'] = self.m2p(self._DSM)
        self._SFL.field_data['SFLset_resolution'] = self.spacing[0]

        self._SFL.field_data['PTS_cliped'] = self._point_cloud_clipedByBbox
        self._SFL.field_data['DTM_cliped'] = self.clip_Points(self.m2p(self._DTM), bbox)
        self._SFL.field_data['DSM_cliped'] = self.clip_Points(self.m2p(self._DSM), bbox)

        self._SFL.add_field_data([self._obsType], 'OBS_Type')
        self._SFL.add_field_data([self.temPath], 'temPath')

        self._SFL.save(self.temPath)
        print('\033[35mSFL has been generated!' + '\033[0m')


    def gen_OBSbyUserDefined(self, udXSpacing, udYSpacing, udZNum):
        # This function is used to generate the observation points by user-defined spacing.
        # Parameters:
        #   udXSpacing: the user-defined x spacing of the OBS.
        #   udYSpacing: the user-defined y spacing of the OBS.
        #   udZSpacing: the user-defined z spacing of the OBS.

        if udXSpacing <= 0 or udYSpacing <= 0 or udZNum <= 0:
            raise ValueError("Spacing values must be positive.")

        xOrigin = self.origin[0]
        yOrigin = self.origin[1]

        xEnd = self.endP[0]
        yEnd = self.endP[1]

        xx, yy = np.meshgrid(np.arange(xOrigin, xEnd, udXSpacing), np.arange(yOrigin, yEnd, udYSpacing), indexing='ij')
        x_centers = xx + udXSpacing / 2
        y_centers = yy + udYSpacing / 2
        x_centers = x_centers.ravel()
        y_centers = y_centers.ravel()
        z_centersDSM = self.get_ValueByGivenPointsOnRasterMatrix(x_centers.ravel(), y_centers.ravel(), self.DSM).T
        z_centersDTM = self.get_ValueByGivenPointsOnRasterMatrix(x_centers.ravel(), y_centers.ravel(), self.DTM).T

        total_points = len(np.arange(xOrigin, xEnd, udXSpacing)) * len(np.arange(yOrigin, yEnd, udYSpacing)) * udZNum
        obsGen = np.zeros((total_points, 3))

        idx = 0
        for i in range(len(x_centers)):
            z_centers = np.linspace(z_centersDTM[i], z_centersDSM[i], udZNum)
            xyz = np.column_stack((np.full_like(z_centers, x_centers[i]),
                                   np.full_like(z_centers, y_centers[i]),
                                   z_centers))
            obsGen[idx:idx + udZNum, :] = xyz
            idx += udZNum


        self._obs = obsGen
        self._obsType = 3


    def normlization_height(self, inPoints):
        # This function is used to normalize the height of the points.
        # Parameters:
        #   inPoints: the points need to be normalized.
        # Return:
        #   zNormedCoords: the normalized points.

        xs = inPoints[:, 0]
        ys = inPoints[:, 1]
        zs = inPoints[:, 2]
        grabed = self.get_ValueByGivenPointsOnRasterMatrix(xs, ys, self._DTM)

        zNorm = zs - grabed
        zNorm[zNorm < 0] = 0

        zNormedCoords = np.vstack((xs, ys, zNorm)).transpose()

        return zNormedCoords



    @staticmethod
    def clip_Points(inPoints, bbox):
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

    @staticmethod
    def voxelDownsampling(inPoints, resolution=0.5):
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

    @staticmethod
    def get_DSM_ndarray(input_points, x_bounds = None, y_bounds = None, resolution=1):


        x = input_points[:, 0]
        y = input_points[:, 1]
        z = input_points[:, 2]
        if x_bounds is not None and y_bounds is not None:
            x_min, x_max = x_bounds
            y_min, y_max = y_bounds

        else:
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)



        rows = math.ceil((y_max - y_min) / resolution)
        cols = math.ceil((x_max - x_min) / resolution)

        x_bins = np.linspace(x_min, x_max, (cols + 1), )
        y_bins = np.linspace(y_max, y_min, (rows + 1), )

        _DSMraster = np.full((rows, cols), -np.inf)

        col_indices = np.digitize(x, x_bins) - 1
        row_indices = np.digitize(y, y_bins[::-1]) - 1
        np.maximum.at(_DSMraster, (row_indices, col_indices), z)
        _DSMraster[_DSMraster == -np.inf] = np.nan

        coords_nan = np.argwhere(np.isnan(_DSMraster))
        coords_not_nan = np.argwhere(~np.isnan(_DSMraster))

        values_not_nan = _DSMraster[~np.isnan(_DSMraster)]
        filledNan = interpolate.griddata(coords_not_nan, values_not_nan, coords_nan, method='nearest')
        _DSMraster[np.isnan(_DSMraster)] = filledNan

        x_coords, y_coords = np.mgrid[
                             (x_min + resolution / 2):(x_max - resolution / 2 + resolution):resolution,
                             (y_min + resolution / 2):(y_max - resolution / 2 + resolution):resolution
                             ]

        return _DSMraster, x_coords, y_coords

    def get_DSM(self, input_points, x_bounds, y_bounds, resolution=1,
                threshold=1.5, min_size=1, dilationTimes = 1, specificHeight = None):
        # This function can do the interpolation based on given points

        # Parameters:
        # |-input_points: array, the input points' format is XYZ array. The function will run using the data derived from the parant class if its None.
        # |-voxelDownSampling: bool, if True, all points loaded will be down-sample as voxel method.
        # |-resolution: define the resolution of the raster.

        # Return: export an interpolated Digital Surface Model.

        _DSM_ndarray, x_coords, y_coords = self.get_DSM_ndarray(input_points, x_bounds, y_bounds, resolution=resolution)

        chm_layer = np.copy(_DSM_ndarray)
        gaps = (chm_layer <= threshold).astype(int)

        labeled, num_features = label(gaps)
        filled_gaps = np.copy(chm_layer)
        gap_mask = np.zeros(chm_layer.shape, dtype=bool)

        struct_element = generate_binary_structure(2, 2)


        for i in range(1, num_features + 1):
            mask_window = labeled == i
            boundary = mask_window.copy()
            for _ in range(dilationTimes):
                boundary = binary_dilation(boundary, structure=struct_element)
            buffered_boundary = boundary ^ mask_window

            if specificHeight is None:
                surrounding_tree_max = np.nanmax(chm_layer[buffered_boundary])
            else:
                surrounding_tree_max = specificHeight

            gap_area = np.sum(mask_window) * resolution ** 2

            if min_size <= gap_area:
                filled_gaps[mask_window] = surrounding_tree_max
                gap_mask[mask_window] = True

        mask_of_filled_gaps = np.array(~gap_mask, dtype = np.uint8)
        distances, indices = distance_transform_edt(mask_of_filled_gaps, return_indices=True)
        nearest_filled_gap_values = filled_gaps[tuple(indices)]
        mask_to_increase = filled_gaps < nearest_filled_gap_values
        filled_gaps[mask_to_increase] = nearest_filled_gap_values[mask_to_increase]


        self._DSM_filled_ndarray = filled_gaps + self._DTM_ndarray

        da_DSM = xr.DataArray(
            data=self._DSM_filled_ndarray,
            dims=["y", "x"],
            coords=dict(
                x=x_coords[:, 0],
                y=y_coords[0, :],
            ),
        )

        self._DSM = da_DSM
        return filled_gaps


    def get_DTM(self, input_points, x_bounds, y_bounds, resolution=1):
        # This function can do the interpolation based on given points
        # Parameters:
        # |-input_points: array, the input points' format is XYZ array. The function will run using the data derived from the parant class if its None.
        # |-voxelDownSampling: bool, if True, all points loaded will be down-sample as voxel method.
        # |-resolution: define the resolution of the raster.

        # Return: export an interpolated Digital Terrain Model.
        x = input_points[:, 0]
        y = input_points[:, 1]
        z = input_points[:, 2]

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        rows = math.ceil((y_max - y_min) / resolution)
        cols = math.ceil((x_max - x_min) / resolution)

        x_bins = np.linspace(x_min, x_max, (cols + 1), )
        y_bins = np.linspace(y_max, y_min, (rows + 1), )

        _DTMraster = np.full((rows, cols), -np.inf)

        col_indices = np.digitize(x, x_bins) - 1
        row_indices = np.digitize(y, y_bins[::-1]) - 1

        np.maximum.at(_DTMraster, (row_indices, col_indices), z)
        _DTMraster[_DTMraster == -np.inf] = np.nan

        coords_nan = np.argwhere(np.isnan(_DTMraster))
        coords_not_nan = np.argwhere(~np.isnan(_DTMraster))

        values_not_nan = _DTMraster[~np.isnan(_DTMraster)]
        filledNan = interpolate.griddata(coords_not_nan, values_not_nan, coords_nan, method='nearest')
        _DTMraster[np.isnan(_DTMraster)] = filledNan
        self._DTM_ndarray = _DTMraster

        x_coords, y_coords = np.mgrid[
                             (x_min + resolution / 2):(x_max - resolution / 2 + resolution):resolution,
                             (y_min + resolution / 2):(y_max - resolution / 2 + resolution):resolution
                             ]

        da = xr.DataArray(
            data=_DTMraster,
            dims=["y", "x"],
            coords=dict(
                x=x_coords[:, 0],
                y=y_coords[0, :],
            ),
        )


        self._DTM = da

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


    def classify_groundPoints(self, filePath, srs = None):
        # This function is used to classify the ground points automatically.
        # Parameters:
        #   filePath: the path of the las file.
        #   srs: the spatial reference system of the las file.

        if srs is None:
            nosrs = True
        outputFilePath = filePath.replace('.las', '_classified.las')
        pipeline_dict = {
            "pipeline": [
                {
                    "type": "readers.las",
                    "filename": filePath,
                    "nosrs": nosrs
                },
                {
                    "type": "filters.smrf",
                    "ignore": "Classification[7:7]",
                    "scalar": 1.25,
                    "slope": 0.2,
                    "threshold": 0.45,
                    "window": 16.0
                },
                {
                    "type": "writers.las",
                    "filename": outputFilePath
                }
            ]
        }
        pipeline_json = json.dumps(pipeline_dict)
        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()
        self.read_LasData(outputFilePath)


    @staticmethod
    def cart2pol(x, y):
        # This function is used to convert the cartesian coordinates to polar coordinates.
        # Parameters:
        #   x: the x coordinate of the point.
        #   y: the y coordinate of the point.
        # Return:
        #   r: the radius of the point.
        #   theta: the angle of the point.
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta


    def multiDirectinalGradientFilter(self, inArr):
        value1D = np.array(inArr).flatten()
        coords = np.meshgrid(np.arange(inArr.shape[0]), np.arange(inArr.shape[1]))
        X = coords[0].flatten()
        Y = coords[1].flatten()
        origin = np.array([(np.max(X) - np.min(X)) / 2, (np.max(Y) - np.min(Y)) / 2])
        X_tra = X - origin[0]
        Y_tra = Y - origin[1]

        coords_r, coords_t = self.cart2pol(X_tra, Y_tra)

        Ir1 = value1D[coords_r < self._r1]
        Ir1_median = np.median(Ir1)

        Pd = np.zeros(8)
        for i in range(8):
            subZone_t_min = -np.pi + i * np.pi / 4
            subZone_t_max = -np.pi + (i + 1) * np.pi / 4

            annulus_r_min = self._r1
            annulus_r_max = self._r1 + self._radd

            logics_t = np.logical_and(coords_t >= subZone_t_min, coords_t <= subZone_t_max)
            logics_r = np.logical_and(coords_r >= annulus_r_min, coords_r <= annulus_r_max)
            logics_tANDr = np.logical_and(logics_t, logics_r)

            Pd[i] = Ir1_median - np.median(value1D[logics_tANDr])

        if np.max(Pd) - np.min(Pd) == 0:
            C_r1_r2 = np.nan
        else:
            C_r1_r2 = np.mean(Pd) / np.abs((np.max(Pd) - np.min(Pd)))
        return C_r1_r2

    def Mapping_specificSize(self):
        width, height = self._CHM.shape

        half_subSize = (self._r1 + self._radd) // 2
        self._mapped = np.full((width, height), np.nan)

        for i in tqdm(range(width), desc='Mapping...Scale Added = %i' % self._radd, ncols=100, total=width):
            for j in range(height):
                # Calculate the start and end for x and y using max and min to handle border cases
                x_start = max(0, i - half_subSize)
                x_end = min(width, i + half_subSize + 1)
                y_start = max(0, j - half_subSize)
                y_end = min(height, j + half_subSize + 1)

                target_x_start = half_subSize - min(i, half_subSize)
                target_x_end = target_x_start + (x_end - x_start)
                target_y_start = half_subSize - min(j, half_subSize)
                target_y_end = target_y_start + (y_end - y_start)

                window_data = np.zeros((self._r1 + self._radd + 1, self._r1 + self._radd + 1))
                window_data[target_x_start:target_x_end, target_y_start:target_y_end] = self._CHM[x_start:x_end, y_start:y_end]


                if self._CHM[i, j] != 0:
                    self._mapped[i, j] = self.multiDirectinalGradientFilter(window_data)
                else:
                    self._mapped[i, j] = np.nan

    @staticmethod
    def normalize_data(data):
        return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

    def get_MDGF(self, input_points, x_bounds, y_bounds, resolution=1, r1 = 1, radd = 1, numRings = 9):
        _DSM_ndarray, x_coords, y_coords = self.get_DSM_ndarray(input_points, x_bounds, y_bounds, resolution=resolution)
        self._CHM = _DSM_ndarray - self._DTM_ndarray
        multiBands = []
        for __numRings in range(1, numRings + 1):
            self._r1 = r1
            self._radd = radd * __numRings
            self.Mapping_specificSize()
            multiBands.append(self._mapped)

        self._MDGF = np.stack(multiBands, axis = -1)
        self._MDGF = np.array([self.normalize_data(self._MDGF[:, :, i]) for i in range(self._MDGF.shape[2])]).transpose(1, 2, 0)







class dataInput(object):

    def __init__(self, filePath = None):
        self.version = 2.0
        self.rootI = os.getcwd()
        self.filePath = filePath

    def read_VTK(self):
        # This function is used to read the vtk file.
        # Parameters:
        #   filePath: the path of the vtk file.
        # Return:
        #   vtkR: the vtk data.
        self._vtkR = pv.read(self.filePath)
        return self._vtkR

    def chk_SFL(self):
        return

