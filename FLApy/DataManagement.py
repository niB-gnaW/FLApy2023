# -*- coding: utf-8 -*-
#---------------------------------------------------------------------#
#   FLApy: Forest Light Analyzer python package                       #
#   Developer: Wang Bin (Yunnan University, Kunming, China)           #

import pyvista as pv
import os
import numpy as np
import xarray as xr
import open3d as o3d
import rasterio


from laspy.file import File
from pyvista.core.grid import UniformGrid
from PVGeo.model_build import CreateUniformGrid
from PVGeo.grids import ExtractTopography
from scipy import interpolate
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation, label, binary_closing, generic_filter
from scipy.ndimage.morphology import generate_binary_structure
from pysheds.grid import Grid
from rasterio.transform import from_origin




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
    def p2m(inPoints):
        #points to mesh
        # This function is used to convert the point data(xyv) to mesh data(xarray).
        # Parameters:
        #   inPoints: the point data.
        # Return:
        #   _mesh: the mesh data.
        pointsArr = np.array(inPoints)
        unique_x, x_indices = np.unique(pointsArr[:, 0], return_inverse=True)
        unique_y, y_indices = np.unique(pointsArr[:, 1], return_inverse=True)

        array_2d = np.full((len(unique_x), len(unique_y)), np.nan)
        count_array = np.zeros_like(array_2d, dtype=int)

        for i, (x_ind, y_ind, val) in enumerate(zip(x_indices, y_indices, pointsArr[:, 2])):
            if np.isnan(array_2d[x_ind, y_ind]):
                array_2d[x_ind, y_ind] = val
                count_array[x_ind, y_ind] = 1
            else:
                array_2d[x_ind, y_ind] += val
                count_array[x_ind, y_ind] += 1

        valid_counts = count_array > 0
        array_2d[valid_counts] /= count_array[valid_counts]

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




    def gen_SFL(self, bbox, resolution, bufferSize = 100, obsType = None, udXSpacing = None, udYSpacing = None, udZNum = None):
        # This function is used to generate the SFL. If users don't provide the DSM, DTM and DEM, the function will generate automatically.
        # Parameters:
        #   bbox: the bounding box of the SFL.
        #   resolution: the resolution of the SFL.
        #   bufferSize: the buffer size of the SFL.
        #   udXSpacing: the user-defined x spacing of the OBS if the traverse method forbidden.
        #   udYSpacing: the user-defined y spacing of the OBS if the traverse method forbidden.
        #   udZSpacing: the user-defined z spacing of the OBS if the traverse method forbidden.




        if self._point_cloud is None:
            raise OSError('Point cloud data has not been read.')
        self._obsExternalLabel = obsType


        keptPoints = self.clip_Points(self._point_cloud, bbox)

        xMin = bbox[0]
        yMin = bbox[2]
        zMin = min(keptPoints[:, 2])
        xMax = bbox[1]
        yMax = bbox[3]
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


        self.point_cloud_buffered = self.clip_Points(self._vegPoints, bboxBuffered)
        self._vegPoints_buffered = self.clip_Points(self._vegPoints, bboxBuffered)
        self._terPoints_buffered = self.clip_Points(self._terrainPoints, bboxBuffered)


        if self._DTM is None:
            print('\033[35mDTM has not been read or constructed. FLApy will generate automatically\033[0m')
            self.get_DTM(self._terPoints_buffered, voxelDownSampling=True, resolution=resolution)


        self._vegPoints_buffered_norm = self.normlization_height(self._vegPoints_buffered)
        self._point_cloud_ = keptPoints


        if self._DSM is None:
            print('\033[35mDSM has not been read or constructed. FLApy will generate automatically\033[0m')
            self.gapFilled = self.fillGap3(self._vegPoints_buffered_norm, bboxBuffered[:2], bboxBuffered[2:], resolution)
            self.get_DSM(self.gapFilled, voxelDownSampling=False, resolution=resolution)



        if self._DEM is None:
            print('\033[35mDEM has not been read or constructed. FLApy will generate automatically. The DEM will be used as DEM due to no DEM detected.\033[0m')
            self._DEM = self._DTM

        self._DSMp = pv.PolyData(self.m2p(self._DSM))
        self._DTMp = pv.PolyData(self.m2p(self._DTM))


        self._SFL = CreateUniformGrid(origin=self.origin, spacing=self.spacing, extent=self.dimensions).apply()
        ext_dtm = ExtractTopography(invert=True).apply(self._SFL, self._DTMp)
        ext_dsm = ExtractTopography(invert=False).apply(self._SFL, self._DSMp)
        ext_merge = ext_dsm.cell_data['Extracted'] * ext_dtm.cell_data['Extracted']
        self._SFL.cell_data['Classification'] = ext_merge


        #NONE
        if self._obsExternalLabel is None:
            cellCenters = self._SFL.cell_centers()
            self._obs = np.array(cellCenters.points)
            self._cellPoints_TerrainNormalization = self.normlization_height(np.array(cellCenters.points))
            self._obs = self._obs[self._SFL.cell_data['Classification'] == 1]
            self._SFL.field_data['OBS_SFL'] = self._obs
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]

        #xyz
        elif self._obsExternalLabel == 1:
            self._cellPoints_TerrainNormalization = self.normlization_height(self._obs)
            self._SFL.field_data['OBS_SFL'] = self._obs
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]

        #xyzv
        elif self._obsExternalLabel == 2:
            self._cellPoints_TerrainNormalization = self.normlization_height(self._obs[:, :3])
            self._SFL.field_data['OBS_SFL'] = self._obs
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]

        #User-defined
        elif self._obsExternalLabel == 3:
            if udXSpacing is None or udYSpacing is None or udZNum is None:
                raise ValueError('The user-defined spacing is not specified.')
            self.gen_OBSbyUserDefined(udXSpacing, udYSpacing, udZNum)
            self._SFL.field_data['OBS_SFL'] = self._obs

            cellCenters = self._SFL.cell_centers()
            self._cellPoints_TerrainNormalization = self.normlization_height(np.array(cellCenters.points))
            self._SFL.cell_data['Z_normed'] = self._cellPoints_TerrainNormalization[:, -1]


        Cdsm = self.m2p(self._DEM.sel(x = slice(bbox[0], bbox[1]), y = slice(bbox[2], bbox[3])))

        self._SFL.field_data['PTS'] = self.point_cloud_buffered
        self._SFL.field_data['DEM'] = Cdsm
        self._SFL.field_data['DTM'] = self.m2p(self._DTM)
        self._SFL.field_data['DSM'] = self.m2p(self._DSM)

        self._SFL.add_field_data([self._obsExternalLabel], 'obsExternalLabel')
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
        z_centersDSM = self.get_ValueByGivenPointsOnRasterMatrix(x_centers.ravel(), y_centers.ravel(), self.DSM)
        z_centersDTM = self.get_ValueByGivenPointsOnRasterMatrix(x_centers.ravel(), y_centers.ravel(), self.DTM)

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
        self._obsExternalLabel = 3



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

    def interp_ByPoints(self, input_points = None, voxelDownSampling = False, resolution = 1):
        # This function is used to interpolate the points.
        # Parameters:
        #   input_points: the points need to be interpolated.
        #   voxelDownSampling: whether to downsample the points.
        #   resolution: the resolution of the matrix interpolated.
        # Return:
        #   xyv_array: the interpolated array.

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


        cell_size = resolution

        X_grid, Y_grid = np.meshgrid(np.arange(X_min, X_max, cell_size), np.arange(Y_min, Y_max, cell_size))
        grid_data = interpolate.griddata(points, Z, (X_grid, Y_grid), method='nearest')

        x_flat = X_grid.ravel()
        y_flat = Y_grid.ravel()
        z_flat = grid_data.ravel()
        xyv_array = np.column_stack((x_flat, y_flat, z_flat))

        return xyv_array


    def fillGap1(self, inPoints, x_bounds, y_bounds, resolution = 1):
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

        num_rows, num_cols = _raster.shape
        transFrom = from_origin(x_min, y_max, resolution, resolution)

        with rasterio.open(str(self._workspace + '/.dsmTemp.tif'), 'w', driver='GTiff', height=num_rows, width=num_cols, count=1, dtype=_raster.dtype, transform=transFrom, crs = 'EPSG:3857') as dst:
            dst.write(_raster, 1)


        gridTran = Grid.from_raster(str(self._workspace + '/.dsmTemp.tif'))
        dsmRead = gridTran.read_raster(str(self._workspace + '/.dsmTemp.tif'))
        pitFilled = gridTran.fill_pits(dsmRead)
        _raster_filled = gridTran.fill_depressions(pitFilled)
        _raster_filled = np.array(_raster_filled)

        yy, xx = np.mgrid[y_max:y_min:-resolution, x_min:x_max:resolution]
        out_array = np.vstack((xx.ravel(), yy.ravel(), _raster_filled.ravel())).T

        return out_array

    @staticmethod
    def fillGap2(inPoints, x_bounds, y_bounds, resolution = 1):
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

        s = generate_binary_structure(2, 2)
        eroded = binary_erosion(_raster, structure=s)
        mask = _raster > eroded
        labeled, num_features = label(mask)

        for i in range(1, num_features + 1):
            boundary = binary_dilation(labeled == i, structure=s) ^ (labeled == i)
            boundary_mean = np.mean(_raster[boundary == 1])
            _raster[labeled == i] = boundary_mean

        yy, xx = np.mgrid[y_max:y_min:-resolution, x_min:x_max:resolution]
        out_array = np.vstack((xx.ravel(), yy.ravel(), _raster.ravel())).T

        return out_array

    @staticmethod
    def fillGap3(inPoints, x_bounds, y_bounds, threshold=1, min_size=10, resolution=1):
        x = inPoints[:, 0]
        y = inPoints[:, 1]
        z = inPoints[:, 2]

        x_min, x_max = x_bounds
        y_min, y_max = y_bounds

        max_size = (x_max - x_min) * (y_max - y_min)

        col_indices = np.digitize(x, np.arange(x_min, x_max, resolution)) - 1
        row_indices = np.digitize(y, np.arange(y_max, y_min, -resolution)) - 1

        _raster = np.full((int((y_max - y_min) / resolution), int((x_max - x_min) / resolution)), -np.inf)
        np.maximum.at(_raster, (row_indices, col_indices), z)
        _raster[_raster == -np.inf] = np.nan

        # Identify forest gaps based on threshold
        chm_layer = np.copy(_raster)
        gaps = (chm_layer <= threshold).astype(int)

        # Close the gaps to connect adjacent forest gaps
        struct_element = generate_binary_structure(2, 2)
        closed_gaps = binary_closing(gaps, structure=struct_element)

        # Label each gap
        labeled, num_features = label(closed_gaps)
        filled_gaps = np.copy(chm_layer)
        gap_mask = np.zeros(chm_layer.shape, dtype=bool)

        for i in range(1, num_features + 1):
            mask_window = labeled == i
            # First dilation
            boundary_1 = binary_dilation(mask_window, structure=struct_element)

            # Second dilation
            boundary_2 = binary_dilation(boundary_1, structure=struct_element)

            # Third dilation
            boundary_3 = binary_dilation(boundary_2, structure=struct_element)

            # Now, get the three layers of boundaries by subtracting
            buffered_boundary_3 = boundary_3 ^ boundary_2
            buffered_boundary_2 = boundary_2 ^ boundary_1
            buffered_boundary_1 = boundary_1 ^ mask_window

            # If you want a single mask that combines all three boundary layers:
            combined_boundary = buffered_boundary_1 | buffered_boundary_2 | buffered_boundary_3

            surrounding_tree_mean = np.nanmax(chm_layer[combined_boundary])

            gap_area = np.sum(mask_window) * resolution ** 2
            if min_size <= gap_area <= max_size:
                filled_gaps[mask_window] = surrounding_tree_mean
                gap_mask[mask_window] = True




        # The next step is to adjust the filled raster based on the filled gaps
        mask_of_filled_gaps = np.array(~gap_mask, dtype=np.uint8)

        # Compute distance to the nearest filled gap for each pixel and get the nearest filled gap value
        distances, indices = distance_transform_edt(mask_of_filled_gaps, return_indices=True)
        nearest_filled_gap_values = filled_gaps[tuple(indices)]

        # Increase the heights of those pixels which are below their nearest filled gap
        mask_to_increase = filled_gaps < nearest_filled_gap_values
        filled_gaps[mask_to_increase] = nearest_filled_gap_values[mask_to_increase]

        yy, xx = np.mgrid[y_max:y_min:-resolution, x_min:x_max:resolution]
        out_array = np.vstack((xx.ravel(), yy.ravel(), filled_gaps.ravel())).T

        return out_array



    @staticmethod
    def nanmean_filter(input_Array):
        output_Array = np.nanmean(input_Array)
        return output_Array

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


        self._DSM = self.p2m(self.interp_ByPoints(TP_vd, voxelDownSampling=voxelDownSampling, resolution=resolution))
        dsmArray = self._DSM.values

        arrayFilled = generic_filter(dsmArray, self.nanmean_filter, [5, 5])

        self._DSM = xr.DataArray(arrayFilled, coords=self._DSM.coords, dims=self._DSM.dims, attrs=self._DSM.attrs)










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

        self._DTM = self.p2m(self.interp_ByPoints(_TP_vd, voxelDownSampling=voxelDownSampling, resolution=resolution))






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

    def __init__(self, filePath = None):
        self.version = 2.0
        self.rootI = os.getcwd()
        self.filePath = filePath


    def read_VTK(self):
        _vtkR = pv.read(self.filePath)
        return _vtkR



