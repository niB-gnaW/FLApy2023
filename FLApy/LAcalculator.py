# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import itertools
import os
import time

from scipy.spatial import KDTree
from joblib import Parallel, delayed
from FLApy.DataManagement import StudyFieldLattice

class LAcalculator(StudyFieldLattice):
    # The class is used to calculate the LA based on the input SFL.
    # Parameters:
    # inDataContainer: the input SFL data container.
    # lensMapRadius: the radius of the lens map. The default value is 500.
    # pointSizeRange: the range of the point size in the lens map. The default value is (0.5, 7).
    #                 More near points will be larger than the far points. That is, the most near points will be 7, and the most far points will be 0.5.
    #                 the unit is pixel. It should be calibrated according to the actual situation and use some field data to optimize.

    def __init__(self,
                 inDataContainer = None,
                 lensMapRadius = 500,
                 pointSizeRange = (0.5, 7),
                 centerTerrain = True,
                 downSample = True):

        if inDataContainer is None:
            raise ValueError('Please input a SFL data container!')

        elif inDataContainer is not None and os.path.isfile(inDataContainer) is True:
            self._DataContainer = pv.read(inDataContainer)
            self.tempSFL = str(self._DataContainer.field_data['temPath'][0])

        elif inDataContainer is not None and os.path.isfile(inDataContainer) is False:
            self._DataContainer = inDataContainer._SFL
            self.tempSFL = str(self._DataContainer.field_data['temPath'][0])

        self._mapRadius = lensMapRadius
        self.__obsExternalLabel = str(self._DataContainer.field_data['OBS_Type'][0])

        self.pointSizeRange = pointSizeRange

        self.pointsBuffered = self._DataContainer.field_data['PTS']
        if downSample is True:
            self.pointsBuffered = self.voxelDownsampling(self.pointsBuffered, 0.5)



        self.mergeTerrain = np.concatenate((self._DataContainer.field_data['DEM'], self._DataContainer.field_data['DTM']), axis=0)
        self._obsSet = self._DataContainer.field_data['OBS_SFL']

        self.centerTerrain = centerTerrain

        if centerTerrain is True:
            obsXcenter = np.mean(self._obsSet[:, 0])
            obsYcenter = np.mean(self._obsSet[:, 1])
            obsZcenter = np.mean(self._obsSet[:, 2])
            obsCenter = np.array([obsXcenter, obsYcenter, obsZcenter])
            self.centerTerrainDrawed = self.drawIn_terrain(self.mergeTerrain, obsCenter)


    def pol2cart(self, rho, phi):
        # Convert polar coordinates (rho, phi) to Cartesian coordinates (x, y).
        # Parameters:
        #   rho: the radius of the point in the lens map.
        #   phi: the azimuth of the point in the lens map.
        # Return:
        #   x: the x coordinate of the point in the lens map.
        #   y: the y coordinate of the point in the lens map.

        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y

    def sph2cart(self, theta, phi, r):
        # Convert spherical coordinates (theta, phi, r) to Cartesian coordinates (x, y, z).
        # Parameters:
        #   theta: the zenith angle of the point in the lens map.
        #   phi: the azimuth of the point in the lens map.
        #   r: the radius of the point in the lens map.
        # Return:
        #   x: the x coordinate of the point in the lens map.
        #   y: the y coordinate of the point in the lens map.
        #   z: the z coordinate of the point in the lens map.

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    def sph2pol(self, theta, phi, mapRadius = 500):
        # Convert spherical coordinates (theta, phi, r) to polar coordinates (rho, phi).
        # Parameters:
        #   theta: the zenith angle of the point in a spherical coordinate system.
        #   phi: the azimuth of the point in a spherical coordinate system.
        # Return:
        #   rho: the radius of the point in a polar coordinate system.
        #   phi: the azimuth of the point in a polar coordinate system.

        r = mapRadius*np.ones(len(phi))
        rho = r * np.sin(theta) / (1 + np.cos(theta))
        phi = phi
        return rho, phi

    def cart2sph(self, x, y, z):
        # Convert Cartesian coordinates (x, y, z) to spherical coordinates (theta, phi, r).
        # Parameters:
        #   x: the x coordinate of the point in a Cartesian coordinate system.
        #   y: the y coordinate of the point in a Cartesian coordinate system.
        #   z: the z coordinate of the point in a Cartesian coordinate system.
        # Return:
        #   theta: the zenith angle of the point in a spherical coordinate system.
        #   phi: the azimuth of the point in a spherical coordinate system.
        #   r: the radius of the point in a spherical coordinate system.

        coords = np.vstack((x, y, z)).transpose()
        r = np.sqrt(np.sum((coords) ** 2, axis=1))
        theta = np.arccos(z / (r))
        phi = np.arctan2(y, x)
        #r = self._mapRadius*np.ones(len(phi))
        return r, theta, phi

    def cart2pol(self, x, y):
        # Convert Cartesian coordinates (x, y) to polar coordinates (r, theta).
        # Parameters:
        #   x: the x coordinate of the point in a Cartesian coordinate system.
        #   y: the y coordinate of the point in a Cartesian coordinate system.
        # Return:
        #   r: the radius of the point in a polar coordinate system.
        #   theta: the azimuth of the point in a polar coordinate system.

        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)
        return r, theta

    def plot_ReferenceMap(self, inMap = None):
        # Plot the reference map.
        # Parameters:
        #   inMap: the input map. The default value is None.
        # Return:
        #   Show the reference map.
        if inMap is None:
            inMap = self.cMapAll
        else:
            inMap = inMap
        plt.imshow(inMap, interpolation='nearest')
        plt.colorbar()
        title_text = 'Light Availability (SVF_flat = {:.2f}, SVF_hemi = {:.2f})'
        plt.title(title_text.format(self._SVF_[0], self._SVF_[1]))
        return plt.show()


    def referenceGrid(self):
        # Generate a reference grid.
        # Return:
        #   grid_image: the reference grid.
        #   gridCoord: the coordinates of the reference grid.

        radius = self._mapRadius
        xgrid, ygrid = np.meshgrid(np.arange(1, radius * 2 + 1), np.arange(1, radius * 2 + 1))
        xgrid = (xgrid - radius - 0.5)
        ygrid = (ygrid - radius - 0.5)
        gridCoord = np.column_stack((xgrid.ravel(), ygrid.ravel()))
        grid_rad, grid_theta = self.cart2pol(xgrid.ravel(), ygrid.ravel())
        grid_rad[grid_rad > radius] = np.nan

        grid_image = np.ones((radius * 2, radius * 2))
        imdx = np.reshape(np.isnan(grid_rad), grid_image.shape)
        grid_image[imdx] = 0
        return grid_image, gridCoord


    def drawIn_vegPoints(self, inPoints, inObs, pointSizeRangeSet):
        # Draw the vegetation points.
        # Parameters:
        #   inPoints: the input points (x, 3).
        #   inObs: the input observations (1, 3).
        #   pointSizeRangeSet: the point size range set.
        # Return:
        #   image2ev: the map occupied by the vegetation points.


        image2ev, gridCoord = self.referenceGrid()
        pointSizeRangeMin = min(pointSizeRangeSet)
        pointSizeRangeMax = max(pointSizeRangeSet)
        vegCBOed = inPoints - inObs
        vegCBOed = vegCBOed[vegCBOed[:, 2] > 0]
        if len(vegCBOed) == 0:
            self.vegCoverMap = image2ev
        else:
            veg2sph_r, veg2sph_theta, veg2sph_phi = self.cart2sph(vegCBOed[:, 0], vegCBOed[:, 1], vegCBOed[:, 2])
            veg2pol_rho, veg2pol_phi = self.sph2pol(veg2sph_theta, veg2sph_phi)
            tx, ty = self.pol2cart(veg2pol_rho, veg2pol_phi)
            datcart = np.column_stack((tx, ty))
            gridCoordCellSize = np.abs(gridCoord[1][0]-gridCoord[0][0])
            Dmin, Dmax = np.min(veg2sph_r), np.max(veg2sph_r)
            if Dmin == Dmax:
                position = 0
            else:
                position = (veg2sph_r - Dmin) / (Dmax - Dmin)
            rmax = (pointSizeRangeMax / 2) * gridCoordCellSize
            rmin = (pointSizeRangeMin / 2) * gridCoordCellSize
            told = (((1 - position) * (rmax - rmin)) + rmin)
            tree = KDTree(gridCoord)
            pointsWithin = tree.query_ball_point(x=datcart, r=told)
            indx = np.array(list(itertools.chain.from_iterable(pointsWithin)))
            indx = np.unique(indx)
            ndx = np.zeros(gridCoord[:, 0].size, dtype=bool)

            if len(indx) == 0:
                self.vegCoverMap = image2ev
            else:
                ndx[indx] = True
                imdx = np.reshape(ndx, image2ev.shape)
                image2ev[imdx] = 0
                self.vegCoverMap = image2ev
        return self.vegCoverMap

    def drawIn_terrain(self, inTerrain, inObs):
        # Draw the terrain.
        # Parameters:
        #   inTerrain: the input terrain (x, 3).
        #   inObs: the input observations (1, 3).
        # Return:
        #   image2ev: the map occupied by the terrain.

        image2ev, gridCoord = self.referenceGrid()
        ndx = np.zeros(gridCoord[:, 0].size, dtype=bool)
        terCBOed = inTerrain - inObs
        terCBOed = terCBOed[terCBOed[:, 2] > 0]
        ter2sph_r, ter2sph_theta, ter2sph_phi = self.cart2sph(terCBOed[:, 0], terCBOed[:, 1], terCBOed[:, 2])
        ter2pol_rho, ter2pol_phi = self.sph2pol(ter2sph_theta, ter2sph_phi)
        bins = 360
        gridCoordRho, gridCoordPhi = self.cart2pol(gridCoord[:, 0], gridCoord[:, 1])
        ndx = update_terrain_indices(gridCoordPhi, ter2sph_phi, ter2pol_rho, bins, gridCoordRho, ndx)
        imdx = np.reshape(ndx, image2ev.shape)
        image2ev[imdx] = 0
        self.terCoverMap = image2ev
        return self.terCoverMap


    def cal_LA(self, image2ev):
        # Calculate the LA. Including the flat LA and the hemispherical LA.
        # Parameters:
        #   image2ev: the map occupied by the vegetation points and the terrain.
        # Return:
        #   LA: the LA.

        _, gridCoord = self.referenceGrid()
        radius = self._mapRadius
        gridCoordRho, gridCoordPhi = self.cart2pol(gridCoord[:, 0], gridCoord[:, 1])

        image2ev = image2ev.ravel()

        n = 9
        lens_profile_tht = np.arange(0, 91, 10)
        lens_profile_rpix = np.linspace(0, 1, n + 1)  # Linear profile per 10deg zenith angle
        ring_tht = np.linspace(0, 90, n + 1)  # Zenith angle in degree, i.e., zenith is tht = 0; horizon is tht = 90
        ring_radius = np.interp(ring_tht, lens_profile_tht, lens_profile_rpix * radius)


        num_rings = len(ring_radius) - 1
        white_to_all_ratio = np.empty(num_rings) * np.nan
        surface_area_ratio_hemi = np.empty(num_rings) * np.nan
        surface_area_ratio_flat = np.empty(num_rings) * np.nan


        for rix in range(num_rings):
            inner_radius = ring_radius[rix]
            outer_radius = ring_radius[rix + 1]
            relevant_pix = np.where((gridCoordRho > inner_radius) & (gridCoordRho <= outer_radius))[0]

            white_to_all_ratio[rix] = np.sum(image2ev[relevant_pix] == 1) / len(relevant_pix)
            surface_area_ratio_hemi[rix] = np.cos(np.radians(ring_tht[rix])) - np.cos(np.radians(ring_tht[rix + 1]))
            surface_area_ratio_flat[rix] = np.sin(np.radians(ring_tht[rix + 1])) ** 2 - np.sin(np.radians(ring_tht[rix])) ** 2


        flat_SVF = np.sum(white_to_all_ratio * surface_area_ratio_flat)
        hemi_SVF = np.sum(white_to_all_ratio * surface_area_ratio_hemi)

        return (flat_SVF, hemi_SVF)

    def computeSingle(self, index):
        # Compute the LA for a single observation.
        # Parameters:
        #   index: the index of the observation.
        # Return:
        #   LA: the LA.
        obsIn = self._obsSet[index]

        result4oneObs = self.drawIn_vegPoints(self.pointsBuffered, obsIn, self.pointSizeRange)
        result4oneObsTer = self.drawIn_terrain(self.mergeTerrain, obsIn)
        mergeResult = result4oneObsTer * result4oneObs

        self.cMap4veg = result4oneObs
        self.cMap4ter = result4oneObsTer
        self.cMapAll = mergeResult
        SVF = self.cal_LA(self.cMapAll)
        self._SVF_ = SVF
        return (SVF[0], SVF[1])

    def computeSingleFAST(self, index):
        # Compute the LA for a single observation quickly.
        # Parameters:
        #   index: the index of the observation.
        # Return:
        #   LA: the LA.

        obsIn = self._obsSet[index]

        result4oneObs = self.drawIn_vegPoints(self.pointsBuffered, obsIn, self.pointSizeRange)
        result4oneObsTer = self.centerTerrainDrawed
        mergeResult = result4oneObsTer * result4oneObs

        self.cMap4veg = result4oneObs
        self.cMap4ter = result4oneObsTer
        self.cMapAll = mergeResult
        SVF = self.cal_LA(self.cMapAll)
        self._SVF_ = SVF
        return (SVF[0], SVF[1])

    def computeBatch(self, save = None, multiPro = 'p_map', CPU_count = None):
        # Compute the LA for a batch of observations.
        # Parameters:
        #   save: the path to save the LA. If None, the LA will be saved in a temporary file.
        #   multiPro: the method to use for multiprocessing. 'joblib' for joblib, 'p_tqdm' for p_tqdm.
        #   CPU_count: the number of CPUs to use. If None, all CPUs will be used.

        if CPU_count is None:
            numCPU = os.cpu_count() - 1
        else:
            numCPU = int(CPU_count)

        obsIdx = np.arange(len(self._obsSet))

        if multiPro == 'joblib':
            time_start = time.time()
            print('\033[32mProcessing started!'+ 'Number of obs:'+ str(len(obsIdx)) + '\033[0m')

            if self.centerTerrain is False:
                SVFset = Parallel(n_jobs=numCPU, verbose=100)(delayed(self.computeSingle)(arg) for arg in obsIdx)
            else:
                SVFset = Parallel(n_jobs=numCPU, verbose=100)(delayed(self.computeSingleFAST)(arg) for arg in obsIdx)

            time_end = time.time()
            print('\033[32mProcessing finished!'+ str(time_end - time_start)+'s' + '\033[0m')

        if multiPro == 'p_map':
            from p_tqdm import p_map
            time_start = time.time()
            print('\033[32mProcessing started!' + 'Number of obs:' + str(len(obsIdx)) + '\033[0m')

            if self.centerTerrain is False:
                SVFset = p_map(self.computeSingle, obsIdx, num_cpus = numCPU, desc='Batch processing', ncols=100)
            else:
                SVFset = p_map(self.computeSingleFAST, obsIdx, num_cpus=numCPU, desc='Batch processing', ncols=100)
            time_end = time.time()
            print('\033[32mProcessing finished!' + str(time_end - time_start) + 's' + '\033[0m')

        SVFcellData = np.array(SVFset)
        self._DataContainer.field_data['SVF_flat'] = SVFcellData[:, 0]
        self._DataContainer.field_data['SVF_hemi'] = SVFcellData[:, 1]



        if save is None:
            self._DataContainer.save(self.tempSFL)
        elif save is not None:
            self._DataContainer.save(save)

def update_terrain_indices(gridCoordPhi, ter2sph_phi, ter2pol_rho, bins, gridCoordRho, ndx):
    # Update the indices of the terrain points.
    # Parameters:
    #   gridCoordPhi: the azimuth angle of the grid points.
    #   ter2sph_phi: the azimuth angle of the terrain points.
    #   ter2pol_rho: the radius of the terrain points.
    #   bins: the number of bins.
    #   gridCoordRho: the radius of the grid points.
    #   ndx: the index of the grid point.
    # Return:
    #   ter2sph_phi: the azimuth angle of the terrain points.
    #   ter2pol_rho: the radius of the terrain points.

    if len(ter2sph_phi) == 0:
        return ndx

    else:
        from tqdm import tqdm

        for idx in tqdm(range(bins)):
            tbinMin = np.deg2rad(-180) + np.deg2rad(idx)
            tbinMax = np.deg2rad(-180) + np.deg2rad(idx + 1)
            keptPointsRho = ter2pol_rho[(ter2sph_phi >= tbinMin) & (ter2sph_phi < tbinMax)]

            if len(keptPointsRho) != 0:
                keptPointsRhoMin = np.min(keptPointsRho)
            else:
                Aloop = 2
                while len(keptPointsRho) == 0:
                    if Aloop > bins:
                        keptPointsRho = 0
                        break
                    tbinMax = np.deg2rad(-180) + np.deg2rad(idx + Aloop)
                    keptPointsRho = ter2pol_rho[(ter2sph_phi >= tbinMin) & (ter2sph_phi < tbinMax)]
                    Aloop = Aloop + 1
                keptPointsRhoMin = np.min(keptPointsRho)

            keptGridRhoIdx = np.where((gridCoordPhi >= tbinMin) & (gridCoordPhi < tbinMax))[0]
            keptGridRho = gridCoordRho[(gridCoordPhi >= tbinMin) & (gridCoordPhi < tbinMax)]
            keptGridRhoIdx = keptGridRhoIdx[keptGridRho >= keptPointsRhoMin]
            ndx[keptGridRhoIdx] = True
        return ndx




if __name__ == "__main__":
    pass