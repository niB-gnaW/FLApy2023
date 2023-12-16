import FLApy as fp
import requests
import os
import pytest
import numpy as np
import xarray as xr


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

@pytest.fixture(scope="module")
def demo_data():
    url = 'https://raw.githubusercontent.com/niB-gnaW/FLApy2023/master/demo_Data/demoData.las'
    filename = 'demoData.las'
    download_file(url, filename)
    yield filename
    os.remove(filename)

@pytest.fixture(scope="module")
def demo_data_raster():
    url = 'https://raw.githubusercontent.com/niB-gnaW/FLApy2023/master/demo_Data/complexTerrain.tif'
    filename = 'demo_data_raster.tif'
    download_file(url, filename)
    yield filename
    os.remove(filename)

@pytest.fixture(scope="module")
def demo_data_csv():
    url = 'https://raw.githubusercontent.com/niB-gnaW/FLApy2023/master/demo_Data/demoCSV.csv'
    filename = 'demoCSV.csv'
    download_file(url, filename)
    yield filename
    os.remove(filename)





def test_read_LasData(demo_data):
    inLasFile = demo_data
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(inLasFile)
    assert isinstance(site.point_cloud, np.ndarray) is True and site.point_cloud.shape[1] == 3

def test_read_RasterData(demo_data_raster):
    inRasterFile = demo_data_raster
    site = fp.DataManagement.StudyFieldLattice()
    site.read_RasterData(inRasterFile)
    assert isinstance(site.DSM, xr.DataArray) is True

def test_read_csvData(demo_data_csv):
    inCSVFile = demo_data_csv
    site = fp.DataManagement.StudyFieldLattice()
    site.read_csvData(inCSVFile)
    assert isinstance(site.OBS, np.ndarray) is True and site.OBS.shape[1] == 4

def test_m2p():
    one_mesh = xr.DataArray(np.ones((10, 10)), dims=('x', 'y'))
    outPoints = fp.DataManagement.StudyFieldLattice.m2p(one_mesh)
    assert isinstance(outPoints, np.ndarray) is True and outPoints.shape[1] == 3

def test_p2m():
    X = np.arange(0, 10, 1)
    Y = np.arange(0, 10, 1)
    XY = np.meshgrid(X, Y)
    one_points = np.vstack((XY[0].flatten(), XY[1].flatten(), np.ones((100,)))).T
    outMesh = fp.DataManagement.StudyFieldLattice.p2m(one_points, 1)
    assert isinstance(outMesh, xr.DataArray) is True

