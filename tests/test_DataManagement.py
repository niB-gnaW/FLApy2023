import FLApy as fp
import requests
import os
import pytest
import numpy as np

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





def test_read_LasData(demo_data):
    inLasFile = demo_data
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(inLasFile)
    assert isinstance(site.point_cloud, np.ndarray) is True and site.point_cloud.shape[1] == 3

def test_read_RasterData(demo_data_raster):
    inRasterFile = demo_data_raster
    site = fp.DataManagement.StudyFieldLattice()
    site.read_RasterData(inRasterFile)
    assert isinstance(site.raster_data, np.ndarray) is True and site.raster_data.shape[1] == 3