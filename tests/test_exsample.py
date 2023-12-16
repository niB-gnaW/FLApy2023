import requests
import os
import pytest
import FLApy as fp
import pandas as pd
import pyvista as pv


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

# 测试读取 LasData 的功能
def test_read_LasData(demo_data):
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(demo_data)
    assert site.point_cloud.shape[1] == 3


def test_gen_SFL(demo_data):
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(demo_data)
    bbox = [100, 200, 100, 200]
    site.gen_SFL(bbox, resolution=1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=2)
    assert hasattr(site, '_SFL') is True


def test_LAcalculator(demo_data):
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(demo_data)
    bbox = [100, 200, 100, 200]
    site.gen_SFL(bbox, resolution=1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=2)
    siteLA = fp.LAcalculator.LAcalculator(site)
    siteLA.computeBatch(multiPro='joblib', CPU_count=4)
    assert isinstance(siteLA._DataContainer.field_data['SVF_flat'], pv.pyvista_ndarray) is True


def test_LAH_analysis(demo_data):
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(demo_data)
    bbox = [100, 200, 100, 200]
    site.gen_SFL(bbox, resolution=1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=2)
    siteLA = fp.LAcalculator.LAcalculator(site)
    siteLA.computeBatch(multiPro='joblib', CPU_count=4)
    siteLAH = fp.LAHanalysis.LAH_analysis(siteLA)
    result = siteLAH.com_allLAH(givenHeight=1.5)
    assert isinstance(result, pd.DataFrame) is True

