import FLApy as fp
import requests
import os
import pytest
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

def test_LAcalculator(demo_data):
    inLasFile = demo_data
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(inLasFile)
    site.gen_SFL(bbox=[100, 200, 100, 200], resolution=1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=2)
    siteLA = fp.LAcalculator.LAcalculator(site)
    siteLA.computeBatch(multiPro = 'joblib', CPU_count=4)
    assert (isinstance(siteLA._DataContainer.field_data['SVF_flat'], pv.pyvista_ndarray) is True and
            isinstance(siteLA._DataContainer.field_data['SVF_hemi'], pv.pyvista_ndarray) is True)