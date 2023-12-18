import FLApy as fp
import requests
import os
import pytest
import pandas as pd

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

def test_LAH_analysis(demo_data):
    inLasFile = demo_data
    site = fp.DataManagement.StudyFieldLattice()
    site.read_LasData(inLasFile)
    site.gen_SFL(bbox=[100, 200, 100, 200], resolution=1, obsType=3, udXSpacing=20, udYSpacing=20, udZNum=2)
    siteLA = fp.LAcalculator.LAcalculator(site)
    siteLA.computeBatch(multiPro = 'joblib')
    siteLAH = fp.LAHanalysis.LAH_analysis(siteLA)
    result = siteLAH.com_allLAH(givenHeight=1.5)
    assert isinstance(result, pd.DataFrame) is True
    result2 = siteLAH.com_allLAH(mode='v')
    assert isinstance(result2, pd.DataFrame) is True
    result3 = siteLAH.com_allLAH(givenHeight=[1,20,30])
    assert isinstance(result3, pd.DataFrame) is True
    result4 = siteLAH.com_allLAH_subplot(bbox=[100, 200, 100, 200], subNum=16)
    assert isinstance(result4, pd.DataFrame) is True



def test_sigmoid_func():
    assert fp.LAHanalysis.sigmoid_func(1, 1, 1) == 50

def test_sigmoid_func2():
    assert fp.LAHanalysis.sigmoid_func2(1, 1, 1, 1, 1) == 1.5

