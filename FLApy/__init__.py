name = "FLApy"
import os
import vtk
import subprocess
import sys
import urllib.request
import platform
os.environ['USE_PYGEOS'] = '0'

def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_whl_from_github(whl_url):
    whl_file = os.path.basename(whl_url)
    try:
        urllib.request.urlretrieve(whl_url, whl_file)
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', whl_file])
    except Exception as e:
        print(f"An error occurred: {e}")

def check_and_install_dependencies():
    print('Checking and installing dependencies...')
    if not is_package_installed('naturalneighbor'):
        os_type = platform.system()
        if os_type == 'Windows':
            whl_url = 'https://raw.githubusercontent.com/niB-gnaW/FLApy2023/master/dependenciesFLApy/naturalneighbor-0.2.1-cp38-cp38-win_amd64.whl'
        elif os_type == 'Darwin':
            whl_url = 'https://raw.githubusercontent.com/niB-gnaW/FLApy2023/master/dependenciesFLApy/naturalneighbor-0.2.1-cp38-cp38-macosx_10_15_x86_64.whl'
        else:
            print(f'Unsuported OS type: {os_type}')
            return

        install_whl_from_github(whl_url)

check_and_install_dependencies()

from FLApy import DataManagement, LAcalculator, LAHanalysis, Visualization
vtk.vtkObject.GlobalWarningDisplayOff()

import pyvista

FLApy_theme = pyvista.themes.DefaultTheme()
FLApy_theme.background = 'white'
FLApy_theme.title = 'FLApy'
FLApy_theme.font.family = 'Times'
FLApy_theme.font.color = 'black'
FLApy_theme.axes.x_color = 'black'
FLApy_theme.axes.y_color = 'black'
FLApy_theme.axes.z_color = 'black'
FLApy_theme.outline_color = 'black'


pyvista.global_theme.load_theme(FLApy_theme)
