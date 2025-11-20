import subprocess
import sys
import urllib.request
import platform
import os


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
            whl_url = 'https://raw.githubusercontent.com/niB-gnaW/FLApy2023/master/dependenciesFLApy/naturalneighbor-0.2.1-cp311-cp311-win_amd64.whl'
        elif os_type == 'Darwin':
            print('MacOS is not supported yet')
        else:
            print(f'Unsuported OS type: {os_type}')
            return

        install_whl_from_github(whl_url)


if __name__ == "__main__":
    check_and_install_dependencies()