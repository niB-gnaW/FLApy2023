from setuptools import setup, find_packages


with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='FLApy',
    version='1.131',
    description='Forest Light availability heterogeneity Analysis in Python',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author='Bin Wang',
    author_email='wb931022@hotmail.com',
    url='https://github.com/niB-gnaW/FLApy2023',
    packages=find_packages(),
    package_data={'FLApy': ['dependencies/naturalneighbor-0.2.1-cp38-cp38-win_amd64.whl']},
    py_modules=['FLApy.__init__', 'FLApy.DataManagement', 'FLApy.LAcalculator', 'FLApy.LAHanalysis', 'FLApy.Visualization'],
    classifiers=['Programming Language :: Python :: 3.8', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent',],
    python_requires='>=3.7',
    install_requires=['numpy==1.21.2',
                      'scipy==1.6.0',
                      'matplotlib',
                      'open3d',
                      'pyvista',
                      'PVGeo',
                      'laspy==1.7.0',
                      'pandas',
                      'tqdm',
                      'p_tqdm',
                      'miniball',
                      'rasterio',
                      'xarray==0.19.0',
                      'joblib',
                      'pdal'],
    dependency_links=['file:dependencies/naturalneighbor-0.2.1-cp38-cp38-win_amd64.whl'],
)

