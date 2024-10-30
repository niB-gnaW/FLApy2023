from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='FLApy',
    version='2.1.2',
    description='Forest Light availability heterogeneity Analysis in Python',
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    author='Bin Wang',
    author_email='wb931022@hotmail.com',
    url='https://github.com/niB-gnaW/FLApy2023',
    packages=find_packages(),
    py_modules=['FLApy.__init__', 'FLApy.DataManagement', 'FLApy.LAcalculator', 'FLApy.LAHanalysis', 'FLApy.Visualization'],
    classifiers=['Programming Language :: Python :: 3.11', 'License :: OSI Approved :: MIT License', 'Operating System :: OS Independent',],
    python_requires='>=3.11',
    install_requires=['numpy==1.26.4',
                      'scipy',
                      'matplotlib',
                      'open3d',
                      'pyvista',
                      'PVGeo',
                      'laspy',
                      'pandas',
                      'tqdm',
                      'miniball',
                      'xarray',
                      'joblib',
                      'scikit-learn',
                      'seaborn',
                      'vtk',
                      ],
)

