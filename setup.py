from setuptools import setup, find_packages


with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='FLApy',
    version='2.1.4',
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
                      'scipy==1.16.3',
                      'matplotlib==3.10.7',
                      'open3d==0.19.0',
                      'pyvista==0.46.4',
                      'PVGeo==3.0.2',
                      'laspy==2.6.1',
                      'pandas==2.3.3',
                      'tqdm==4.67.1',
                      'miniball==1.2.0',
                      'xarray==2025.11.0',
                      'joblib==1.5.2',
                      'scikit-learn==1.7.2',
                      'seaborn==0.13.2',
                      'vtk==9.3.1',
                      ],
)

