![](https://github.com/niB-gnaW/FLApy2023/blob/master/docs/logo.png)
# Forest Light Analyzer Python package: FLApy 
![PyPI - Downloads](https://img.shields.io/pypi/dm/FLApy?label=Downloads&style=flat-square)



## General description
Forest Light Analyzer python package (FLApy) is a python package for assessing
light availability（LA） condition and analysing Light availability Heterogeneity at
any observers within forest using Airborne Laser Scanning data and analysis the change or
heterogeneity of LA over spatial scale. At the same time, FLApy can also be used to analysing
the Light Availability Heterogeneity (LAH) of forest, and calculate a series of indicators to
describe the 3D LAH of forest at different spatial scales.

## Demo Data
The demo Data can be found in the demo folder.

# Getting started
FLApy is recommended to be installed in a virtual environment (Python version: 3.8.6).
GDAL(http://www.gdal.org/) and C++ 14.0 are required to be installed before installing FLApy.

## Installation
```
pip install FLApy      # Install the package
```

## Usage
The FLApy workflow is as follows:

![](https://github.com/niB-gnaW/FLApy2023/blob/master/docs/WorkFlow_FLApy.png)

A simple example can be found in the [Simple guidance](https://github.com/niB-gnaW/FLApy/blob/master/examples/A_simple_guidance.ipynb).

### Import FLApy

```
import FLApy as fp
```
### Read data
In general, the point cloud data is required, and the DSM, DEM and DTM are optional.
The FLApy package can read the point cloud data in the LAS format. 
And the DSM and DTM can be produced automatically from the point cloud data when the SFL is generated.
Yet, the third-party tools are recommended to produce these raster data.
Especially, the study area is large, and the point cloud data is too big, the DSM, DEM and DTM can be produced by using the [lastools](https://rapidlasso.com/lastools/).
Besides, if the study area locates in a mountainous area, the DEM is recommended to be provided. 
```
site = fp.DataManagement.StudyFieldLattice()
site.read_LasData('/your/path/to/point.las')    
```


### generate A Study-Field Lattice (SFL)
SFL is a data container for storing the information of each voxel in the study area. And all data processing and analysis in FLApy are based on the SFL.
`gen_SFL` is a function to generate the SFL. It needs a study area extent determined by `[min_X, max_X, min_Y, max_Y]`. 
The `resolution` is the size of each voxel in the SFL. The unit is meter.
```
site.gen_SFL([xmin, xmax, ymin, ymax], resolution=1)
```

### Compute the Light Availability (LA) at voxels
The LA is calculated at each voxel in the SFL. The LA is calculated by the `LAcalculator` class.
The voxel-traverse method is used to calculate the LA within SFL if the observation type is default.
A cutting-edge parallel computing method is used to calculate the LA within SFL, it can save a lot of time if the study area is large.
The actual processing time depends on the number of CPU cores.

```
siteLA = fp.LAcalculator.LAcalculator(site)
siteLA.computeBatch(siteLA)
```

### 3D Light Availability Heterogeneity (LAH) analysis
The LAH is calculated by the `LAH_calculator` class. The `field` is the LA field to be calculated.
If the path of saving is not provided, the LAH will be saved in the same path of the SFL.
```
siteLAH = fp.LAHanalysis.LAH_calculator(siteLA)
results = siteLAH.com_allLAH()
print(results)
```

### Visualization
The `visualize` module is used to visualize the data in the SFL. The `field` is the field to be visualized. 
The module contains the following functions:
- `vis_3Dpoint` is used to visualize the 3D point cloud data.
- `vis_Raster` is used to visualize the raster data.
- `vis_Figures` is used to visualize the summary of the LAH analysis across the four spatial scales.
```
fp.Visualization.vis_Figures(siteLAH)
```
![](https://github.com/niB-gnaW/FLApy2023/blob/master/docs/Figure.png)

### LAH indicators
All support for calculation 3D-LAH indicator system.

|             Indicators              |   Scale    | Abbreviation |
|:-----------------------------------:|:----------:|:------------:|
|               Average               |   Voxel    |   AVE_Vox    |
|         Standard_deviation          |   Voxel    |   STD_Vox    |
|      Coefficient_of_variation       |   Voxel    |    CV_Vox    |
|                Range                |   Voxel    |   RAN_Vox    |
|       Spatial_autocorrelation       |   Voxel    |   SAC_Vox    |
|              Diversity              |   Voxel    |   DIV_Vox    |
|          Gini_coefficient           |   Voxel    |   GINI_Vox   |
|       Light_attenuation_rate        |  Vertical  |   LAR_Ver    |
|     Height_of_inflection_point      |  Vertical  |   HIP_Ver    |
| Relative_height_of_inflection_point |  Vertical  |   HIPr_Ver   |
|          Convex_hull_area           |  Vertical  |   ACH_Ver    |
|               Average               | Horizontal |   AVE_Hor    |
|         Standard_deviation          | Horizontal |   STD_Hor    |
|Height of max Standard deviation|Horizontal|  STDmh_Hor   |
|Relative height of max Standard deviation|Horizontal|  STDmhr_Hor  |
|      Coefficient_of_variation       | Horizontal |    CV_Hor    |
|                Range                | Horizontal |   RAN_Hor    |
|       Spatial_autocorrelation       | Horizontal |   SAC_Hor    |
|              Diversity              | Horizontal |   DIV_Hor    |
|          Gini_coefficient           | Horizontal |   GINI_Hor   |
|             Hot_volume              | 3D_Cluster |   HVOL_3D    |
|             Cold_volume             | 3D_Cluster |   CVOL_3D    |
|         Relative_hot_volume         | 3D_Cluster |   HVOLr_3D   |
|        Relative_cold_volume         | 3D_Cluster |   CVOLr_3D   |
|     Volume_ratio_of_hot_to_cold     | 3D_Cluster |   VRH2C_3D   |
|         Largest_hot_volume          | 3D_Cluster |    LHV_3D    |
|         Largest_cold_volume         | 3D_Cluster |    LCV_3D    |
|            Hot_abundance            | 3D_Cluster |    HAB_3D    |
|           Cold_abundance            | 3D_Cluster |    CAB_3D    |
|         Hot_volume_average          | 3D_Cluster |    HVA_3D    |
|         Cold_volume_average         | 3D_Cluster |    CVA_3D    |
|            Hot_cohesion             | 3D_Cluster |    HCO_3D    |
|            Cold_cohesion            | 3D_Cluster |    CCO_3D    |
|          Hotspot related circumscribing sphere| 3D_Cluster |    HCC_3D    |
|          Coldspot related circumscribing sphere| 3D_Cluster |    CCC_3D    |
|           Hot_shape_index           | 3D_Cluster |    HSI_3D    |
|          Cold_shape_index           | 3D_Cluster |    CSI_3D    |


# Notation
The FLApy package is developed by Bin Wang (wb931022@hotmail.com)


# Authors
Bin Wang<sup>1, 2</sup>,
Cameron Proctor<sup>2</sup>,
Zhiliang Yao<sup>3, 4</sup>,
Ninglv Li<sup>1</sup>,
Qifei Chen<sup>1</sup>,
Wenjun Liu<sup>1</sup>,
Suhui Ma<sup>1</sup>,
Chuanbao Jing<sup>1</sup>,
Zhaoyu Zhou<sup>1</sup>,
Weihong Liu<sup>1</sup>,
Yufeng Ma<sup>1</sup>,
Zimu Wang<sup>1</sup>,
Zhiming Zhang<sup>1</sup>,
Luxiang Lin<sup>3, 5</sup>

1. School of Ecology and Environmental Sciences, Yunnan University, Kunming, 650500, China
2. School of the Environment, University of Windsor, Windsor, N9B 3P4, Canada
3. CAS Key Laboratory of Tropical Forest Ecology, Xishuangbanna Tropical Botanical Garden, Chinese Academy of Sciences, Kunming, China
4. University of Chinese Academy of Sciences, Beijing, China
5. National Forest Ecosystem Research Station at Xishuangbanna, Mengla, Yunnan, China

Author mail: Bin Wang (wb931022@hotmail.com); Zhiming Zhang (zzming76@ynu.edu.cn)

# Acknowledgements
This research is supported by the National Natural Science Foundation of China (32260291), The Second Tibetan Plateau Scientific Expedition and Research (STEP) program (2019QZKK0308), and the Joint Fund of the National Natural Science Foundation of China-Yunnan Province (U1902203). In addition, it has received strong support from The Project for Talent and Platform of Science and Technology in Yunnan Province Science and Technology Department (202205AM070005), the Major Program for Basic Research Project of Yunnan Province (202101BC070002) and the Key Research and Development Program of Yunnan Province (No. 202303AC100009). We also thank the Ailao Mountain Nature Reserve Ecological Station for field work support, and the Southeast Asian Biodiversity Institute (151C53KYSB20200019) for laboratory platforms. Besides, we acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC, RGPIN-2022-04861). Finally, we would like to express our sincere thanks to Prof. Jiajia Liu of Fudan University, Prof. Hans De Boeck of the University of Antwerp, and Dr. Suhui Ma and Dr. Chuanbao Jing of Yunnan University for their valuable suggestions and guidance on the writing of this work.

# Authors' contributions
- Bin Wang, Zhiming Zhang, and Luxiang Lin conceived of the idea and contributed to the concept, and Bin Wang co-designed the package with Cameron Proctor.
- The Python package was developed by Bin Wang and tested by Zhiliang Yao, Qifei Chen and Ninglv Li.
- Zhiliang Yao and Ninglv Li worked together to create the simulated forest.
- Weihong Liu and Zhaoyu Zhou conducted the ground forest species inventory.
- The field UAV data were collected by Zhaoyu Zhou, Qifei Chen, Yufeng Ma, and Zimu Wang in collaboration.
- Bin Wang, Qifei Chen, and Yufeng Ma, Wenjun Liu created the annotations and the user manual for FLApy and are responsible for building and maintaining the GitHub repository.
- Bin Wang drafted the manuscript, and Bin Wang drew all the graphs in the manuscript.
- Cameron Proctor, Zhiliang Yao, Ninglv Li, Suhui Ma, Chuanbao Jing and Zhiming Zhang provided advice for two cases.
- Zhiming Zhang, Cameron Proctor, Zhiliang Yao, Wenjun Liu, Suhui Ma, Chuanbao Jing, and Luxiang Lin reviewed and commented on the manuscript.
- All authors read and accepted the final version of the manuscript.

# References to third-party packages
Pyvista: https://github.com/pyvista/pyvista
- Sullivan and Kaszynski, (2019). PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK). Journal of Open Source Software, 4(37), 1450, https://doi.org/10.21105/joss.01450

numpy: https://numpy.org/
- Harris, C.R., Millman, K.J., van der Walt, S.J. et al., (2020). Array programming with NumPy. Nature 585, 357–362. https://doi.org/10.1038/s41586-020-2649-2

xarray: https://github.com/pydata/xarray
- Hoyer, S. & Hamman, J., (2017). xarray: N-D labeled Arrays and Datasets in Python. Journal of Open Research Software. 5(1), p.10. DOI: http://doi.org/10.5334/jors.148

open3d: http://www.open3d.org/
- Qian-Yi Zhou and Jaesik Park and Vladlen Koltun, (2018). Open3D: A Modern Library for 3D Data Processing. arXiv e-prints, arXiv:1801.09847.

laspy: https://github.com/laspy/laspy/tree/master

PVGeo: https://pvgeo.org/index.html
- Sullivan et al., (2019). PVGeo: an open-source Python package for geoscientific visualization in VTK and ParaView. Journal of Open Source Software, 4(38), 1451, https://doi.org/10.21105/joss.01451

scipy: https://www.scipy.org/
- Virtanen, P., Gommers, R., Oliphant, T.E. et al., (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nat Meth 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

tqdm: https://github.com/tqdm/tqdm
- Casper da Costa-Luis, (2019). tqdm: A Fast, Extensible Progress Bar for Python and CLI. Journal of Open Source Software, 4(40), 1467, https://doi.org/10.21105/joss.01467

matplotlib: https://matplotlib.org/
- Hunter, J.D., (2007). Matplotlib: A 2D graphics environment. Computing In Science & Engineering, 9(3), pp.90-95. https://doi.org/10.1109/MCSE.2007.55

SALib: https://salib.readthedocs.io/en/latest/
- Herman, J. and Usher, W., (2017). SALib: An open-source Python library for Sensitivity Analysis. Journal of Open Source Software, 2(9), p.97. DOI: http://doi.org/10.21105/joss.00097

scikit-learn: https://scikit-learn.org/stable/
- Pedregosa, F., Varoquaux, G., Gramfort, A. et al., (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, pp.2825-2830.

joblib: https://joblib.readthedocs.io/en/latest/

p_tqdm: https://github.com/swansonk14/p_tqdm

naturalneighbor: https://github.com/innolitics/natural-neighbor-interpolation
- Park, S.W., Linsen, L., Kreylos, O., Owens, J.D. and Hamann, B., (2006). Discrete Sibson interpolation. IEEE Transactions on Visualization and Computer Graphics, 12(2), pp.243-253. https://doi.org/10.1109/TVCG.2006.27

pandas: https://pandas.pydata.org/
- McKinney, W., (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51-56).

miniball: https://github.com/marmakoide/miniball


# Citation
In submission.