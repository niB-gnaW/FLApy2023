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
|            Height of max Standard deviation                         |Horizontal|  STDmh_Hor   |
|                   Relative height of max Standard deviation                  |Horizontal|  STDmhr_Hor  |
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
|          Hotspot related circumscribing sphere           | 3D_Cluster |    HCC_3D    |
|          Coldspot related circumscribing sphere          | 3D_Cluster |    CCC_3D    |
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
Zhaoyu Zhou<sup>1</sup>,
Weihong Liu<sup>1</sup>,
Yufeng Ma<sup>1</sup>,
Zimu Wang<sup>1</sup>,
Zhihao Wang<sup>1</sup>,
Luxiang Lin<sup>3, 5</sup>

1. School of Ecology and Environmental Sciences, Yunnan University, Kunming, 650500, China
2. School of the Environment, University of Windsor, Windsor, N9B 3P4, Canada
3. CAS Key Laboratory of Tropical Forest Ecology, Xishuangbanna Tropical Botanical Garden, Chinese Academy of Sciences, Kunming, China
4. University of Chinese Academy of Sciences, Beijing, China
5. National Forest Ecosystem Research Station at Xishuangbanna, Mengla, Yunnan, China

Author mail: wb931022@hotmail.com; zzming76@ynu.edu.cn

# Acknowledgements
This research is supported by the National Natural Science Foundation of China (32260291), The Second Tibetan Plateau Scientific Expedition and Research (STEP) program (2019QZKK0308), and the Joint Fund of the National Natural Science Foundation of China-Yunnan Province (U1902203). In addition, it has received strong support from The Project for Talent and Platform of Science and Technology in Yunnan Province Science and Technology Department (202205AM070005), the Major Program for Basic Research Project of Yunnan Province (202101BC070002) and the Key Research and Development Program of Yunnan Province (No. 202303AC100009). We also thank the Ailao Mountain Nature Reserve Ecological Station for field work support, and the Southeast Asian Biodiversity Institute (151C53KYSB20200019) for laboratory platforms. Besides, we acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC, RGPIN-2022-04861). Finally, we would like to express our sincere thanks to Prof. Jiajia Liu of Fudan University, Prof. Hans De Boeck of the University of Antwerp, and Dr. Suhui Ma and Dr. Chuanbao Jing of Yunnan University for their valuable suggestions and guidance on the writing of this work.

# Authors' contributions
- The FLApy package is developed by Bin Wang
- Cameron Proctor and Zhiming Zhang directed this project.
- Luxing Lin and provided the guidance of the project.

# Citation
In submission.