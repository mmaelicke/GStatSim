<img src ="GStat-Sim/Images/GatorGlaciologyLogo-01.jpg" width="100" align = "right">

# GStat-Sim
GStat-Sim is a collection of functions and demos specifically designed for geostatistical interpolation. It is inspired by open source geostatistical resources such as GeostatsPy, Geostatistics Lessons, and SciKit-GStat. In my own research, I have found that geostatistical tools designed for industry applications do not have the flexibility to address the unique combination of challenges in ice sheet problems, including large crossover errors, spatially variable measurement uncertainty, extremely large datasets, non-linear trends, variability in measurement density, and non-stationarity. These tools are part of our ongoing effort to develop and adapt open-access geostatistical functions.

In its current state, the demos focus on the geostatistical simulation of subglacial topography. However, these protocols could be applied to a number topics in glaciology, or geoscientific problems in general.

We will continuously develop new tools and tutorials to address specific technical challenges in geostatistics. Do you have feedback or suggestions? Specific things that we should account for? Feel free to contact me at emackie@ufl.edu. Our goal is to create tools that are useful and accessible, so we welcome your thoughts and insight.

<img src ="GStat-Sim/Images/GStat-sim_master_figure.png" align = "center">

# Features

## Functions
Some of the tools in GStat-Sim:

* **skrige** - Simple kriging
* **okrige** - Ordinary kriging
* **skrige_sgs** - Sequential Gaussian simulation using simple kriging
* **okrige_sgs** - Sequential Gaussian simulation using ordinary kriging
* **cluster_sgs** - Sequential Gaussian simulation where different variograms are used in different areas
* **cokrige_mm1** - Cokriging (kriging with a secondary constraint) under Markov assumptions 
* **cosim_mm1** - Cosimulation under Markov assumptions

## Demos
We have created tutorials that are designed to provide an intuitive understanding of geostatistical methods and to demonstrate how these methods are used. The current demos are:

* **1_Experimental_Variogram.ipynb** - Demonstration of experimental variogram calculation to quantify spatial relationships.
* **2_Variogram_model.ipynb** - A tutorial on fitting a variogram model to an experimental variogram.
* **3_Simple_kriging_and_ordinary_kriging.ipynb** - Demonstration of simple kriging and ordinary kriging interpolation.
* **4_Sequential_Gaussian_Simulation.ipynb** - An introduction to stochastic simulation.
* **5_interpolation_with_anisotropy.ipynb** - A demonstration of kriging and SGS with anisotropy.
* **6_non-stationary_SGS_example1.ipynb** - A tutorial on SGS with multiple variograms. This demo uses k-means clustering to divide the conditioning data
* **7_non-stationary_SGS_example2.ipynb** - SGS using multiple variograms where the clusters are determined automatically.
* **8_interpolation_with_a_trend.ipynb** - Kriging and SGS in the presence of a large-scale trend
* **9_cokriging_and_cosimulation_MM1.ipynb** - Kriging and SGS using secondary constraints
into groups that are each assigned their own variogram.


# The author
(Emma) Mickey MacKie is an assistant professor at the University of Florida.

# Useage
The functions are in the GStatSim.py document. Just download the GStat-Sim folder and make sure the GStatSim.py script is in your working directory. The datasets for the demos are in the Data folder.

## Package dependencies
* Numpy
* Pandas
* Scipy
* tqdm
* Sklearn

## Requirements for visualization and variogram analysis
* Matplotlib
* earthpy
* SciKit-GStat

These can all be installed using the command *pip install (package name)*

# Datasets

The demos use radar bed measurements from the Center for the Remote Sensing of Ice Sheets (CReSIS, 2020).

CReSIS. 2020. Radar depth sounder, Lawrence, Kansas, USA. Digital Media. http://data.cresis.ku.edu/.
