# proto-planet mass estimator
## Introduction

The aim of this project is to estimate the mass of potential newly-born planet inside protoplanetary disk. 

### The codes include essential steps to estimate the mass:

1. **Display the disk image** from ALMA data, and **plot contours** from VLA data to show relatively high energy emission regions.
2. **Derive bolometric luminosity** from flux measured with CASA, and use Monte Carlo method to **introduce errors**. 
3. **Interpolate evolutionary tracks** and **estimate mass** of the potential sources.
4. Output estimated mass distribution as violin plots and latex tables. 



### The result pictures include:

1. Planetary disk pictures: show the overall pictures of all target disks.
2. Disk pictures with contours: display regions with high free-free emission.
3. Evolutionary track pictures: display the interpolation of evolutionary tracks; demonstrate the process to derive mass from the interpolation of evolutionary tracks.
4. Violin plot pictures: show the distribution of estimated mass. 

## Requirement

Required python packages include:

`astropy, numpy, scipy, matplotlib,shapely`

## Citation

### target planetary disks

### Evolutionary track interpolation

