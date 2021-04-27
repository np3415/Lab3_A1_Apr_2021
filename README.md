# Lab3_A1_Apr_2021
Galaxy counting algorithm

The script containing the galaxy counting algorithm is in the  file.

--------------------
| NO RADIAL FILTER |
--------------------
The following data files were obtained for no radial filter, up to 3.5 standard deviations above the mean beackground of the image:

Binned Data (3.5sNRF_corrected_filter).npy:
  [Magnitudes, Number_of_Galaxies, Number_Error]

Galaxy Data (3.5sNRF_corrected_filter).npy:
  [Galaxy_Magnitudes, Magnitude_Errors, Radii]:

Raw Galaxy Data (3.5sNRF_corrected_filter).npy:
  [Galaxy_Counts, Galaxy_Errors, Radii]
  
Final Filter(3.5sNRF_corrected_filter).fits:
  Viewable on SAOImage ds9
  
Final Image (3.5sNRF_corrected_filter).fits:
  Viewable on SAOImage ds9

-----------------
| RADIAL FILTER |
-----------------
The following data files were obtained for a minimum radial filter of 1.9, up to 3.5 standard deviations above the mean background of the image:

Binned Data (3.5sRF_corrected_filter).npy:
  [Magnitudes, Number_of_Galaxies, Number_Error]

Galaxy Data (3.5sRF_corrected_filter).npy:
  [Galaxy_Magnitudes, Magnitude_Errors, Radii]

Raw Galaxy Data (3.5sRF_corrected_filter).npy:
  [Galaxy_Counts, Galaxy_Errors, Radii]
  
Final Filter(3.5sRF_corrected_filter).fits:
  Viewable on SAOImage ds9
  
Final Image (3.5sRF_corrected_filter).fits:
  Viewable on SAOImage ds9

---------
| OTHER |
---------
Final Galaxy Counting Algorithm.py:
  Python script containing the algorithm

A1_Mosaic.fits:
  The original image (viewable on SAOImage ds9)
  
Correct Filter Shape.fits:
  Shape of the initial filter (viewable on SAOImage ds9)
  

  
