** TRAJECTORY DATA
* Pre-processed, high-resolution trajectories of homing pigeons flying in a flock obtained
from raw data provided by miniature GPS devices

These data can be used freely (in accordance with Nature’s policy) and unconditionally. We
are ready to provide further details in response to email enquiries.

However, when using these data you are asked to cite our related paper:

Nagy M, Ákos Zs, Biro D, Vicsek T:
Hierarchical group dynamics in pigeon flocks, Nature 464, 890-893 (2010).

Additional information: http://hal.elte.hu/pigeonflocks

* Pre-processing of the data:
The GPS device was capable of logging datapoints (latitude, longitude and altitude
coordinates, and time) with temporal resolution 0.2 s. The geodetic coordinates provided by
the GPS were converted into x, y and z coordinates using the Flat Earth Approximation. The
geodetic coordinates of the origin of the local Cartesian coordinate system and the date of the
measurement are included in the data files. The coordinates of the flight trajectories were
smoothed by a Gaussian filter (sigma = 0.4 s), and the cubic B-spline method was used to fit
curves onto the points obtained with the 0.2 s sampling rate.
Occasionally, the devices failed to log every second or third point and sometimes the devices
lost connection with the satellites for prolonged periods; in such cases we interpolated the
position of the missing datapoints by averaging those recorded immediately before and after.
The last column (“GPS signal”) of the data file contains 1 if the corresponding data point was
measured by the device, or 0 if it was interpolated.

* The data files:
11 free flights (ff) and 4 homing flights (hf) of flocking birds (labelled A to M).

* Additional details concerning pre-processing
We used only those segments of the trajectories for evaluations in which the given bird was in
flight. The criterion for “flying” was that the bird travelled a sufficient distance over a fixed
short period. A bird was considered as having landed if both |x(t – 10s) – x(t)| and
|x(t) – x(t + 10s)| were less than 50 m.

Prior to correlation analysis, all trajectory data were filtered according to two criteria. First, in
order to minimise the effect of missing datapoints, we included only those points in the
analysis for which the interval t – 0.4s to t + 0.4s contained at least two of the five possible
positional fixes as logged by the GPS device (i.e., non-interpolated data). Second, for the
calculation of the C_ij(tau) correlation function, we included only those pairs of datapoints from
birds i and j for which the two birds were not farther than 100 m apart (i.e., d_ij < 100 m)*.

---------
* Erratum: In the original versions of the Supplementary Information the value of this parameter was
given as 10 m. The correct value is 100 m. 