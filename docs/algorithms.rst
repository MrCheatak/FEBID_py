===========
Algorithms
===========


Hausdorff distance
-------------------

In the `Monte Carlo module <monte_carlo.html>`_ secondary electrons (SE) that drive both dissociation
and beam heating are divided into two groups after emission. SEs that are emitted in the surface vicinity
are considered contributing to the dissociation, while all others are scattered and contribute to the heating effect.
The main criteria for the division is distance to the surface: if the distance is shorter than the SE's
inelastic mean free path (IMFP), it is added to the 'dissociation' group or to the 'heating' otherwise.

.. figure:: _images/hausdorff_distance.png
    :align: right
    :scale: 25 %

    Hausdorff distance matrix [Lee2019]_

In order to determine to which group an SE belongs, it's emission point is superimposed with a distance or
Hausdorff distance matrix.
Each cell in the matrix is assigned a distance to the nearest surface cell based on the cell size (i.e. 2 nm).
In such manner surface vicinity can be evaluated in an effective manner, that requires only calculation of SE's
position in the matrix and comparison to the integer array.

Taking into account that for a given simulation configuration the IMFP and the cell size are fixed,
the array can be converted to a boolean one, where 1 denotes distances less than IMFP and 0 larger than IMFP.
Such simplification reduces memory consumption and computational comparison cost.

The algorithm producing the initial integer matrix is based on a simple operation of adding unity to all cells
that have at least one non-zero neighbor (starting with unity at the surface). Every n-th iteration will populate
a new layer of cells denoting the n-th surface nearest neighbor.

As the surface evolves dynamically, it is necessary to update the Hausdorff distance matrix according to the new
surface profile. Performing aforementioned operations every time a new solid cell is added is computationally
expensive to perform on the whole array. Thus, it can be performed locally. By selecting a section of the matrix
with a newly deposited cell in the center, the Hausdorff distances can be updated locally. The random nature of
cell filling order updates the matrix evenly and keeps it consistent. Not only this approach reduces computational
time by orders of magnitude, but also sustains it at the same level regardless of the grid size.

.. [Lee2019] Lee K.-I., Lee H.-T. et al., Simulation of dynamic growth rate of focused ion beam-induced deposition using Hausdorff distance, Sensors and Actuators A: Physical 2019, 286, 169-177