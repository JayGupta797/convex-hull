# Convex Hull
Consider a set $S$ of $n$ points fixed in $d$-dimensional space. 
The Convex Hull of this set is the *largest* convex polytope whose vertices are drawn from $S$. 

# Directory Structure
The directory structure is detailed below. Test cases will be added in the near future.

```bash
├── convex-hull
│   ├── qhull
│   │   ├── __init__.py
│   │   ├── qhull.py
│   │   ├── qhull2d.py
│   ├── tests
│   │   ├── __init__.py
├── LICENSE
├── README.md
└── .gitignore
```

# Example Usage

```python3
import numpy as np
from qhull import qhull

points = np.random.rand(100, 2)
qhull(points, tolerance=1e-10, verbose=True)
```

The output for qhull2d.py is a `np.ndarray` of points. 
Each pair of points is an edge in the Convex Hull.

```python3
> qhull2d(points, tolerance=1e-10, verbose=True)
array([[0.07103606, 0.0871293 ],
       [0.0202184 , 0.83261985],
       [0.77815675, 0.87001215],
       [0.56804456, 0.92559664],
       [0.0202184 , 0.83261985],
       [0.56804456, 0.92559664],
       [0.96366276, 0.38344152],
       [0.77815675, 0.87001215],
       [0.96366276, 0.38344152],
       [0.07103606, 0.0871293 ],
       [0.96366276, 0.38344152]])
```

The output for qhull.py is a set of `Facet` objects.
You may directly access `Facet.coordinates` and `Facet.normal` to retrieve the coordinates and normal vector defining the orientation of the facet.

```python3
> facets = qhull(points, tolerance=1e-10, verbose=True)
> facets
{<__main__.Facet at 0x7ff6506205e0>,
 <__main__.Facet at 0x7ff650654d90>,
 <__main__.Facet at 0x7ff6802ed6d0>,
 <__main__.Facet at 0x7ff6802f9e80>,
 <__main__.Facet at 0x7ff690850f10>,
 <__main__.Facet at 0x7ff69085c040>,
 <__main__.Facet at 0x7ff69090d160>,
 <__main__.Facet at 0x7ff690994a30>,
 <__main__.Facet at 0x7ff690994a60>,
 <__main__.Facet at 0x7ff6a257ba60>}

> facets.pop().coordinates
array([[0.0871293 , 0.0202184 , 0.83261985],
       [0.94466892, 0.52184832, 0.41466194],
       [0.11827443, 0.63992102, 0.14335329]])
```

If verbose is set to `True` then each intermediate state of the hull is shown.

2 Dimensional Case         |  3 Dimensional Case
:-------------------------:|:-------------------------:
                           |

# Dependencies
This project uses `NumPy` for numerical computations and `matplotlib` for visualization. 

# Performance

| Dim | Number of Points | qhull Average Time | ConvexHull Average Time |
|-----|------------------|--------------------|-------------------------|
| 2   | 10               | 0.000474           | 0.000350                |
| 2   | 100              | 0.000949           | 0.000384                |
| 2   | 1000             | 0.001692           | 0.000479                |
| 2   | 10000            | 0.003080           | 0.001744                |

| Dim | Number of Points | qhull Average Time | ConvexHull Average Time |
|-----|------------------|--------------------|-------------------------|
| 3   | 10               | 0.002209           | 0.000334                |
| 3   | 100              | 0.020895           | 0.000430                |
| 3   | 1000             | 0.071058           | 0.000902                |
| 3   | 10000            | 0.333961           | 0.003224                |

# Notes
The 2D case accepts inputs in all dimensions but the output is *not* guranteed to be the Convex Hull. 
The 3D case does not perform DFS in CCW order.
The implementation is *not* robust and suffers from numerical precision issues arising from coplanar facets. 
I do not have plans to write proper documentation, but you can visit my [blog post](https://jaygupta797.github.io/posts/convex.html) on this project for theoretic details.

The following resources were invaluable to the development:
  - https://en.wikipedia.org/wiki/Quickhull
  - https://algolist.ru/maths/geom/convhull/qhull3d.php
  - https://media.steampowered.com/apps/valve/2014/DirkGregorius_ImplementingQuickHull.pdf
