#!/usr/bin/env python
# coding: utf-8

import logging
from typing import Optional, List, Tuple, Set, Dict

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def state(dim: int, facets: List['Facet'], removed: set['Facet'], 
          horizon: Optional['Horizon'] = None, 
          outside: Optional[np.ndarray] = None, eye: Optional[np.ndarray] = None) -> None:
    """Visualizes the current state of the convex hull"""
    
    if dim not in {2, 3}:
        raise ValueError("The dim must be 2 or 3 for state visualization!")
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d' if dim == 3 else None)

    labels = set()
    def retrieve(label):
        if label not in labels:
            labels.add(label)
            return label
        return ""
    
    for facet in facets:
        if facet not in removed:
            if dim == 2: 
                ax.plot(*facet.coordinates.T, c='blue', label=retrieve('Facet'))
            if dim == 3:
                for edge in facet.subfacets:
                    ax.plot(*edge.coordinates.T, c='blue', label=retrieve('Facet'))
                    
            if facet.normal is not None and facet.center is not None:
                center = facet.center
                normal = facet.normal
                if dim == 2: ax.quiver(*center, *normal, color='magenta')
                else: ax.quiver(*center, *normal, color='magenta', length=0.15)
    
    if horizon is not None:
        if dim == 2:
            data = np.concatenate([point.coordinates for point in horizon.boundary], axis=0)
            data = np.hsplit(data, dim)
            ax.scatter(*data, c='red', s=50)
            ax.plot(*data, c='red', label=retrieve('Horizon'))
        if dim == 3:
            for edge in horizon.boundary:
                data = [edge.coordinates[:,i] for i in range(dim)]
                ax.scatter(*data, c='red')
                ax.plot(*data, c='red', label=retrieve('Horizon'))
    
    if outside is not None:
        ax.scatter(*outside.T, c='black', s=50, label=retrieve('Outside'))
    
    if eye is not None:
        ax.scatter(*eye, c='magenta', s=50, label=retrieve('Eye'))

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if dim == 3: ax.set_zlim([0, 1])
    
    plt.legend()
    plt.show()

class Point:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates

    def __hash__(self) -> int:
        return hash(self.coordinates.tobytes())
    
    def __eq__(self, other: 'Point') -> bool:
        return np.array_equal(self.coordinates, other.coordinates)

class SubFacet:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates
        self.points = frozenset(Point(c) for c in coordinates)

    def __hash__(self) -> int:
        return hash(self.points)
    
    def __eq__(self, other: 'SubFacet') -> bool:
        return self.points == other.points

class Facet:
    def __init__(self, coordinates: np.ndarray, 
                 normal: Optional[np.ndarray] = None, 
                 internal: Optional[np.ndarray] = None):
        self.coordinates = coordinates
        self.center = np.mean(coordinates, axis=0)
        self.normal = self.compute_normal(internal)
        self.subfacets = frozenset(
            SubFacet(np.delete(self.coordinates, i, axis=0)) 
            for i in range(self.coordinates.shape[0])
        )
    
    def compute_normal(self, internal: np.ndarray) -> np.ndarray:
        centered = self.coordinates - self.center
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1, :]
        normal /= np.linalg.norm(normal)

        if np.dot(normal, self.center - internal) < 0:
            normal *= -1

        return normal

    def __hash__(self) -> int:
        return hash(self.subfacets)
    
    def __eq__(self, other: 'Facet') -> bool:
        return self.subfacets == other.subfacets

class Horizon:
    def __init__(self):
        self.facets: Set[Facet] = set()
        self.boundary: List[SubFacet] = []

class QuickHull:
    def __init__(self, tolerance: float = 1e-10, verbose: bool = True):
        self.facets: List[Facet] = []
        self.removed: Set[Facet] = set()
        self.outside: Dict[Facet, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
        self.neighbors: Dict[SubFacet, Set[Facet]] = {}
        self.unclaimed: Optional[np.ndarray] = None
        self.internal: Optional[np.ndarray] = None
        self.tolerance = tolerance
        self.verbose = verbose
    
    def initialize(self, points: np.ndarray) -> None:
        # Sample Points
        simplex = points[np.random.choice(points.shape[0], points.shape[1] + 1, replace=False)]
        self.unclaimed = points
        self.internal = np.mean(simplex, axis=0)
        
        # Build Simplex
        for c in range(simplex.shape[0]):
            facet = Facet(np.delete(simplex, c, axis=0), internal=self.internal)
            self.classify(facet)
            self.facets.append(facet)
        
        # Attach Neighbors
        for f in self.facets:
            for sf in f.subfacets:
                self.neighbors.setdefault(sf, set()).add(f)
    
    def classify(self, facet: Facet) -> None:
        if not self.unclaimed.size:
            self.outside[facet] = (None, None)
            return
        
        # Compute Projections
        projections = (self.unclaimed - facet.center) @ facet.normal
        arg = np.argmax(projections)
        mask = projections > self.tolerance
        
        # Identify Eye and Outside Set
        eye = self.unclaimed[arg] if projections[arg] > self.tolerance else None
        outside = self.unclaimed[mask]
        self.outside[facet] = (outside, eye)
        self.unclaimed = self.unclaimed[~mask]
    
    def compute_horizon(self, eye: np.ndarray, start_facet: Facet) -> Horizon:
        horizon = Horizon()
        self._recursive_horizon(eye, start_facet, horizon)
        return horizon
    
    def _recursive_horizon(self, eye: np.ndarray, facet: Facet, horizon: Horizon) -> int:
        # If the eye is visible from the facet...
        if np.dot(facet.normal, eye - facet.center) > 0:
            # Label the facet as visible and cross each edge
            horizon.facets.add(facet)
            for subfacet in facet.subfacets:
                neighbor = (self.neighbors[subfacet] - {facet}).pop()
                # If the neighbor is not visible, then the edge shared must be on the boundary
                if neighbor and neighbor not in horizon.facets:
                    if not self._recursive_horizon(eye, neighbor, horizon):
                        horizon.boundary.append(subfacet)
            return 1
        return 0
    
    def build(self, points: np.ndarray) -> Set[Facet]:
        num, dim = points.shape
        if num < dim + 1:
            logging.warning("Not enough points to construct a convex hull!")
            return
        if dim == 0:
            logging.warning("Empty NumPy array received!")
            return
        if dim == 1:
            logging.info("The Convex hull of 1D data is its min-max.")
            return np.array([np.min(points), np.max(points)])
        if dim > 3 and self.verbose == True:
            logging.warning("State visualization only supported for 3d data!")
            logging.warning("Turning off verbose output.")
            self.verbose = False
          
        self.initialize(points)
        while True:
            updated = False
            for facet in self.facets:
                if facet in self.removed: 
                    continue
                outside, eye = self.outside[facet]
                if eye is not None:
                    updated = True
                    horizon = self.compute_horizon(eye, facet)
                    if self.verbose:
                        state(dim, self.facets, self.removed, horizon, outside, eye)
                    for f in horizon.facets:
                        self.unclaimed = np.vstack((self.unclaimed, self.outside[f][0]))
                        self.removed.add(f)
                        for sf in f.subfacets:
                            self.neighbors[sf].discard(f)
                            if self.neighbors[sf] == set(): del self.neighbors[sf]
                    for sf in horizon.boundary:
                        nf = Facet(np.vstack((sf.coordinates, eye)), internal=self.internal)
                        self.classify(nf)
                        self.facets.append(nf)
                        for nsf in nf.subfacets:
                            self.neighbors.setdefault(nsf, set()).add(nf)
            if not updated:
                break
        
        if self.verbose:
            state(dim, self.facets, self.removed, None, None)
            logging.info("Convex hull construction complete.")
            
        return set(self.facets) - self.removed

def qhull(points: np.ndarray, tolerance: float = 1e-10, verbose: bool = True):
    instance = QuickHull(tolerance=tolerance, verbose=verbose)
    return instance.build(points)