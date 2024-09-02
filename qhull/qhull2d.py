#!/usr/bin/env python
# coding: utf-8

import logging
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuickHullSettings:
    """Settings for the QuickHull algorithm."""
    def __init__(self, tolerance: float = 1e-10, verbose: bool = True):
        self.tolerance = tolerance
        self.verbose = verbose

def state(points: np.ndarray, hull: Optional[np.ndarray] = None, triangle: Optional[np.ndarray] = None):
    """Visualizes the current state of the convex hull."""
    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], c='black', label='Points')
    
    if hull is not None and len(hull) > 0:
        hull = np.vstack(hull)
        plt.scatter(hull[:, 0], hull[:, 1], c='red', label='Hull')
    
    if triangle is not None:
        triangle = np.vstack([triangle, triangle[0]])
        plt.plot(triangle[:, 0], triangle[:, 1], c='blue', label='Simplex')
        plt.scatter(triangle[:, 0], triangle[:, 1], c='blue')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

def compute_normals(simplex: np.ndarray) -> np.ndarray:
    """Computes outward normal vectors of a simplex."""
    num, dim = simplex.shape
    if (num <= dim): 
        return None
    try:
        M = np.hstack((simplex, np.ones((num, 1))))
        inv = np.linalg.inv(M)
        normals = inv[:-1, :]
        # Notice that the normal opposite to the eye is omitted
        return -normals.T[:-1]
    except np.linalg.LinAlgError:
        return None

def QuickHull(points: np.ndarray, settings: QuickHullSettings) -> np.ndarray:
    """Main function to execute the QuickHull algorithm."""
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
    if dim > 2:
        logging.warning("qhull2d only gurantees correctness for 2D data!")
        if settings.verbose == True:
            logging.warning("State visualization only supported for 2D data!")
            logging.warning("Turning off verbose output.")
            settings.verbose = False
    
    # Sample Extreme Points
    max_points = points[np.argmax(points, axis=0)]
    min_points = points[np.argmin(points, axis=0)]
    candidates = np.unique(np.vstack((min_points, max_points)), axis=0)
    samples = candidates[np.random.choice(candidates.shape[0], min(2, candidates.shape[0]), replace=False)]
    
    # Find Farthest Points from sampled points
    center = np.mean(samples, axis=0)
    normal = np.linalg.svd(samples - center)[2][-1]
    normal /= np.linalg.norm(normal)
    
    projections = (points - center) @ normal
    top = points[np.argmax(projections)]
    bot = points[np.argmin(projections)]
    
    # Build Simplices
    simplex_top = np.vstack((samples, top))
    simplex_bot = np.vstack((samples, bot))
    
    if settings.verbose:
        state(points, None, simplex_top)
        state(points, None, simplex_bot)
    
    # Initiate Recursion
    S1 = FindHull(simplex_top, points[projections >= 0], settings)
    S2 = FindHull(simplex_bot, points[projections < 0], settings)
    hull = np.vstack((S1, S2))
    
    if settings.verbose:
        state(points, [S1, S2], None)
        logging.info("Convex hull construction complete.")
    
    return hull

def FindHull(simplex: np.ndarray, Sk: np.ndarray, settings: QuickHullSettings) -> np.ndarray:
    """Recursively finds the convex hull using the QuickHull algorithm."""
    # Base Case: If the simplex is degenerate, return it
    normals = compute_normals(simplex)
    if normals is None:
        return simplex
    
    # Recursive Case: Otherwise, enumerate each facet and recurse
    normals = compute_normals(simplex)
    hull = []
    for index, normal in enumerate(normals):
        facet = np.delete(simplex, index, axis=0)
        projections = (Sk - np.mean(facet, axis=0)) @ normal
        
        # Build child simplex
        arg = np.argmax(projections)
        if projections[arg] > settings.tolerance:
            farthest = Sk[arg]
            child = np.vstack((facet, farthest))
        else: child = facet
        
        if settings.verbose:
            state(Sk, hull, child)
        
        # Recurse
        S = Sk[projections > -settings.tolerance]
        hull.append(FindHull(child, S, settings))
    
    return np.vstack(hull)
    
def qhull2d(points: np.ndarray, tolerance: float = 1e-10, verbose: bool = True):
    settings = QuickHullSettings(tolerance=tolerance, verbose=verbose)
    return QuickHull(points, settings)
