from abc import ABC, abstractmethod
from enum import Enum, auto

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions

class NumericalSchemesBase(ABC):
    def __init__(self, params, mesh, boundary_conditions, corner_points, field_manager):
        self.params = params
        self.mesh = mesh
        self.bc = boundary_conditions
        self.cp = corner_points
        self.field_manager = field_manager
        self.coefficients = []

    def apply(self, direction, solver, time, var_name, multiplier = 1.0):
        # compute latest coefficients
        self.coefficients = []
        self._compute_coefficients(direction, time, var_name, multiplier)

        # get matrix coefficients for all blocks
        for block_id in range(0, self.mesh.num_blocks):
            # apply numerical scheme on interior elements
            self._compute_interior(direction, block_id, solver, var_name)

            # apply numerical scheme on boundary elements
            self._east_boundary(direction, block_id, solver, var_name)
            self._west_boundary(direction, block_id, solver, var_name)
            self._north_boundary(direction, block_id, solver, var_name)
            self._south_boundary(direction, block_id, solver, var_name)

            # apply numerical scheme on corner elements
            self._bottom_left_corner(direction, block_id, solver, var_name)
            self._bottom_right_corner(direction, block_id, solver, var_name)
            self._top_left_corner(direction, block_id, solver, var_name)
            self._top_right_corner(direction, block_id, solver, var_name)

    @abstractmethod
    def _compute_coefficients(self, direction, block_id, time, var_name, multiplier):
        pass
    
    @abstractmethod
    def _compute_interior(self, direction, block_id, solver, var_name):
        pass
    
    @abstractmethod
    def _east_boundary(self, direction, block_id, solver):
        pass

    @abstractmethod
    def _west_boundary(self, direction, block_id, solver):
        pass

    @abstractmethod
    def _north_boundary(self, direction, block_id, solver):
        pass

    @abstractmethod
    def _south_boundary(self, direction, block_id, solver):
        pass

    @abstractmethod
    def _bottom_left_corner(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _bottom_right_corner(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _top_left_corner(self, direction, block_id, solver, var_name):
        pass

    @abstractmethod
    def _top_right_corner(self, direction, block_id, solver, var_name):
        pass