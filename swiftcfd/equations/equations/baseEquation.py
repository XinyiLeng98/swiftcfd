from abc import ABC, abstractmethod

from petsc4py import PETSc

from swiftcfd.equations.boundaryConditions.boundaryConditions import BoundaryConditions
from swiftcfd.equations.boundaryConditions.cornerPoint import CornerPoint
from swiftcfd.equations.linearAlgebraSolver.linearAlgebraSolver import LinearAlgebraSolver
from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.enums import BCType, CornerType

class BaseEquation(ABC):
    def __init__(self, params, mesh, field_manager):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager

        self.bc = BoundaryConditions(self.params, self.mesh, self.get_variable_name())
        self.cp = CornerPoint(self.mesh, self.bc)

        self.has_first_order_time_derivative = False
        self.has_first_order_space_derivative = False
        self.has_second_order_space_derivative = False
        self.has_source = False

        # specify whether this equation should be linearised (picard iterations will be applied)
        self.requires_linearisation = False

        self.solver = LinearAlgebraSolver(self.params, self.mesh, self.get_variable_name(), self.bc.is_fully_neumann())

        self.field_manager.add_field(self.get_variable_name())

    def pre_solve_task(self, runtime):
        """Anything that needs to be done before the solver is called."""
        pass

    def post_solve_task(self, runtime):
        """Anything that needs to be done after the solver is called."""
        pass

    def solve(self, runtime):
        # ensure matrix and right-hand side vector are zeroed
        self.solver.reset_A()
        self.solver.reset_b()
        
        # set internal matrix coefficients
        if self.has_first_order_time_derivative:
            self.first_order_time_derivative(runtime)
        if self.has_first_order_space_derivative:
            self.first_order_space_derivative(runtime) 
        if self.has_second_order_space_derivative:
            self.second_order_space_derivative(runtime)
        if self.has_source:
            self.source(runtime)

        # apply dirichlet boundary conditions exactly once here
        self.apply_dirichlet_boundary_conditions()

        # # set corner points
        # self.cp.apply_corner_points(self.solver, self.field_manager.fields[self.get_variable_name()])

        # assemble matrix
        self.solver.assemble()
        # if self.get_variable_name() == 'p':
        #     self.solver.view()
        #     exit(0)

        # solver linear system of equations after coefficient matrix has been assembled
        self.solver.solve(self.field_manager.fields[self.get_variable_name()])

        # # adjust corner points
        # self.cp.average_field_at_corner_point(self.field_manager.fields[self.get_variable_name()])

        # apply under-relaxation
        self.under_relaxation()

    def under_relaxation(self):
        alpha = self.params('solver', 'linearSolver', 'underRelaxation', self.get_variable_name())

        for (block, i, j) in self.mesh.internal_loop_all_blocks():
            self.field_manager.fields[self.get_variable_name()][block, i, j] = \
                alpha * self.field_manager.fields[self.get_variable_name()][block, i, j] + \
                (1.0 - alpha) * self.field_manager.fields[self.get_variable_name()].picard_old[block, i, j]
    
    def apply_dirichlet_boundary_conditions(self):
        # apply dirichlet boundary conditions at edges
        faces = ["east", "west", "north", "south"]
        face_loops = [self.mesh.loop_east, self.mesh.loop_west, self.mesh.loop_north, self.mesh.loop_south]
        for block_id in range(0, self.mesh.num_blocks):
            for i in range(0, len(faces)):
                if self.bc.get_bc_type(block_id, faces[i]) == BCType.dirichlet:
                    bc_value = self.bc.get_bc_value(block_id, faces[i])
                    for (i, j) in face_loops[i](block_id, 1):
                        ap_index = self.mesh.map3Dto1D(block_id, i, j)
                        self.solver.add_to_A(ap_index, ap_index, 1.0)
                        self.solver.add_to_b(ap_index, bc_value)
        
        # apply dirichlet boundary conditions at corners
        for block_id in range(0, self.mesh.num_blocks):
            corners = self.cp.get_corners(block_id)
            for corner_location, corner in corners.items():
                if corner['type'] == BCType.dirichlet:
                    bc_value = corner['value']
                    ap_index = self.mesh.map3Dto1D(block_id, corner['i'], corner['j'])
                    self.solver.add_to_A(ap_index, ap_index, 1.0)
                    self.solver.add_to_b(ap_index, bc_value)

    def first_order_time_derivative(self, runtime):
        """Handle time derivatives of the equation."""
        pass

    def first_order_space_derivative(self, runtime):
        """Handle advection (first-order spatial derivatives)."""
        pass

    def second_order_space_derivative(self, runtime):
        """Handle diffusion (second-order spatial derivatives)."""
        pass

    def source(self, runtime):
        """Handle sources of the equation."""
        pass

    @abstractmethod
    def get_diffusion_coefficients(self):
        pass

    @abstractmethod
    def get_variable_name(self):
        pass