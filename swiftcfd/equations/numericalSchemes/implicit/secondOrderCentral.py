from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType, WRT, CornerType
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        for block_id in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.coefficients.append({
                'apx': multiplier * (-2.0/pow(dx, 2)),
                'apy': multiplier * (-2.0/pow(dy, 2)),
                'ae': multiplier * ( 1.0/pow(dx, 2)),
                'aw': multiplier * ( 1.0/pow(dx, 2)),
                'an': multiplier * ( 1.0/pow(dy, 2)),
                'as': multiplier * ( 1.0/pow(dy, 2)),
                'b': multiplier
            })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.loop_internal_cells(block_id):
            if direction == WRT.x:
                self.__apply_in_x(solver, block_id, i, j)
                
            elif direction == WRT.y:
                self.__apply_in_y(solver, block_id, i, j)

    def _east_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_east_bc(block_id):
                self.__apply_east_boundary(block_id, i, j, solver)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_east_bc(block_id, 1):
                self.__apply_in_y(solver, block_id, i, j)
            
            # corner points
            self.__apply_north_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, self.mesh.num_cells_y[block_id] - 1, solver)        
            self.__apply_south_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, 0, solver)        

    def _west_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_west_bc(block_id):
                self.__apply_west_boundary(block_id, i, j, solver)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_west_bc(block_id, 1):
                self.__apply_in_y(solver, block_id, i, j)
            
            # corner points
            self.__apply_north_boundary(block_id, 0, self.mesh.num_cells_y[block_id] - 1, solver)        
            self.__apply_south_boundary(block_id, 0, 0, solver) 
    
    def _north_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_north_bc(block_id, 1):
                self.__apply_in_x(solver, block_id, i, j)
            
            # corner points
            self.__apply_east_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, self.mesh.num_cells_y[block_id] - 1, solver)        
            self.__apply_west_boundary(block_id, 0, self.mesh.num_cells_y[block_id] - 1, solver) 

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_north_bc(block_id):
                self.__apply_north_boundary(block_id, i, j, solver)          

    def _south_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_south_bc(block_id, 1):
                self.__apply_in_x(solver, block_id, i, j)
            
            # corner points
            self.__apply_east_boundary(block_id, self.mesh.num_cells_x[block_id] - 1, 0, solver)
            self.__apply_west_boundary(block_id, 0, 0, solver) 

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_south_bc(block_id):
                self.__apply_south_boundary(block_id, i, j, solver)

    def __apply_east_boundary(self, block_id, i, j, solver):       
        if self.bc.get_bc_type(block_id, "east") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

            ap_value = 1.5 * self.coefficients[block_id]['apx']
            aw_value = self.coefficients[block_id]['aw']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "east")
            rhs = -b_value * 2.0 * phi / pow(dx, 2)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, aw_index, aw_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

            ap_value = 0.5 * self.coefficients[block_id]['apx']
            aw_value = self.coefficients[block_id]['aw']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "east")
            rhs = b_value * phi / (2.0 * dx)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, aw_index, aw_value)
            solver.add_to_b(ap_index, rhs)
        
        elif self.bc.get_bc_type(block_id, "east") == BCType.interface:
            neighbour_block_id = self.bc.get_bc_value(block_id, "east")
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
            ae_index = self.mesh.map3Dto1D(neighbour_block_id, 0, j)

            ap_value = self.coefficients[block_id]['apx']
            aw_value = self.coefficients[block_id]['aw']
            ae_value = self.coefficients[neighbour_block_id]['ae']

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)
            solver.add_to_A(ap_index, aw_index, aw_value)

    def __apply_west_boundary(self, block_id, i, j, solver):
        if self.bc.get_bc_type(block_id, "west") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

            ap_value = 1.5 * self.coefficients[block_id]['apx']
            ae_value = self.coefficients[block_id]['ae']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "west")
            rhs = -b_value * 2.0 * phi / pow(dx, 2)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

            ap_value = 0.5 * self.coefficients[block_id]['apx']
            ae_value = self.coefficients[block_id]['ae']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "west")
            rhs = -b_value * phi / (2.0 * dx)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "west") == BCType.interface:
            neighbour_block_id = self.bc.get_bc_value(block_id, "west")
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
            aw_index = self.mesh.map3Dto1D(neighbour_block_id, self.mesh.num_cells_x[neighbour_block_id] - 1, j)

            ap_value = self.coefficients[block_id]['apx']
            ae_value = self.coefficients[block_id]['ae']
            aw_value = self.coefficients[neighbour_block_id]['aw']

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, ae_index, ae_value)
            solver.add_to_A(ap_index, aw_index, aw_value)
    
    def __apply_north_boundary(self, block_id, i, j, solver):
        if self.bc.get_bc_type(block_id, "north") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            ap_value = 1.5 * self.coefficients[block_id]['apy']
            as_value = self.coefficients[block_id]['as']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "north")
            rhs = -b_value * 2.0 * phi / pow(dy, 2)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, as_index, as_value)
            solver.add_to_b(ap_index, rhs)
            
        elif self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            ap_value = 0.5 * self.coefficients[block_id]['apy']
            as_value = self.coefficients[block_id]['as']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "north")
            rhs = b_value * phi / (2.0 * dy)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, as_index, as_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "north") == BCType.interface:
            neighbour_block_id = self.bc.get_bc_value(block_id, "north")
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
            an_index = self.mesh.map3Dto1D(neighbour_block_id, i, 0)

            ap_value = self.coefficients[block_id]['apy']
            as_value = self.coefficients[block_id]['as']
            an_value = self.coefficients[neighbour_block_id]['an']

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
            solver.add_to_A(ap_index, as_index, as_value)
    
    def __apply_south_boundary(self, block_id, i, j, solver):
        if self.bc.get_bc_type(block_id, "south") == BCType.dirichlet:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

            ap_value = 1.5 * self.coefficients[block_id]['apy']
            an_value = self.coefficients[block_id]['an']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "south")
            rhs = -b_value * 2.0 * phi / pow(dy, 2)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

            ap_value = 0.5 * self.coefficients[block_id]['apy']
            an_value = self.coefficients[block_id]['an']
            b_value = self.coefficients[block_id]['b']

            dx, dy = self.mesh.get_spacing(block_id)
            phi = self.bc.get_bc_value(block_id, "south")
            rhs = -b_value * phi / (2.0 * dy)

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "south") == BCType.interface:
            neighbour_block_id = self.bc.get_bc_value(block_id, "south")
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
            as_index = self.mesh.map3Dto1D(neighbour_block_id, i, self.mesh.num_cells_y[neighbour_block_id] - 1)

            ap_value = self.coefficients[block_id]['apy']
            an_value = self.coefficients[block_id]['an']
            as_value = self.coefficients[neighbour_block_id]['as']

            solver.add_to_A(ap_index, ap_index, ap_value)
            solver.add_to_A(ap_index, an_index, an_value)
            solver.add_to_A(ap_index, as_index, as_value)
    
    def __apply_in_x(self, solver, block_id, i, j):
        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
        aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

        solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
        solver.add_to_A(ap_index, ae_index, self.coefficients[block_id]['ae'])
        solver.add_to_A(ap_index, aw_index, self.coefficients[block_id]['aw'])

    def __apply_in_y(self, solver, block_id, i, j):
        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
        as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

        solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
        solver.add_to_A(ap_index, an_index, self.coefficients[block_id]['an'])
        solver.add_to_A(ap_index, as_index, self.coefficients[block_id]['as'])
