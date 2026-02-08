from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType, WRT, CornerType
from math import pow

class SecondOrderCentral(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, cp, field_manager):
        super().__init__(params, mesh, bc, cp, field_manager)

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
            })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            if direction == WRT.x:
                self.__apply_in_x(solver, block_id, i, j)
                
            elif direction == WRT.y:
                self.__apply_in_y(solver, block_id, i, j)

    def _east_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_east(block_id, 1):
                self.__apply_in_x_east(solver, block_id, i, j)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_east(block_id, 1):
                self.__apply_in_y(solver, block_id, i, j)

    def _west_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_west(block_id, 1):
                self.__apply_in_x_west(solver, block_id, i, j)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_west(block_id, 1):
                self.__apply_in_y(solver, block_id, i, j)

    def _north_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_north(block_id, 1):
                self.__apply_in_x(solver, block_id, i, j)

        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_north(block_id, 1):
                self.__apply_in_y_north(solver, block_id, i, j)

    def _south_boundary(self, direction, block_id, solver, var_name):
        if direction == WRT.x:
            for (i, j) in self.mesh.loop_south(block_id, 1):
                self.__apply_in_x(solver, block_id, i, j)
                
        elif direction == WRT.y:
            for (i, j) in self.mesh.loop_south(block_id, 1):
                self.__apply_in_y_south(solver, block_id, i, j)

    def _bottom_left_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.BOTTOM_LEFT]['i']
        j = corners[CornerType.BOTTOM_LEFT]['j']

        if direction == WRT.x:
            self.__apply_in_x_west(solver, block_id, i, j)

        elif direction == WRT.y:
            self.__apply_in_y_south(solver, block_id, i, j)

    def _bottom_right_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.BOTTOM_RIGHT]['i']
        j = corners[CornerType.BOTTOM_RIGHT]['j']

        if direction == WRT.x:
            self.__apply_in_x_east(solver, block_id, i, j)

        elif direction == WRT.y:
            self.__apply_in_y_south(solver, block_id, i, j)

    def _top_left_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.TOP_LEFT]['i']
        j = corners[CornerType.TOP_LEFT]['j']
        
        if direction == WRT.x:
            self.__apply_in_x_west(solver, block_id, i, j)
        elif direction == WRT.y:
            self.__apply_in_y_north(solver, block_id, i, j)

    def _top_right_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.TOP_RIGHT]['i']
        j = corners[CornerType.TOP_RIGHT]['j']

        if direction == WRT.x:
            self.__apply_in_x_east(solver, block_id, i, j)
        elif direction == WRT.y:
            self.__apply_in_y_north(solver, block_id, i, j)
    
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

    def __apply_in_x_east(self, solver, block_id, i, j):
        if self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

            dx, _ = self.mesh.get_spacing(block_id)
            phi_east = self.bc.get_bc_value(block_id, "east")
            rhs = -2.0 * phi_east / dx 

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
            solver.add_to_A(ap_index, aw_index, 2.0 * self.coefficients[block_id]['aw'])

            solver.add_to_b(ap_index, rhs)
        
        elif self.bc.get_bc_type(block_id, "east") == BCType.interface:
            neighbour_block_id = self.cp.get_block_id_at_interface(block_id, "east")
            
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(neighbour_block_id, 1, j)
            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
            solver.add_to_A(ap_index, ae_index, self.coefficients[block_id]['ae'])
            solver.add_to_A(ap_index, aw_index, self.coefficients[block_id]['aw'])
    
    def __apply_in_x_west(self, solver, block_id, i, j):
        if self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

            dx, _ = self.mesh.get_spacing(block_id)
            phi_west = self.bc.get_bc_value(block_id, "west")
            rhs = 2.0 * phi_west / dx 

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
            solver.add_to_A(ap_index, ae_index, 2.0 * self.coefficients[block_id]['ae'])

            solver.add_to_b(ap_index, rhs)

        elif self.bc.get_bc_type(block_id, "west") == BCType.interface:
            neighbour_block_id = self.cp.get_block_id_at_interface(block_id, "west")
            
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
            aw_index = self.mesh.map3Dto1D(neighbour_block_id, self.mesh.num_x[neighbour_block_id] - 2, j)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apx'])
            solver.add_to_A(ap_index, ae_index, self.coefficients[block_id]['ae'])
            solver.add_to_A(ap_index, aw_index, self.coefficients[block_id]['aw'])

    def __apply_in_y_north(self, solver, block_id, i, j):
        if self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            _, dy = self.mesh.get_spacing(block_id)
            phi_north = self.bc.get_bc_value(block_id, "north")
            rhs = -2.0 * phi_north / dy 

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
            solver.add_to_A(ap_index, as_index, 2.0 * self.coefficients[block_id]['as'])

            solver.add_to_b(ap_index, rhs)
        
        elif self.bc.get_bc_type(block_id, "north") == BCType.interface:
            neighbour_block_id = self.cp.get_block_id_at_interface(block_id, "north")
            
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(neighbour_block_id, i, 1)
            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
            solver.add_to_A(ap_index, an_index, self.coefficients[block_id]['an'])
            solver.add_to_A(ap_index, as_index, self.coefficients[block_id]['as'])

    def __apply_in_y_south(self, solver, block_id, i, j):
        if self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

            _, dy = self.mesh.get_spacing(block_id)
            phi_south = self.bc.get_bc_value(block_id, "south")
            rhs = 2.0 * phi_south / dy 

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
            solver.add_to_A(ap_index, an_index, 2.0 * self.coefficients[block_id]['an'])

            solver.add_to_b(ap_index, rhs)
                
        elif self.bc.get_bc_type(block_id, "south") == BCType.interface:
            neighbour_block_id = self.cp.get_block_id_at_interface(block_id, "south")
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
            as_index = self.mesh.map3Dto1D(neighbour_block_id, i, self.mesh.num_y[neighbour_block_id] - 2)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['apy'])
            solver.add_to_A(ap_index, an_index, self.coefficients[block_id]['an'])
            solver.add_to_A(ap_index, as_index, self.coefficients[block_id]['as'])