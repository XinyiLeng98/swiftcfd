from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType, CornerType

class FirstOrderEuler(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, cp, field_manager):
        super().__init__(params, mesh, bc, cp, field_manager)

    def _compute_coefficients(self, direction, runtime, var_name, multiplier):
        dt = runtime.dt
        for block_id in range(0, self.mesh.num_blocks):
            self.coefficients.append({
                'ap': 1.0/dt * multiplier,
                'b':  1.0/dt * multiplier
            })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            b_index  = self.mesh.map3Dto1D(block_id, i, j)

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
            solver.add_to_b(b_index, self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j])
    
    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        block1, i1, j1 = ij
        multiplier = self.coefficients[block1]['b']
        return multiplier * self.field_manager.fields[var_name].old[block1, i1, j1]

    def _east_boundary(self, direction, block_id, solver, var_name):
        self.__update_boundary(block_id, solver, var_name, "east", self.mesh.loop_east)

    def _west_boundary(self, direction, block_id, solver, var_name):
        self.__update_boundary(block_id, solver, var_name, "west", self.mesh.loop_west)

    def _north_boundary(self, direction, block_id, solver, var_name):
        self.__update_boundary(block_id, solver, var_name, "north", self.mesh.loop_north)

    def _south_boundary(self, direction, block_id, solver, var_name):
        self.__update_boundary(block_id, solver, var_name, "south", self.mesh.loop_south)

    def __update_boundary(self, block_id, solver, var_name, face, bc_loop):
        bc_type = self.bc.get_bc_type(block_id, face)
        if bc_type == BCType.neumann or bc_type == BCType.interface:
            for (i, j) in bc_loop(block_id, 1):
                bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
                ap_index = self.mesh.map3Dto1D(block_id, i, j)
                solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
                solver.add_to_b(ap_index, bc_value)
    
    def _bottom_left_corner(self, direction, block_id, solver, var_name):
        self.__update_corner(block_id, solver, var_name, CornerType.BOTTOM_LEFT)

    def _bottom_right_corner(self, direction, block_id, solver, var_name):
        self.__update_corner(block_id, solver, var_name, CornerType.BOTTOM_RIGHT)

    def _top_left_corner(self, direction, block_id, solver, var_name):
        self.__update_corner(block_id, solver, var_name, CornerType.TOP_LEFT)

    def _top_right_corner(self, direction, block_id, solver, var_name):
        self.__update_corner(block_id, solver, var_name, CornerType.TOP_RIGHT)

    def __update_corner(self, block_id, solver, var_name, corner_id):
        corners = self.cp.get_corners(block_id)
        bc_type = corners[corner_id]['type']

        if bc_type == BCType.neumann or bc_type == BCType.interface:
            i = corners[corner_id]['i']
            j = corners[corner_id]['j']

            bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            
            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
            solver.add_to_b(ap_index, bc_value)

    # def __update_corner(self, block_id, i, j, solver, var_name):
    #     bc_type = self.bc.get_bc_type(block_id, face)
    #     if bc_type == BCType.neumann or bc_type == BCType.interface:
    #         bc_value = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]
    #         ap_index = self.mesh.map3Dto1D(block_id, i, j)
    #         solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
    #         solver.add_to_b(ap_index, bc_value)