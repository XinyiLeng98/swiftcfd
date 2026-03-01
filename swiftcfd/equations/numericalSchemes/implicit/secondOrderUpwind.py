from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import WRT, BCType
from swiftcfd.enums import PrimitiveVariables as pv

class SecondOrderUpwind(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        pass
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        dx, dy = self.mesh.get_spacing(block_id)
        inv_dx, inv_dy = 1.0 / dx, 1.0 / dy
        inv_2dx, inv_2dy = 1.0 / (2.0 *dx), 1.0 / (2.0 * dy)

        for (i, j) in self.mesh.internal_loop_single_block(block_id):
            
            # at boundaries, apply a first-order scheme
            if i == 1 or i == self.mesh.num_x[block_id] - 2 or j == 1 or j == self.mesh.num_y[block_id] - 2:
                if direction == WRT.x:
                    max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                    min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                    aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)

                    ap_value = (max_u - min_u) * inv_dx
                    ae_value =  min_u * inv_dx
                    aw_value = -1.0 * max_u * inv_dx

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, ae_index, ae_value)
                    solver.add_to_A(ap_index, aw_index, aw_value)

                elif direction == WRT.y:
                    max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                    min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                    as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

                    ap_value = (max_v - min_v) * inv_dy
                    an_value =  min_v * inv_dy
                    as_value = -1.0 * max_v * inv_dy

                    solver.add_to_A(ap_index, ap_index, ap_value)
                    solver.add_to_A(ap_index, an_index, an_value)
                    solver.add_to_A(ap_index, as_index, as_value)
            
            # apply second-order upwind scheme for internal vertices
            else:
                if direction == WRT.x:
                    max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
                    min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
                    aee_index = self.mesh.map3Dto1D(block_id, i + 2, j)
                    aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
                    aww_index = self.mesh.map3Dto1D(block_id, i - 2, j)

                    ap_value = (3.0 * max_u * inv_2dx) - (3.0 * min_u * inv_2dx)
                    ae_value = 4.0 * min_u * inv_2dx
                    aee_value = -1.0 * min_u * inv_2dx
                    aw_value = -4.0 * max_u * inv_2dx
                    aww_value = 1.0 * max_u * inv_2dx

                    solver.add_to_A(ap_index, ap_index,  ap_value )
                    solver.add_to_A(ap_index, ae_index,  ae_value )
                    solver.add_to_A(ap_index, aee_index, aee_value)
                    solver.add_to_A(ap_index, aw_index,  aw_value )
                    solver.add_to_A(ap_index, aww_index, aww_value)

                elif direction == WRT.y:
                    max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
                    min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

                    ap_index = self.mesh.map3Dto1D(block_id, i, j)
                    an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
                    ann_index = self.mesh.map3Dto1D(block_id, i, j + 2)
                    as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
                    ass_index = self.mesh.map3Dto1D(block_id, i, j - 2)

                    ap_value = (3.0 * max_v * inv_2dy) - (3.0 * min_v * inv_2dy)
                    an_value = 4.0 * min_v * inv_2dy
                    ann_value = -1.0 * min_v * inv_2dy
                    as_value = -4.0 * max_v * inv_2dy
                    ass_value = 1.0 * max_v * inv_2dy

                    solver.add_to_A(ap_index, ap_index,  ap_value )
                    solver.add_to_A(ap_index, an_index,  an_value )
                    solver.add_to_A(ap_index, ann_index, ann_value)
                    solver.add_to_A(ap_index, as_index,  as_value )
                    solver.add_to_A(ap_index, ass_index, ass_value)

    def get_right_hand_side_contribution(self, direction, ij, ip1j, im1j, ijp1, ijm1, var_name):
        return 0.0
    
    def _east_boundary(self, direction, block_id, solver):
        pass

    def _west_boundary(self, direction, block_id, solver, var_name):
        pass

    def _north_boundary(self, direction, block_id, solver, var_name):
        pass

    def _south_boundary(self, direction, block_id, solver, var_name):
        pass

    def _bottom_left_corner(self, direction, block_id, solver, var_name):
        pass

    def _bottom_right_corner(self, direction, block_id, solver, var_name):
        pass

    def _top_left_corner(self, direction, block_id, solver, var_name):
        pass

    def _top_right_corner(self, direction, block_id, solver, var_name):
        pass