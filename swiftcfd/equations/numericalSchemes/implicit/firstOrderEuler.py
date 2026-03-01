from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import BCType, CornerType

class FirstOrderEuler(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, field_manager):
        super().__init__(params, mesh, bc, field_manager)

    def _compute_coefficients(self, direction, runtime, var_name, multiplier):
        dt = runtime.dt
        for block_id in range(0, self.mesh.num_blocks):
            self.coefficients.append({
                'ap': 1.0/dt * multiplier,
                'b':  1.0/dt * multiplier
            })
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.loop_cells(block_id):
            ap_index = self.mesh.map3Dto1D(block_id, i, j)
            b_index  = self.mesh.map3Dto1D(block_id, i, j)
            b_value  = self.coefficients[block_id]['b'] * self.field_manager.fields[var_name].old[block_id, i, j]

            solver.add_to_A(ap_index, ap_index, self.coefficients[block_id]['ap'])
            solver.add_to_b(b_index, b_value)
    
    def _east_boundary(self, direction, block_id, solver, var_name):
        pass

    def _west_boundary(self, direction, block_id, solver, var_name):
        pass

    def _north_boundary(self, direction, block_id, solver, var_name):
        pass

    def _south_boundary(self, direction, block_id, solver, var_name):
        pass
