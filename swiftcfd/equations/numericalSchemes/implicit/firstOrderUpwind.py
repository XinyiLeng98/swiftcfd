from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
from swiftcfd.enums import WRT, BCType, CornerType
from swiftcfd.enums import PrimitiveVariables as pv

class FirstOrderUpwind(NumericalSchemesBase):
    def __init__(self, params, mesh, bc, cp, field_manager):
        super().__init__(params, mesh, bc, cp, field_manager)

        self.inv_dx = []
        self.inv_dy = []
        for block_id in range(self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block_id)
            self.inv_dx.append(1.0 / dx)
            self.inv_dy.append(1.0 / dy)

    def _compute_coefficients(self, direction, time, var_name, multiplier):
        pass
    
    def _compute_interior(self, direction, block_id, solver, var_name):
        for (i, j) in self.mesh.internal_loop_single_block(block_id):            
            if direction == WRT.x:
                self.upwind_wrt_x(block_id, i, j, solver)

            elif direction == WRT.y:
                self.upwind_wrt_y(block_id, i, j, solver)

    def _east_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "east") == BCType.neumann:
            for (i, j) in self.mesh.loop_east(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x_neumann(block_id, i, j, solver, "east")

                elif direction == WRT.y:
                    self.upwind_wrt_y(block_id, i, j, solver)
        elif self.bc.get_bc_type(block_id, "east") == BCType.interface:
            for (i, j) in self.mesh.loop_east(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x_interface(block_id, i, j, solver, "east")

                elif direction == WRT.y:
                    self.upwind_wrt_y(block_id, i, j, solver)

    def _west_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "west") == BCType.neumann:
            for (i, j) in self.mesh.loop_west(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x_neumann(block_id, i, j, solver, "west")

                elif direction == WRT.y:
                    self.upwind_wrt_y(block_id, i, j, solver)
        elif self.bc.get_bc_type(block_id, "west") == BCType.interface:
            for (i, j) in self.mesh.loop_west(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x_interface(block_id, i, j, solver, "west")

                elif direction == WRT.y:
                    self.upwind_wrt_y(block_id, i, j, solver)

    def _north_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "north") == BCType.neumann:
            for (i, j) in self.mesh.loop_north(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x(block_id, i, j, solver)

                elif direction == WRT.y:
                    self.upwind_wrt_y_neumann(block_id, i, j, solver, "north")
        elif self.bc.get_bc_type(block_id, "north") == BCType.interface:
            for (i, j) in self.mesh.loop_north(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x(block_id, i, j, solver)

                elif direction == WRT.y:
                    self.upwind_wrt_y_interface(block_id, i, j, solver, "north")

    def _south_boundary(self, direction, block_id, solver, var_name):
        if self.bc.get_bc_type(block_id, "south") == BCType.neumann:
            for (i, j) in self.mesh.loop_south(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x(block_id, i, j, solver)

                elif direction == WRT.y:
                    self.upwind_wrt_y_neumann(block_id, i, j, solver, "south")
        elif self.bc.get_bc_type(block_id, "south") == BCType.interface:
            for (i, j) in self.mesh.loop_south(block_id, 1):
                if direction == WRT.x:
                    self.upwind_wrt_x(block_id, i, j, solver)

                elif direction == WRT.y:
                    self.upwind_wrt_y_interface(block_id, i, j, solver, "south")
    
    def _bottom_left_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.BOTTOM_LEFT]['i']
        j = corners[CornerType.BOTTOM_LEFT]['j']
        self.negative_in_x(block_id, i, j, solver)
        self.negative_in_y(block_id, i, j, solver)

    def _bottom_right_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.BOTTOM_RIGHT]['i']
        j = corners[CornerType.BOTTOM_RIGHT]['j']
        self.positive_in_x(block_id, i, j, solver)
        self.negative_in_y(block_id, i, j, solver)

    def _top_left_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.TOP_LEFT]['i']
        j = corners[CornerType.TOP_LEFT]['j']
        self.negative_in_x(block_id, i, j, solver)
        self.positive_in_y(block_id, i, j, solver)

    def _top_right_corner(self, direction, block_id, solver, var_name):
        corners = self.cp.get_corners(block_id)
        i = corners[CornerType.TOP_RIGHT]['i']
        j = corners[CornerType.TOP_RIGHT]['j']
        self.positive_in_x(block_id, i, j, solver)
        self.positive_in_y(block_id, i, j, solver)
    
    def upwind_wrt_x(self, block_id, i, j, solver):
        self.positive_in_x(block_id, i, j, solver)
        self.negative_in_x(block_id, i, j, solver)

    def upwind_wrt_y(self, block_id, i, j, solver):
        self.positive_in_y(block_id, i, j, solver)
        self.negative_in_y(block_id, i, j, solver)

    def upwind_wrt_x_neumann(self, block_id, i, j, solver, face):
        rhs = self.bc.get_bc_value(block_id, face)
        if face == "east":
            self.positive_in_x(block_id, i, j, solver)
        elif face == "west":
            self.negative_in_x(block_id, i, j, solver)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        solver.add_to_b(ap_index, rhs)

    def upwind_wrt_y_neumann(self, block_id, i, j, solver, face):
        rhs = self.bc.get_bc_value(block_id, face)
        if face == "north":
            self.positive_in_y(block_id, i, j, solver)
        elif face == "south":
            self.negative_in_y(block_id, i, j, solver)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        solver.add_to_b(ap_index, rhs)

    def upwind_wrt_x_interface(self, block_id, i, j, solver, face):
        max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)
        min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        ap_value = (max_u - min_u) * self.inv_dx[block_id]

        neighbour_block_id = self.bc.get_bc_value(block_id, face)
        if face == "east":
            ae_index = self.mesh.map3Dto1D(neighbour_block_id, 1, j)
            ae_value =  min_u * self.inv_dx[neighbour_block_id]

            aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
            aw_value = -1.0 * max_u * self.inv_dx[block_id]
        elif face == "west":
            ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)
            ae_value =  min_u * self.inv_dx[block_id]

            aw_index = self.mesh.map3Dto1D(neighbour_block_id, self.mesh.num_x[neighbour_block_id] - 2, j)
            aw_value = -1.0 * max_u * self.inv_dx[neighbour_block_id]
        else:
            raise Exception("Unknown face")
           
        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, ae_index, ae_value)
        solver.add_to_A(ap_index, aw_index, aw_value)

    def upwind_wrt_y_interface(self, block_id, i, j, solver, face):
        max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)
        min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        ap_value = (max_v - min_v) * self.inv_dy[block_id]

        neighbour_block_id = self.bc.get_bc_value(block_id, face)
        if face == "north":
            an_index = self.mesh.map3Dto1D(neighbour_block_id, i, 1)
            an_value =  min_v * self.inv_dy[neighbour_block_id]

            as_index = self.mesh.map3Dto1D(block_id, i, j - 1)
            as_value = -1.0 * max_v * self.inv_dy[block_id]
        elif face == "south":
            an_index = self.mesh.map3Dto1D(block_id, i, j + 1)
            an_value =  min_v * self.inv_dy[block_id]

            as_index = self.mesh.map3Dto1D(neighbour_block_id, i, self.mesh.num_y[neighbour_block_id] - 2)
            as_value = -1.0 * max_v * self.inv_dy[neighbour_block_id]
        else:
            raise Exception("Unknown face")

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, an_index, an_value)
        solver.add_to_A(ap_index, as_index, as_value)

    def positive_in_x(self, block_id, i, j, solver):
        max_u = max(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        aw_index = self.mesh.map3Dto1D(block_id, i - 1, j)
        
        ap_value =  max_u * self.inv_dx[block_id]
        aw_value = -max_u * self.inv_dx[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, aw_index, aw_value)

    def negative_in_x(self, block_id, i, j, solver):
        min_u = min(self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        ae_index = self.mesh.map3Dto1D(block_id, i + 1, j)

        ap_value = -min_u * self.inv_dx[block_id]
        ae_value =  min_u * self.inv_dx[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, ae_index, ae_value)

    def positive_in_y(self, block_id, i, j, solver):
        max_v = max(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        as_index = self.mesh.map3Dto1D(block_id, i, j - 1)

        ap_value =  max_v * self.inv_dy[block_id]
        as_value = -max_v * self.inv_dy[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, as_index, as_value)
    
    def negative_in_y(self, block_id, i, j, solver):
        min_v = min(self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j], 0.0)

        ap_index = self.mesh.map3Dto1D(block_id, i, j)
        an_index = self.mesh.map3Dto1D(block_id, i, j + 1)

        ap_value = -min_v * self.inv_dy[block_id]
        an_value =  min_v * self.inv_dy[block_id]

        solver.add_to_A(ap_index, ap_index, ap_value)
        solver.add_to_A(ap_index, an_index, an_value)

# from swiftcfd.equations.numericalSchemes.numericalSchemesBase import NumericalSchemesBase
# from swiftcfd.enums import WRT, BCType, CornerType
# from swiftcfd.enums import PrimitiveVariables as pv


# class FirstOrderUpwind(NumericalSchemesBase):
#     def __init__(self, params, mesh, bc, cp, field_manager):
#         super().__init__(params, mesh, bc, cp, field_manager)

#         self.inv_dx = []
#         self.inv_dy = []
#         for block_id in range(self.mesh.num_blocks):
#             dx, dy = self.mesh.get_spacing(block_id)
#             self.inv_dx.append(1.0 / dx)
#             self.inv_dy.append(1.0 / dy)

#     # ------------------------------------------------------------------
#     # Public entry
#     # ------------------------------------------------------------------
#     def _compute_coefficients(self, direction, time, var_name, multiplier):
#         pass

#     def _compute_interior(self, direction, block_id, solver, var_name):
#         for (i, j) in self.mesh.internal_loop_single_block(block_id):
#             if direction == WRT.x:
#                 self.assemble_x(block_id, i, j, solver)
#             elif direction == WRT.y:
#                 self.assemble_y(block_id, i, j, solver)

#     def _east_boundary(self, direction, block_id, solver, var_name):
#         for (i, j) in self.mesh.loop_east(block_id, 1):
#             if direction == WRT.x:
#                 self.assemble_x(block_id, i, j, solver)
#             else:
#                 self.assemble_y(block_id, i, j, solver)

#     def _west_boundary(self, direction, block_id, solver, var_name):
#         for (i, j) in self.mesh.loop_west(block_id, 1):
#             if direction == WRT.x:
#                 self.assemble_x(block_id, i, j, solver)
#             else:
#                 self.assemble_y(block_id, i, j, solver)

#     def _north_boundary(self, direction, block_id, solver, var_name):
#         for (i, j) in self.mesh.loop_north(block_id, 0):
#             if direction == WRT.x:
#                 self.assemble_x(block_id, i, j, solver)
#             else:
#                 self.assemble_y(block_id, i, j, solver)

#     def _south_boundary(self, direction, block_id, solver, var_name):
#         for (i, j) in self.mesh.loop_south(block_id, 0):
#             if direction == WRT.x:
#                 self.assemble_x(block_id, i, j, solver)
#             else:
#                 self.assemble_y(block_id, i, j, solver)
    
#     def _bottom_left_corner(self, direction, block_id, solver, var_name):
#         pass
#         # corners = self.cp.get_corners(block_id)
#         # i = corners[CornerType.BOTTOM_LEFT]['i']
#         # j = corners[CornerType.BOTTOM_LEFT]['j']

#         # if direction == WRT.x:
#         #     self.assemble_x(block_id, i, j, solver)
#         # else:
#         #     self.assemble_y(block_id, i, j, solver)

#     def _bottom_right_corner(self, direction, block_id, solver, var_name):
#         pass
#         # corners = self.cp.get_corners(block_id)
#         # i = corners[CornerType.BOTTOM_RIGHT]['i']
#         # j = corners[CornerType.BOTTOM_RIGHT]['j']

#         # if direction == WRT.x:
#         #     self.assemble_x(block_id, i, j, solver)
#         # else:
#         #     self.assemble_y(block_id, i, j, solver)

#     def _top_left_corner(self, direction, block_id, solver, var_name):
#         pass
#         # corners = self.cp.get_corners(block_id)
#         # i = corners[CornerType.TOP_LEFT]['i']
#         # j = corners[CornerType.TOP_LEFT]['j']

#         # if direction == WRT.x:
#         #     self.assemble_x(block_id, i, j, solver)
#         # else:
#         #     self.assemble_y(block_id, i, j, solver)

#     def _top_right_corner(self, direction, block_id, solver, var_name):
#         pass
#         # corners = self.cp.get_corners(block_id)
#         # i = corners[CornerType.TOP_RIGHT]['i']
#         # j = corners[CornerType.TOP_RIGHT]['j']

#         # if direction == WRT.x:
#         #     self.assemble_x(block_id, i, j, solver)
#         # else:
#         #     self.assemble_y(block_id, i, j, solver)

#     # ------------------------------------------------------------------
#     # Core assembly routines
#     # ------------------------------------------------------------------
#     def assemble_x(self, block_id, i, j, solver):
#         u = self.field_manager.fields[pv.velocity_x.name()].picard_old[block_id, i, j]
#         ap = self.mesh.map3Dto1D(block_id, i, j)

#         if u >= 0.0:
#             self._x_west_flux(block_id, i, j, solver, ap, u)
#         else:
#             self._x_east_flux(block_id, i, j, solver, ap, u)

#     def assemble_y(self, block_id, i, j, solver):
#         v = self.field_manager.fields[pv.velocity_y.name()].picard_old[block_id, i, j]
#         ap = self.mesh.map3Dto1D(block_id, i, j)

#         if v >= 0.0:
#             self._y_south_flux(block_id, i, j, solver, ap, v)
#         else:
#             self._y_north_flux(block_id, i, j, solver, ap, v)

#     # ------------------------------------------------------------------
#     # X-direction fluxes
#     # ------------------------------------------------------------------
    # def _x_west_flux(self, block_id, i, j, solver, ap, u):
    #     inv_dx = self.inv_dx[block_id]

    #     if i > 0:
    #         aw = self.mesh.map3Dto1D(block_id, i - 1, j)
    #         solver.add_to_A(ap, ap,  u * inv_dx)
    #         solver.add_to_A(ap, aw, -u * inv_dx)
    #         return

    #     bc_type = self.bc.get_bc_type(block_id, "west")

    #     if bc_type == BCType.interface:
    #         nb = self.cp.get_block_id_at_interface(block_id, "west")
    #         aw = self.mesh.map3Dto1D(nb, self.mesh.num_x[nb] - 2, j)
    #         solver.add_to_A(ap, ap,  u * inv_dx)
    #         solver.add_to_A(ap, aw, -u * inv_dx)
    #     elif bc_type == BCType.neumann:
    #         g = self.bc.get_bc_value(block_id, "west")
    #         solver.add_to_A(ap, ap, u * inv_dx)
    #         solver.add_to_b(ap, u * g)

    # def _x_east_flux(self, block_id, i, j, solver, ap, u):
    #     inv_dx = self.inv_dx[block_id]

    #     if i < self.mesh.num_x[block_id] - 1:
    #         ae = self.mesh.map3Dto1D(block_id, i + 1, j)
    #         solver.add_to_A(ap, ap, -u * inv_dx)
    #         solver.add_to_A(ap, ae,  u * inv_dx)
    #         return

    #     bc_type = self.bc.get_bc_type(block_id, "east")

    #     if bc_type == BCType.interface:
    #         nb = self.cp.get_block_id_at_interface(block_id, "east")
    #         ae = self.mesh.map3Dto1D(nb, 1, j)
    #         solver.add_to_A(ap, ap, -u * inv_dx)
    #         solver.add_to_A(ap, ae,  u * inv_dx)
    #     elif bc_type == BCType.neumann:
    #         g = self.bc.get_bc_value(block_id, "east")
    #         solver.add_to_A(ap, ap, -u * inv_dx)
    #         solver.add_to_b(ap, u * g)

    # # ------------------------------------------------------------------
    # # Y-direction fluxes
    # # ------------------------------------------------------------------
    # def _y_south_flux(self, block_id, i, j, solver, ap, v):
    #     inv_dy = self.inv_dy[block_id]

    #     if j > 0:
    #         as_ = self.mesh.map3Dto1D(block_id, i, j - 1)
    #         solver.add_to_A(ap, ap,  v * inv_dy)
    #         solver.add_to_A(ap, as_, -v * inv_dy)
    #         return

    #     bc_type = self.bc.get_bc_type(block_id, "south")

    #     if bc_type == BCType.interface:
    #         nb = self.cp.get_block_id_at_interface(block_id, "south")
    #         as_ = self.mesh.map3Dto1D(nb, i, self.mesh.num_y[nb] - 2)
    #         solver.add_to_A(ap, ap,  v * inv_dy)
    #         solver.add_to_A(ap, as_, -v * inv_dy)
    #     elif bc_type == BCType.neumann:
    #         g = self.bc.get_bc_value(block_id, "south")
    #         solver.add_to_A(ap, ap, v * inv_dy)
    #         solver.add_to_b(ap, v * g)

    # def _y_north_flux(self, block_id, i, j, solver, ap, v):
    #     inv_dy = self.inv_dy[block_id]

    #     if j < self.mesh.num_y[block_id] - 1:
    #         an = self.mesh.map3Dto1D(block_id, i, j + 1)
    #         solver.add_to_A(ap, ap, -v * inv_dy)
    #         solver.add_to_A(ap, an,  v * inv_dy)
    #         return

    #     bc_type = self.bc.get_bc_type(block_id, "north")

    #     if bc_type == BCType.interface:
    #         nb = self.cp.get_block_id_at_interface(block_id, "north")
    #         an = self.mesh.map3Dto1D(nb, i, 1)
    #         solver.add_to_A(ap, ap, -v * inv_dy)
    #         solver.add_to_A(ap, an,  v * inv_dy)
    #     elif bc_type == BCType.neumann:
    #         g = self.bc.get_bc_value(block_id, "north")
    #         solver.add_to_A(ap, ap, -v * inv_dy)
    #         solver.add_to_b(ap, v * g)
