from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.enums import WRT
from swiftcfd.equations.numericalSchemes.numericalSchemeFactory import NumericalSchemeFactory
from swiftcfd.gradients.firstOrderGradient import FirstOrderGradient as gradient

class Pressure(BaseEquation):
    def __init__(self, params, mesh, field_manager, u_solver, v_solver):
        super().__init__(params, mesh, field_manager)
        
        self.has_second_order_space_derivative = True
        self.has_source = True

        numerical_schemes = NumericalSchemeFactory(self.params, self.mesh, self.bc, self.field_manager)

        self.d2pdx2 = numerical_schemes.create_second_order_space_derivative_scheme(self)
        self.d2pdy2 = numerical_schemes.create_second_order_space_derivative_scheme(self)

        self.grad_u = gradient(self.mesh, self.field_manager, pv.velocity_x.name())
        self.grad_v = gradient(self.mesh, self.field_manager, pv.velocity_y.name())
        self.grad_p = gradient(self.mesh, self.field_manager, pv.pressure.name())

        self.u_solver = u_solver
        self.v_solver = v_solver

        self.uses_second_order_time_integration = \
            self.params('solver', 'schemes', 'timeIntegrationScheme') == 'secondOrderBackwards'

    def second_order_space_derivative(self, runtime):
        self.d2pdx2.apply(WRT.x, self.solver, runtime, self.get_variable_name())
        self.d2pdy2.apply(WRT.y, self.solver, runtime, self.get_variable_name())

    def source(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt

        self.grad_u.compute()
        self.grad_v.compute()

        ap_u = self.u_solver.A.getDiagonal().getArray()
        ap_v = self.v_solver.A.getDiagonal().getArray()

        for block in range(0, self.mesh.num_blocks):
            dx, dy = self.mesh.get_spacing(block)
            V = dx * dy
            for (i, j) in self.mesh.internal_loop_single_block(block):

                index = self.mesh.map3Dto1D(block, i, j)

                uip1 = self.field_manager.fields[pv.velocity_x.name()][block, i + 1, j]
                ui   = self.field_manager.fields[pv.velocity_x.name()][block, i, j]
                uim1 = self.field_manager.fields[pv.velocity_x.name()][block, i - 1, j]
                
                vjp1 = self.field_manager.fields[pv.velocity_y.name()][block, i, j + 1]
                vj   = self.field_manager.fields[pv.velocity_y.name()][block, i, j]
                vjm1 = self.field_manager.fields[pv.velocity_y.name()][block, i, j - 1]

                pip1 = self.field_manager.fields[pv.pressure.name()][block, i + 1, j]
                pi   = self.field_manager.fields[pv.pressure.name()][block, i, j]
                pim1 = self.field_manager.fields[pv.pressure.name()][block, i - 1, j]

                pjp1 = self.field_manager.fields[pv.pressure.name()][block, i, j + 1]
                pj   = self.field_manager.fields[pv.pressure.name()][block, i, j]
                pjm1 = self.field_manager.fields[pv.pressure.name()][block, i, j - 1]

                if i == 1:
                    pim2 = 2.0 * pi - pim1
                else:
                    pim2 = self.field_manager.fields[pv.pressure.name()][block, i - 2, j]

                if i == self.mesh.num_x[block] - 2:
                    pip2 = 2.0 * pi - pip1
                else:
                    pip2 = self.field_manager.fields[pv.pressure.name()][block, i + 2, j]

                if j == 1:
                    pjm2 = 2.0 * pj - pjm1
                else:
                    pjm2 = self.field_manager.fields[pv.pressure.name()][block, i, j - 2]

                if j == self.mesh.num_y[block] - 2:
                    pjp2 = 2.0 * pj - pjp1
                else:
                    pjp2 = self.field_manager.fields[pv.pressure.name()][block, i, j + 2]

                # face velocities
                u_e = 0.5 * (uip1 + ui)
                u_w = 0.5 * (ui + uim1)
                v_n = 0.5 * (vjp1 + vj)
                v_s = 0.5 * (vj + vjm1)

                # correction scaling
                du = V / ap_u[index]
                dv = V / ap_v[index]

                # du = V / (rho * 3.0 / (2.0 * dt))
                # dv = V / (rho * 3.0 / (2.0 * dt))

                # pressure gradient at faces
                dp_e = (pip1 - pi) / dx
                dp_w = (pi - pim1) / dx
                dp_n = (pjp1 - pj) / dy
                dp_s = (pj - pjm1) / dy

                # pressure gradient at cell centers
                dpi = (pip1 - pim1) / (2.0 * dx)
                dpj = (pjp1 - pjm1) / (2.0 * dy)

                # pressure gradient at cell neighbours
                dpip1 = (pip2 - pi) / (2.0 * dx)
                dpim1 = (pi - pim2) / (2.0 * dx)
                dpjp1 = (pjp2 - pj) / (2.0 * dy)
                dpjm1 = (pj - pjm2) / (2.0 * dy)

                # interpolated pressure gradient at faces
                dp_e_interpolated = 0.5 * (dpi + dpip1)
                dp_w_interpolated = 0.5 * (dpi + dpim1)
                dp_n_interpolated = 0.5 * (dpj + dpjp1)
                dp_s_interpolated = 0.5 * (dpj + dpjm1)

                u_e_corrected = u_e - du * (dp_e - dp_e_interpolated)
                u_w_corrected = u_w - du * (dp_w - dp_w_interpolated)
                v_n_corrected = v_n - dv * (dp_n - dp_n_interpolated)
                v_s_corrected = v_s - dv * (dp_s - dp_s_interpolated)

                # divergence = (u_e_corrected - u_w_corrected) / dx + (v_n_corrected - v_s_corrected) / dy
                divergence = self.grad_u.x[block, i, j] + self.grad_v.y[block, i, j]

                if self.uses_second_order_time_integration and runtime.current_timestep > 1:
                    rhs = (3.0 * rho / (2.0 * dt)) * divergence
                else:
                    rhs = (rho / dt) * divergence
                
                self.solver.add_to_b(index, rhs)
            
    def post_solve_task(self, runtime):
        rho = self.params('solver', 'fluid', 'rho')
        dt = runtime.dt
        self.grad_p.compute()

        if self.uses_second_order_time_integration and runtime.current_timestep > 1:
            multiplier = (2.0 * dt) / (3.0 * rho)
        else:
            multiplier = dt / rho

        for (block, i, j) in self.mesh.internal_loop_all_blocks():
            u = self.field_manager.fields[pv.velocity_x.name()][block, i, j] - multiplier * self.grad_p.x[block, i, j]
            v = self.field_manager.fields[pv.velocity_y.name()][block, i, j] - multiplier * self.grad_p.y[block, i, j]

            self.field_manager.fields[pv.velocity_x.name()][block, i, j] = u
            self.field_manager.fields[pv.velocity_y.name()][block, i, j] = v

    def get_diffusion_coefficients(self):
        pass

    def get_variable_name(self):
        return pv.pressure.name()
