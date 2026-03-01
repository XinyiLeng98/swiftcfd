from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.enums import WRT
from swiftcfd.equations.numericalSchemes.numericalSchemeFactory import NumericalSchemeFactory

class yMomentum(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_first_order_time_derivative = True
        self.has_first_order_space_derivative = True
        self.has_second_order_space_derivative = True

        self.requires_linearisation = True

        numerical_schemes = NumericalSchemeFactory(self.params, self.mesh, self.bc, self.field_manager)

        self.dvdt = numerical_schemes.create_time_integration_scheme(self)
        self.dvdx = numerical_schemes.create_first_order_space_derivative_scheme(self)
        self.dvdy = numerical_schemes.create_first_order_space_derivative_scheme(self)

        self.d2vdx2 = numerical_schemes.create_second_order_space_derivative_scheme(self)
        self.d2vdy2 = numerical_schemes.create_second_order_space_derivative_scheme(self)

    def first_order_time_derivative(self, runtime):
        self.dvdt.apply(WRT.t, self.solver, runtime, self.get_variable_name())

    def first_order_space_derivative(self, runtime):
        self.dvdx.apply(WRT.x, self.solver, runtime, self.get_variable_name())
        self.dvdy.apply(WRT.y, self.solver, runtime, self.get_variable_name())

    def second_order_space_derivative(self, runtime):
        nu = self.params('solver', 'fluid', 'nu')
        self.d2vdx2.apply(WRT.x, self.solver, runtime, self.get_variable_name(), -1.0 * nu)
        self.d2vdy2.apply(WRT.y, self.solver, runtime, self.get_variable_name(), -1.0 * nu)
    
    def get_diffusion_coefficients(self):
        nu = self.params('solver', 'fluid', 'nu')
        return nu

    def get_variable_name(self):
        return pv.velocity_y.name()
