from swiftcfd.enums import PrimitiveVariables as pv
from swiftcfd.equations.equations.baseEquation import BaseEquation
from swiftcfd.enums import WRT
from swiftcfd.equations.numericalSchemes.implicit.firstOrderEuler import FirstOrderEuler
from swiftcfd.equations.numericalSchemes.implicit.secondOrderBackwards import SecondOrderBackwards
from swiftcfd.equations.numericalSchemes.implicit.secondOrderCentral import SecondOrderCentral

class HeatDiffusion(BaseEquation):
    def __init__(self, params, mesh, field_manager):
        super().__init__(params, mesh, field_manager)
        
        self.has_first_order_time_derivative = True
        self.has_second_order_space_derivative = True

        constructor_arguments = (self.params, self.mesh, self.bc, self.field_manager)

        self.dTdt = SecondOrderBackwards(*constructor_arguments)
        self.d2Tdx2 = SecondOrderCentral(*constructor_arguments)
        self.d2Tdy2 = SecondOrderCentral(*constructor_arguments)

    def first_order_time_derivative(self, runtime):
        self.dTdt.apply(WRT.t, self.solver, runtime, self.get_variable_name())

    def second_order_space_derivative(self, runtime):
        alpha = self.params('solver', 'fluid', 'alpha')
        self.d2Tdx2.apply(WRT.x, self.solver, runtime, self.get_variable_name(), -1.0 * alpha)
        self.d2Tdy2.apply(WRT.y, self.solver, runtime, self.get_variable_name(), -1.0 * alpha)
    
    def get_diffusion_coefficients(self):
        alpha = self.params('solver', 'fluid', 'alpha')
        return alpha

    def get_variable_name(self):
        return pv.temperature.name()
