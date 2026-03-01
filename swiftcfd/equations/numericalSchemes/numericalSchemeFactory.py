from swiftcfd.equations.numericalSchemes.implicit.firstOrderEuler import FirstOrderEuler
from swiftcfd.equations.numericalSchemes.implicit.secondOrderBackwards import SecondOrderBackwards
from swiftcfd.equations.numericalSchemes.implicit.firstOrderUpwind import FirstOrderUpwind
from swiftcfd.equations.numericalSchemes.implicit.secondOrderUpwind import SecondOrderUpwind
from swiftcfd.equations.numericalSchemes.implicit.secondOrderCentral import SecondOrderCentral

class NumericalSchemeFactory:
    def __init__(self, params, mesh, boundary_conditions, field_manager):
        self.params = params
        self.mesh = mesh
        self.bc = boundary_conditions
        self.field_manager = field_manager
        self.constructor_arguments = (self.params, self.mesh, self.bc, self.field_manager)

    def create_time_integration_scheme(self, equation):
        # time integration scheme
        if equation.has_first_order_time_derivative:
            scheme = self.params('solver', 'schemes', 'timeIntegrationScheme')
            if scheme == 'firstOrderEuler':
                return FirstOrderEuler(*self.constructor_arguments)
            elif scheme == 'secondOrderBackwards':
                return SecondOrderBackwards(*self.constructor_arguments)
            else:
                raise Exception('Unknown time integration scheme: ' + scheme)
    
    def create_first_order_space_derivative_scheme(self, equation):
        # non-linear scheme
        if equation.has_first_order_space_derivative:
            scheme = self.params('solver', 'schemes', 'nonLinearScheme')
            if scheme == 'firstOrderUpwind':
                return FirstOrderUpwind(*self.constructor_arguments)
            elif scheme == 'secondOrderUpwind':
                return SecondOrderUpwind(*self.constructor_arguments)
            elif scheme == 'centralDifferencing':
                return CentralDifferencing(*self.constructor_arguments)
            else:
                raise Exception('Unknown non-linear scheme: ' + scheme)

    def create_second_order_space_derivative_scheme(self, equation):
        # diffusion scheme
        if equation.has_second_order_space_derivative:
            scheme = self.params('solver', 'schemes', 'diffusionScheme')
            if scheme == 'secondOrderCentral':
                return SecondOrderCentral(*self.constructor_arguments)
            else:
                raise Exception('Unknown diffusion scheme: ' + scheme)



