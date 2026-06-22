# Lightweight imports that do NOT require PETSc — always available, including
# for ML-only training on a machine without PETSc installed (e.g. an HPC venv).
from swiftcfd.cla import CommandLineArgumentParser as command_line_argument_parser
from swiftcfd.parameters import Parameters as parameters
from swiftcfd.mesh import Mesh as mesh
from swiftcfd.output.output import Output as output
from swiftcfd.residuals import Residuals as residuals
from swiftcfd.log import Log as log
from swiftcfd.performanceStatistics import PerformanceStatistics as performance_statistics
from swiftcfd.machineLearning.dataManager import DataManager as ML_data_manager
from swiftcfd.machineLearning.model.modelFactory import create_model

# Full CFD solver imports — these require PETSc (petsc4py). Wrapped so the
# package can still be imported for `--train` in a PETSc-free environment.
# If PETSc is missing, these names are set to None and only the actual CFD
# solve (not ML training) is unavailable.
try:
    from swiftcfd.field.fieldManager import FieldManager as field_manager
    from swiftcfd.runtime import Runtime as runtime
    from swiftcfd.equations.equations.equationManager import EquationManager as equation_manager
    _PETSC_AVAILABLE = True
except ImportError as _petsc_import_error:
    _PETSC_AVAILABLE = False
    _PETSC_ERROR_MESSAGE = str(_petsc_import_error)
    field_manager = None
    runtime = None
    equation_manager = None
