from petsc4py import PETSc
from petsc4py import init as petsc_init

from swiftcfd.equations.linearAlgebraSolver.solverFactory import SolverFactory

class LinearAlgebraSolver():
    def __init__(self, params, mesh, var_name, has_nullspace):
        # total points in mesh
        self.total_cells = mesh.total_cells
        self.mesh = mesh
        self.var_name = var_name
        self.is_diagonal = True
        self.has_nullspace = has_nullspace

        # create coefficient matrix A
        self.A = PETSc.Mat().create()
        self.A.setSizes([self.total_cells, self.total_cells])
        self.A.setType(PETSc.Mat.Type.SEQAIJ)
        self.A.setPreallocationNNZ(9)
        self.A.setUp()

        # if the solver has fuly neumann conditions, remove the nullspace
        if self.has_nullspace:
            ns = PETSc.NullSpace().create(constant=True)
            self.A.setNullSpace(ns)
            self.A.setNearNullSpace(ns)

        # create right-hand side vector
        self.b = PETSc.Vec().createSeq(self.total_cells)

        # create linear solver
        self.ksp = SolverFactory().create(params, self.var_name)
        self.ksp.setOperators(self.A)
    
    def reset_A(self):
        self.A.zeroEntries()
    
    def reset_b(self):
        self.b.zeroEntries()

    def add_to_A(self, row, col, value):
        self.A.setValue(row, col, value, addv=PETSc.InsertMode.ADD_VALUES)

    def insert_into_A(self, row, col, value):
        self.A.setValue(row, col, value, addv=PETSc.InsertMode.INSERT_VALUES)
    
    def add_to_b(self, row, value):
        self.b.setValue(row, value, addv=PETSc.InsertMode.ADD_VALUES)

    def insert_into_b(self, row, value):
        self.b.setValue(row, value, addv=PETSc.InsertMode.INSERT_VALUES)

    def assemble(self):
        self.A.assemblyBegin()
        self.A.assemblyEnd()

        self.b.assemblyBegin()
        self.b.assemblyEnd()

        # if self.var_name == 'p':
        #     self.A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
        #     self.A.setOption(PETSc.Mat.Option.SPD, True)

    def field_to_petsc_vec(self, field):
        vec = PETSc.Vec().createWithArray(field._data)
        return vec

    def solve(self, field):
        self.ksp.reset()
        self.ksp.setOperators(self.A)

        # check if matrix can be inverted trivally (only contains diagonal)
        self.is_diagonal = self.__check_for_diagonal_matrix()

        # convert field to PETSc compatible vector with zero copy
        field_petsc = self.field_to_petsc_vec(field)

        if self.is_diagonal:
            field_petsc.pointwiseDivide(self.b, self.A.getDiagonal())
        else:
            self.ksp.solve(self.b, field_petsc)

    def view(self):
        viewer = PETSc.Viewer().createASCII(f"{self.var_name}.txt")
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_INFO)
        self.A.view(viewer)
        viewer.destroy()

        viewer = PETSc.Viewer().createASCII(f"{self.var_name}_values.txt")
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
        self.A.view(viewer)
        viewer.destroy()

        viewer = PETSc.Viewer().createASCII(f"{self.var_name}_structure.txt")
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_DENSE)
        self.A.view(viewer)
        viewer.destroy()

        # sigma_max, sigma_min = self.ksp.computeExtremeSingularValues()
        # cond_est = sigma_max / sigma_min
        # self.condition_number.append(cond_est)

    def get_solver_statistics(self):
        num_iterations = self.ksp.getIterationNumber()
        res_norm = self.ksp.getResidualNorm()
        has_converged = self.ksp.getConvergedReason() >= 0
        return self.is_diagonal, num_iterations, res_norm, has_converged

    def __check_for_diagonal_matrix(self):
        number_of_rows, _ = self.A.getSize()
        info = self.A.getInfo(PETSc.Mat.InfoType.LOCAL)
        # nz_used is the "non-zero" entries in the matrix. If there are as many
        # nz entries as there are rows, there is only one entry per row.
        # Thus, the matrix is diagonal and can be trivally inverted for
        # explicit time integration
        return info["nz_used"] == number_of_rows