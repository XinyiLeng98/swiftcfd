import numpy as np
from petsc4py import PETSc

class Field():
    def __init__(self, mesh, name):
        self.mesh = mesh
        self.name = name
        self.old = None
        self.oldold = None
        self.oldoldold = None
        self.picard_old = None

        # solution data
        self._data = np.zeros(self.mesh.total_cells, dtype=PETSc.ScalarType())

    def update_solution(self):
        np.copyto(self.oldoldold._data, self.oldold._data)
        np.copyto(self.oldold._data, self.old._data)
        np.copyto(self.old._data, self._data)

    def update_picard_solution(self):
        np.copyto(self.picard_old._data, self._data)

    def __getitem__(self, idx):
        block_id, i, j = idx
        index = self.mesh.map3Dto1D(block_id, i, j)
        return self._data[index]

    def __setitem__(self, idx, value):
        block_id, i, j = idx
        index = self.mesh.map3Dto1D(block_id, i, j)
        self._data[index] = value