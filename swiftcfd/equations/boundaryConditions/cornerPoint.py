from enum import Enum, auto
from swiftcfd.enums import BCType, CornerType

class CornerPoint:
    def __init__(self, mesh, bc):
        self.mesh = mesh
        self.bc = bc

        self.corner_points = []

        self.__setup_corners()

    def get_corners(self, block_id):
        return self.corner_points[block_id]

    def get_block_id_at_interface(self, block_id, face):
        assert self.bc.bc_type[block_id][face] == BCType.interface
        return self.bc.bc_value[block_id][face]

    def __setup_corners(self):
        for block_id in range(self.mesh.num_blocks):
            corner = {}

            east_bc_type = self.bc.bc_type[block_id]["east"]
            west_bc_type = self.bc.bc_type[block_id]["west"]
            north_bc_type = self.bc.bc_type[block_id]["north"]
            south_bc_type = self.bc.bc_type[block_id]["south"]

            east_bc_value = self.bc.bc_value[block_id]["east"]
            west_bc_value = self.bc.bc_value[block_id]["west"]
            north_bc_value = self.bc.bc_value[block_id]["north"]
            south_bc_value = self.bc.bc_value[block_id]["south"]

            bc_type, bc_value = self.__get_bc_type_at_corner(west_bc_type, west_bc_value, south_bc_type, south_bc_value)
            corner[CornerType.BOTTOM_LEFT] = {
                'type': bc_type,
                'value': bc_value,
                'i': 0,
                'j': 0
            }

            bc_type, bc_value = self.__get_bc_type_at_corner(east_bc_type, east_bc_value, south_bc_type, south_bc_value)
            corner[CornerType.BOTTOM_RIGHT] = {
                'type': bc_type,
                'value': bc_value,
                'i': self.mesh.num_x[block_id] - 1,
                'j': 0
            }

            bc_type, bc_value = self.__get_bc_type_at_corner(west_bc_type, west_bc_value, north_bc_type, north_bc_value)
            corner[CornerType.TOP_LEFT] = {
                'type': bc_type,
                'value': bc_value,
                'i': 0,
                'j': self.mesh.num_y[block_id] - 1
            }

            bc_type, bc_value = self.__get_bc_type_at_corner(east_bc_type, east_bc_value, north_bc_type, north_bc_value)
            corner[CornerType.TOP_RIGHT] = {
                'type': bc_type,
                'value': bc_value,
                'i': self.mesh.num_x[block_id] - 1,
                'j': self.mesh.num_y[block_id] - 1
            }

            self.corner_points.append(corner)

    def __get_bc_type_at_corner(self, bc_type1, bc_value1, bc_type2, bc_value2):
        if bc_type1 == BCType.dirichlet and bc_type2 == BCType.dirichlet:
            return BCType.dirichlet, 0.5 * (bc_value1 + bc_value2)
        elif bc_type1 == BCType.dirichlet:
            return BCType.dirichlet, bc_value1
        elif bc_type2 == BCType.dirichlet:
            return BCType.dirichlet, bc_value2
        elif bc_type1 == BCType.neumann and bc_type2 == BCType.neumann:
            return BCType.neumann, 0.5 * (bc_value1 + bc_value2)
        elif bc_type1 == BCType.neumann:
            return BCType.neumann, bc_value1
        elif bc_type2 == BCType.neumann:
            return BCType.neumann, bc_value2
        elif bc_type1 == BCType.interface and bc_type2 == BCType.interface:
            return BCType.interface, 0
        else:
            raise RuntimeError("Invalid corner point type")