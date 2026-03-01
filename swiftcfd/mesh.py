import numpy as np

class Mesh():
    def __init__(self, params):
        self.num_blocks = len(params('mesh'))

        self.x_start = np.zeros(self.num_blocks)
        self.x_end = np.zeros(self.num_blocks)
        self.y_start = np.zeros(self.num_blocks)
        self.y_end = np.zeros(self.num_blocks)
        self.num_cells_x = np.zeros(self.num_blocks, dtype = int)
        self.num_cells_y = np.zeros(self.num_blocks, dtype = int)
        self.num_points_x = np.zeros(self.num_blocks, dtype = int)
        self.num_points_y = np.zeros(self.num_blocks, dtype = int)
        self.cells_offset = np.zeros(self.num_blocks, dtype = int)
        self.points_offset = np.zeros(self.num_blocks, dtype = int)

        self.total_cells = 0
        self.total_points = 0
        self.cells_per_block = []
        self.points_per_block = []

        for i in range(0, self.num_blocks):
            block = f'block{i + 1}'
            self.x_start[i] = params('mesh', block, 'x', 'start')
            self.x_end[i] = params('mesh', block, 'x', 'end')
            self.y_start[i] = params('mesh', block, 'y', 'start')
            self.y_end[i] = params('mesh', block, 'y', 'end')
            self.num_cells_x[i] = int(params('mesh', block, 'x', 'numCells'))
            self.num_cells_y[i] = int(params('mesh', block, 'y', 'numCells'))
            self.num_points_x[i] = self.num_cells_x[i] + 1
            self.num_points_y[i] = self.num_cells_y[i] + 1

            self.cells_per_block.append(self.num_cells_x[i] * self.num_cells_y[i])
            self.points_per_block.append(self.num_points_x[i] * self.num_points_y[i])
            self.total_cells += self.cells_per_block[i]
            self.total_points += self.points_per_block[i]

            if i == 0:
                self.cells_offset[i] = 0
                self.points_offset[i] = 0
            else:
                self.cells_offset[i] = self.cells_offset[i - 1] + self.cells_per_block[i - 1]
                self.points_offset[i] = self.points_offset[i - 1] + self.num_points_x[i - 1] * self.num_points_y[i - 1]
            
        self.x = []
        self.y = []

    def map3Dto1D(self, block_id, i, j):
        offset = self.cells_offset[block_id]
        stride = self.num_cells_x[block_id] * j
        return int(offset + stride + i)        

    def create(self):
        for block in range(0, self.num_blocks):
            x_start = self.x_start[block]
            x_end = self.x_end[block]
            
            y_start = self.y_start[block]
            y_end = self.y_end[block]
            
            num_points_x = self.num_points_x[block]
            num_points_y = self.num_points_y[block]

            x = np.zeros((num_points_x, num_points_y))
            y = np.zeros((num_points_x, num_points_y))

            for i in range(0, num_points_x):
                for j in range(0, num_points_y):
                    x[i][j] = x_start + i * (x_end - x_start) / (num_points_x - 1)
                    y[i][j] = y_start + j * (y_end - y_start) / (num_points_y - 1)

            self.x.append(x)
            self.y.append(y)
    
    def get_min_spacing(self):
        dx = self.x[0][1][0] - self.x[0][0][0]
        dy = self.y[0][0][1] - self.y[0][0][0]
        for i in range(1, self.num_blocks):
            dx = min(dx, self.x[i][1][0] - self.x[i][0][0])
            dy = min(dy, self.y[i][0][1] - self.y[i][0][0])
        return min(dx, dy)

    def get_spacing(self, block_id):
        dx = self.x[block_id][1][0] - self.x[block_id][0][0]
        dy = self.y[block_id][0][1] - self.y[block_id][0][0]
        return dx, dy

    def loop_all_internal_cells(self):
        for block in range(0, self.num_blocks):
            for i in range(1, self.num_cells_x[block] - 1):
                for j in range(1, self.num_cells_y[block] - 1):
                    yield block, i, j

    def loop_internal_cells(self, block_id):
        for i in range(1, self.num_cells_x[block_id] - 1):
            for j in range(1, self.num_cells_y[block_id] - 1):
                yield i, j

    def loop_all_cells(self):
        for block in range(0, self.num_blocks):
            for i in range(0, self.num_cells_x[block]):
                for j in range(0, self.num_cells_y[block]):
                    yield block, i, j

    def loop_cells(self, block_id):
        for i in range(0, self.num_cells_x[block_id]):
            for j in range(0, self.num_cells_y[block_id]):
                yield i, j
    
    def loop_east_bc(self, block_id, offset = 0):
        for j in range(offset, self.num_cells_y[block_id] - offset):
            yield int(self.num_cells_x[block_id]) - 1, j
    
    def loop_west_bc(self, block_id, offset = 0):
        for j in range(offset, self.num_cells_y[block_id] - offset):
            yield 0, j
    
    def loop_north_bc(self, block_id, offset = 0):
        for i in range(offset, self.num_cells_x[block_id] - offset):
            yield i, int(self.num_cells_y[block_id] - 1)

    def loop_south_bc(self, block_id, offset = 0):
        for i in range(offset, self.num_cells_x[block_id] - offset):
            yield i, 0
    