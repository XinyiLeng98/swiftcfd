from os.path import join, exists
from os import listdir, remove
from os import mkdir

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

class Output():
    def __init__(self, params, mesh, field_manager):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager
        self.case = self.params('solver', 'output', 'filename')
        self.writing_frequency = self.params('solver', 'output', 'writingFrequency')
        self.out_folder = join('output', self.case)

        # create output folder if not exists and delete old files
        if not exists(self.out_folder):
            mkdir(self.out_folder)

    def write_tecplot_file(self, iteration = -1):
        filename = self.case

        if iteration == -1:
            filename += '.dat' 
            self._write_tecplot(self.case, filename)
        else:
            if iteration % self.writing_frequency == 0:
                filename += f'_{iteration:010d}.dat' 
                self._write_tecplot(self.case, filename)

    def _write_tecplot(self, case, filename):
        with open(join(self.out_folder, filename), 'w') as f:
            f.write(f'TITLE = "{case}"\n')
            f.write('VARIABLES = "x", "y"')
            for key in self.field_manager.fields.keys():
                f.write(f', "{key}"')
            f.write('\n')

            total_number_of_variables = 2 + len(self.field_manager.fields)

            for block in range(self.mesh.num_blocks):
                num_points_x = self.mesh.num_points_x[block]
                num_points_y = self.mesh.num_points_y[block]
                num_cells_x = self.mesh.num_cells_x[block]
                num_cells_y = self.mesh.num_cells_y[block]

                f.write(
                    f'\nZONE T="Block{block+1}", '
                    f'I={num_points_x}, J={num_points_y}, '
                    f'DATAPACKING=BLOCK, '
                    f'VARLOCATION=([3-{total_number_of_variables}]=CELLCENTERED)\n'
                )

                for j in range(num_points_y):
                    for i in range(num_points_x):
                        f.write(f"{self.mesh.x[block][i][j]}\n")

                for j in range(num_points_y):
                    for i in range(num_points_x):
                        f.write(f"{self.mesh.y[block][i][j]}\n")

                for key, value in self.field_manager.fields.items():
                    for j in range(num_cells_y):
                        for i in range(num_cells_x):
                            f.write(f"{value[block, i, j]}\n")

    
    def plot_contours(self):
        n_fields = len(self.field_manager.fields)

        fig, axes = plt.subplots(
            n_fields, 1,
            figsize=(8, 2.25 * n_fields),
            sharex=True
        )

        # In case there is only one field
        if n_fields == 1:
            axes = [axes]

        for ax, (field_name, field_data) in zip(axes, self.field_manager.fields.items()):
            vmin = np.inf
            vmax = -np.inf

            # ---- first pass: get min/max for this subplot ----
            for block in range(self.mesh.num_blocks):
                nx = self.mesh.num_cells_x[block]
                ny = self.mesh.num_cells_y[block]
                offset = self.mesh.cells_offset[block]

                field = field_data._data[offset : offset + nx * ny].reshape(ny, nx)
                vmin = min(vmin, field.min())
                vmax = max(vmax, field.max())

            # ---- second pass: plotting all blocks on the same subplot ----
            cs = None
            for block in range(self.mesh.num_blocks):
                nx = self.mesh.num_cells_x[block]
                ny = self.mesh.num_cells_y[block]
                offset = self.mesh.cells_offset[block]

                field = field_data._data[offset : offset + nx * ny].reshape(ny, nx)
                dx, dy = self.mesh.get_spacing(block)
                x = self.mesh.x[block][:-1, :-1].T + dx/2
                y = self.mesh.y[block][:-1, :-1].T + dy/2

                cs = ax.contourf(
                    x, y, field,
                    levels=20,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="jet"
                )

            ax.set_aspect("equal")
            ax.set_ylabel(field_name)

            # ---- colorbar per subplot ----
            fig.colorbar(
                cs,          # link to the last QuadContourSet of this subplot
                ax=ax,
                orientation="horizontal",
                location="top",
                shrink=0.8,
                pad=0.05
            )

        axes[-1].set_xlabel("x")

        fig.tight_layout()
        fig.savefig(
            join(self.out_folder, "contours.png"),
            dpi=300,
            bbox_inches="tight"
        )
        plt.close(fig)

    def plot_residuals(self):
        df = pd.read_csv(join(self.out_folder, 'residuals.csv'))

        plt.style.use('bmh')
        fig, ax = plt.subplots()

        for var_name in df.columns:
            ax.plot(df.index, df[var_name], label=var_name)

        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual')
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            join(self.out_folder, "residuals.png"),
            dpi=300,
            bbox_inches="tight"
        )  
