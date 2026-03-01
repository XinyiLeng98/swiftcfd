from os.path import join

import pandas as pd

class TrainingData:
    def __init__(self, params, mesh, field_manager):
        self.params = params
        self.mesh = mesh
        self.field_manager = field_manager

        self.case = self.params('solver', 'output', 'filename')
        self.out_folder = join('output', self.case)

        self.generate_training_data = self.params('solver', 'ML', 'generateTrainingData')
        self.training_variables = self.params('solver', 'ML', 'trainingVariables')

        self.data = {}

        for var in self.training_variables:
            self.data[var] = {
                f'{var}^n-2_i,j': [],
                f'{var}^n-2_i-1,j': [],
                f'{var}^n-2_i+1,j': [],
                f'{var}^n-2_i,j-1': [],
                f'{var}^n-2_i,j+1': [],

                f'{var}^n-1_i,j': [],
                f'{var}^n-1_i-1,j': [],
                f'{var}^n-1_i+1,j': [],
                f'{var}^n-1_i,j-1': [],
                f'{var}^n-1_i,j+1': [],

                f'{var}^n_i,j': [],
                f'{var}^n_i-1,j': [],
                f'{var}^n_i+1,j': [],
                f'{var}^n_i,j-1': [],
                f'{var}^n_i,j+1': [],

                f'{var}^n+1_i,j': [],
                f'{var}^n+1_i-1,j': [],
                f'{var}^n+1_i+1,j': [],
                f'{var}^n+1_i,j-1': [],
                f'{var}^n+1_i,j+1': []
            }
    
    def should_train(self, runtime):
        return self.generate_training_data and runtime.current_timestep > 1
    
    def commit_training_data(self):
        for var in self.training_variables:
            for (block, i, j) in self.mesh.loop_all_internal_cells():
                self.data[var][f'{var}^n-2_i,j'].append(self.field_manager.fields[var].oldoldold[block, i, j])
                self.data[var][f'{var}^n-2_i-1,j'].append(self.field_manager.fields[var].oldoldold[block, i - 1, j])
                self.data[var][f'{var}^n-2_i+1,j'].append(self.field_manager.fields[var].oldoldold[block, i + 1, j])
                self.data[var][f'{var}^n-2_i,j-1'].append(self.field_manager.fields[var].oldoldold[block, i, j - 1])
                self.data[var][f'{var}^n-2_i,j+1'].append(self.field_manager.fields[var].oldoldold[block, i, j + 1])

                self.data[var][f'{var}^n-1_i,j'].append(self.field_manager.fields[var].oldold[block, i, j])
                self.data[var][f'{var}^n-1_i-1,j'].append(self.field_manager.fields[var].oldold[block, i - 1, j])
                self.data[var][f'{var}^n-1_i+1,j'].append(self.field_manager.fields[var].oldold[block, i + 1, j])
                self.data[var][f'{var}^n-1_i,j-1'].append(self.field_manager.fields[var].oldold[block, i, j - 1])
                self.data[var][f'{var}^n-1_i,j+1'].append(self.field_manager.fields[var].oldold[block, i, j + 1])

                self.data[var][f'{var}^n_i,j'].append(self.field_manager.fields[var].old[block, i, j])
                self.data[var][f'{var}^n_i-1,j'].append(self.field_manager.fields[var].old[block, i - 1, j])
                self.data[var][f'{var}^n_i+1,j'].append(self.field_manager.fields[var].old[block, i + 1, j])
                self.data[var][f'{var}^n_i,j-1'].append(self.field_manager.fields[var].old[block, i, j - 1])
                self.data[var][f'{var}^n_i,j+1'].append(self.field_manager.fields[var].old[block, i, j + 1])

                self.data[var][f'{var}^n+1_i,j'].append(self.field_manager.fields[var][block, i, j])
                self.data[var][f'{var}^n+1_i-1,j'].append(self.field_manager.fields[var][block, i - 1, j])
                self.data[var][f'{var}^n+1_i+1,j'].append(self.field_manager.fields[var][block, i + 1, j])
                self.data[var][f'{var}^n+1_i,j-1'].append(self.field_manager.fields[var][block, i, j - 1])
                self.data[var][f'{var}^n+1_i,j+1'].append(self.field_manager.fields[var][block, i, j + 1])

    def write(self):
        for var in self.training_variables:
            df = pd.DataFrame(self.data[var])
            case = self.params('solver', 'output', 'filename')
            out_folder = join('output', case)
            df.to_csv(join(out_folder, f'trainingData_{var}.csv'), index=False, float_format="%.5e")