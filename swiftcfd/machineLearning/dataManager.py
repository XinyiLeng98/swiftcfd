from os.path import join, isdir
from os import listdir

import numpy as np
import pandas as pd

class DataManager:
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
        # write out trainign data (variables at different locations and time steps)
        if self.generate_training_data:
            # Skip poisoned data from a diverged run: NaN/Inf rows would silently
            # corrupt the training pool (and still pass a "csv exists" check).
            for var in self.training_variables:
                arr = np.asarray(pd.DataFrame(self.data[var]).values, dtype=float)
                if arr.size and not np.isfinite(arr).all():
                    print(f"[warning] trainingData_{var} has NaN/Inf — run likely "
                          f"diverged; NOT writing training data for this case.")
                    return
            for var in self.training_variables:
                df = pd.DataFrame(self.data[var])
                case = self.params('solver', 'output', 'filename')
                out_folder = join('output', case)
                df.to_csv(join(out_folder, f'trainingData_{var}.csv'), index=False, float_format="%.5e")
        
                # get simulation parameters

                # for now, all blocks have the same spacing dx and dy, so getting it from block_id=0
                # should be the same compared to all other blocks, if there are more than one. 
                dx, dy = self.mesh.get_spacing(0)
                dt = self.params('solver', 'time', 'dt')
                rho = self.params('solver', 'fluid', 'rho')
                nu = self.params('solver', 'fluid', 'nu')
                alpha = self.params('solver', 'fluid', 'alpha')
                
                # write out additional simulation parameters required for training and inference
                df = pd.DataFrame({'dx': [dx], 'dy': [dy], 'dt': [dt], 'rho': [rho], 'nu': [nu], 'alpha': [alpha]})
                df.to_csv(join(out_folder, 'simulationParameters.csv'), index=False, float_format="%.5e")
    
    @staticmethod
    def get_training_data(training_variables, validation_percentage = 0.2):

        variable_list = [v.strip() for v in training_variables.split(',')]

        training_files = DataManager.__find_all_trainig_data_sets()
        training_files_by_variables = DataManager.__organise_training_data_by_variable(
            training_files, training_variables
        )
        print(f'Found {len(training_files)} training data sets for variables: {training_variables}')

        # Load raw data for every variable first
        raw = {}
        for var in variable_list:
            x, y, sim_params = DataManager.__load_training_data(training_files_by_variables[var])
            raw[var] = (x, y, sim_params)

        # Verify all variables have the same number of samples (required for row alignment)
        n_samples = len(raw[variable_list[0]][0])
        for var in variable_list[1:]:
            if len(raw[var][0]) != n_samples:
                raise ValueError(
                    f"Variable '{var}' has {len(raw[var][0])} samples but "
                    f"'{variable_list[0]}' has {n_samples}. All variables must have "
                    f"the same number of training samples."
                )

        # One shared permutation keeps rows aligned across all variables
        n_val = int(validation_percentage * n_samples)
        idx   = np.random.permutation(n_samples)

        training_data = {}
        for var in variable_list:
            x, y, sim_params = raw[var]
            training_data[var] = {
                'x_train':               x[idx[n_val:]],
                'y_train':               y[idx[n_val:]],
                'training_parameters':   sim_params[idx[n_val:]],
                'x_validation':          x[idx[:n_val]],
                'y_validation':          y[idx[:n_val]],
                'validation_parameters': sim_params[idx[:n_val]],
            }

        return training_data

    @staticmethod
    def __find_all_trainig_data_sets():
        training_files = []
    
        for simulation_folder in listdir('output'):
            if not isdir(join('output', simulation_folder)):
                continue

            for output_file in listdir(join('output', simulation_folder)):
                if output_file.find('trainingData') != -1:
                    training_files.append(join('output', simulation_folder, output_file))
        
        return training_files

    @staticmethod
    def __organise_training_data_by_variable(training_files, variable_names):
        variable_names_split = variable_names.split(',')
        training_data_by_variable = {}

        for var in variable_names_split:
            training_data_by_variable[var] = []

        for training_file in training_files:
            var = DataManager.__get_variable_name(training_file)
            if var in variable_names_split:
                training_data_by_variable[var].append(training_file)
        
        return training_data_by_variable

    @staticmethod
    def __get_variable_name(training_file):
        return training_file.split('trainingData_')[1].split('.csv')[0]

    @staticmethod
    def __load_training_data(training_files):
        # load all training data sets, staying in float32 numpy throughout --
        # expanding millions of rows into Python lists (~700 B/row vs 80 B)
        # gets the process OOM-killed on multi-case datasets.
        temp_data_sets = []
        simulation_parameters = []
        for training_file in training_files:
            temp_data_set = pd.read_csv(training_file, dtype=np.float32)
            temp_data_sets.append(temp_data_set.to_numpy())
            number_of_rows = len(temp_data_set)

            # get the file path of the output directory
            output_folder = training_file.split('trainingData_')[0]
            temp_data_set = pd.read_csv(join(output_folder, 'simulationParameters.csv'))

            # get parameters
            dx = temp_data_set['dx'][0]
            dy = temp_data_set['dy'][0]
            dt = temp_data_set['dt'][0]
            rho = temp_data_set['rho'][0]
            nu = temp_data_set['nu'][0]
            alpha = temp_data_set['alpha'][0]

            simulation_parameter = np.tile(
                np.array([dx, dy, dt, rho, nu, alpha], dtype=np.float32),
                (number_of_rows, 1)
            )
            simulation_parameters.append(simulation_parameter)

        # stack the per-case arrays for data and simulation parameters
        data = np.concatenate(temp_data_sets, axis=0)
        simulation_parameters = np.concatenate(simulation_parameters, axis=0)

        # 20-column layout from trainingData.py:
        #   cols  0- 4: T^{n-2}  [center, west, east, south, north]
        #   cols  5- 9: T^{n-1}  [center, west, east, south, north]
        #   cols 10-14: T^{n}    [center, west, east, south, north]
        #   cols 15-19: T^{n+1}  [center, west, east, south, north]
        x = np.column_stack([
            data[:, 10],   # T^n center
            data[:, 12],   # T^n east   (i+1,j)
            data[:, 11],   # T^n west   (i-1,j)
            data[:, 14],   # T^n north  (i,j+1)
            data[:, 13],   # T^n south  (i,j-1)
            data[:,  5],   # T^{n-1} center
            data[:,  0],   # T^{n-2} center
        ])
        y = np.column_stack([
            data[:, 15],   # T^{n+1} center
            data[:, 17],   # T^{n+1} east
            data[:, 16],   # T^{n+1} west
            data[:, 19],   # T^{n+1} north
            data[:, 18],   # T^{n+1} south
        ])

        x = x.astype(np.float32)
        y = y.astype(np.float32)

        return x, y, simulation_parameters