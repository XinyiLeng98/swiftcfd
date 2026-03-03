import copy
import os
import shutil

import swiftcfd

from swiftcfd.enums import PrimitiveVariables as pv

def run():
    # read command line arguments
    cla_parser = swiftcfd.command_line_argument_parser()

    # create parameters
    params = swiftcfd.parameters()
    params.read_from_file(cla_parser.arguments.input)

    # ADD: check for multiple blocks
    original_block = copy.deepcopy(params.params['boundaryCondition'])    
    all_blocks_names = list(original_block.keys())
    num_mesh = len(params.params['mesh'])

    if len(all_blocks_names) > num_mesh:
        training_blocks = all_blocks_names[:-1]
        validation_block = all_blocks_names[-1]
        print(f"\n  Training blocks: {training_blocks}")
        print(f"  Validation block: {validation_block}")
       
    else:
        training_blocks = all_blocks_names
        validation_block = None

    case_name = params('solver', 'output', 'filename')
    out_folder = os.path.join('output', case_name)
    csv_files = []

    if validation_block is not None:
        all_blocks = training_blocks + [validation_block]
    else:
        all_blocks = training_blocks

    for block in all_blocks:
        is_validation = (block == validation_block)
        label = "VALIDATION" if is_validation else "TRAINING"
        params.params['boundaryCondition'] = {'block1': original_block[block]}  # ★ CHANGED: set BC to current block

        print(f"\n--- Running simulation for {label} block {block} ---")
        bc = original_block[block]
        for face in ['west', 'east', 'south', 'north']:
            print(f"    {face}: {bc[face]['T']['type']}({bc[face]['T']['value']})")

        # create mesh
        mesh = swiftcfd.mesh(params)
        mesh.create()

        # create governign equation
        eqm = swiftcfd.equation_manager(params, mesh)

        # create runtime handler
        runtime = swiftcfd.runtime(params, mesh, eqm.field_manager, eqm.equations)

        # create output
        out_file = swiftcfd.output(params, mesh, eqm.field_manager)
        
        # create performance statistics
        stats = swiftcfd.performance_statistics(params, eqm.equations)
        stats.timer_start()

        # logger class to print output to console
        log = swiftcfd.log()

        # create residual calculating object
        residuals = swiftcfd.residuals(params, eqm.field_manager)

        # create machine learning training data
        training = swiftcfd.ML_training_data(params, mesh, eqm.field_manager)

        # loop over time
        while (runtime.has_not_reached_final_time()):
            # copy solution
            eqm.field_manager.update_solution()

            # print time info to console
            log.print_time_info(runtime)

            # linearisation step through picard iterations
            while(runtime.has_not_reached_final_picard_iteration()):
                # update picard solution
                eqm.field_manager.update_picard_solution()

                # solve non-linear equations (e.q. momentum equations)
                eqm.solve_non_linear_equations(runtime, stats)

                # compute picard residuals
                has_converged = residuals.check_picard_convergence(runtime)

                # print time step statistics
                log.print_picard_iteration(runtime, eqm.equations, residuals)

                if has_converged:
                    runtime.current_picard_iteration = 0
                    break

            # solve linear equations (e.g. pressure poisson, temperature)
            eqm.solve_linear_equations(runtime, stats)
            
            # update time steps
            runtime.update_time()

            # cehck for simulation convergence
            has_converged = residuals.check_convergence(runtime)

            # print convergence information for current time step
            log.print_convergence_info(runtime, eqm.equations, residuals)

            # save solution animation
            if params('solver', 'output', 'writingFrequency') > 0 and runtime.current_timestep % params('solver', 'output', 'writingFrequency') == 0:
                out_file.write_tecplot_file(runtime.current_timestep)

            # store training data for ML if required
            if training.should_train(runtime):
                training.commit_training_data()

            if has_converged:
                break

        # print statistics to console
        stats.timer_end()
        stats.write_statistics()

        # write residuals
        residuals.write()

        # write solution
        out_file.write_tecplot_file()
        out_file.plot_contours()
        out_file.plot_residuals()

        # ML training data
        training.write()

        # copy plots before next run overwrites them
        for plotname in ['contours.png', 'residuals.png']:
            src_plot = os.path.join(out_folder, plotname)
            if os.path.exists(src_plot):
                name, ext = os.path.splitext(plotname)
                dst_plot = os.path.join(out_folder, f'{name}_{block}{ext}')
                shutil.copy2(src_plot, dst_plot)

       # copy CSV before next run overwrites it
        src_csv = os.path.join(out_folder, 'trainingData_T.csv')
        if os.path.exists(src_csv):
            if is_validation:
                # validation block → save as validationData_T.csv
                dst_csv = os.path.join(out_folder, 'validationData_T.csv')
                shutil.copy2(src_csv, dst_csv)
            elif len(training_blocks) > 1:
                # training block → save as trainingData_blockN.csv
                dst_csv = os.path.join(out_folder, f'trainingData_{block}.csv')
                shutil.copy2(src_csv, dst_csv)
                csv_files.append(dst_csv)


        # copy plots before next run overwrites them
        for plot_name in ['contours.png', 'residuals.png']:
            src_plot = os.path.join(out_folder, plot_name)
            if os.path.exists(src_plot):
                name, ext = os.path.splitext(plot_name)
                dst_plot = os.path.join(out_folder, f'{name}_{block}{ext}')
                shutil.copy2(src_plot, dst_plot)

    #  if multiple blocks, combine all CSVs into one file
    if len(csv_files) > 1:
        combined = os.path.join(out_folder, 'trainingData_ALL.csv')
        total = 0
        with open(combined, 'w') as out:
            for i, f in enumerate(csv_files):
                for ln, line in enumerate(open(f)):
                    if ln == 0 and i == 0:
                        out.write(line)        # header once
                    elif ln > 0:
                        out.write(line)
                        total += 1
        print(f"\n  Combined {len(csv_files)} CSVs → {combined} ({total} rows)")
    params.params['boundaryCondition'] = original_block  

if __name__ == '__main__':
    run()

