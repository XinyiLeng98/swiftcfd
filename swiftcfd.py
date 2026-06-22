import swiftcfd

from swiftcfd.enums import PrimitiveVariables as pv

def run():
    # read command line arguments
    cla_parser = swiftcfd.command_line_argument_parser()

    # check if machine learning should be trained only
    if cla_parser.arguments.train:
        print('Training machine learning model...')

        # seed every RNG that affects training so two identical configs are
        # reproducible: numpy global RNG drives the train/val split permutation
        # in the DataManager, torch global RNG drives weight init + DataLoader shuffle.
        import random
        import numpy as np
        import torch
        seed = cla_parser.arguments.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f'  Seed: {seed}')

        input_vars    = [v.strip() for v in cla_parser.arguments.input_variables.split(',')]
        output_var    = cla_parser.arguments.output_variables.strip()
        equation_type = cla_parser.arguments.equation_type

        all_vars = list(dict.fromkeys(input_vars + [output_var]))
        training_data = swiftcfd.ML_data_manager.get_training_data(','.join(all_vars))

        features_per_var = training_data[input_vars[0]]['x_train'].shape[1]
        input_size  = features_per_var * len(input_vars)
        output_size = training_data[output_var]['y_train'].shape[1]

        # create ML model
        model = swiftcfd.create_model(
            cla_parser.arguments.model,
            cla_parser.arguments.input_variables,
            output_var,
            equation_type=equation_type,
            input_size=input_size,
            output_size=output_size,
            hidden_size=cla_parser.arguments.hidden_size,
            num_layers=cla_parser.arguments.num_layers,
            dropout=cla_parser.arguments.dropout,
        )

        # train ML model
        model.train_network(
            training_data,
            epochs=cla_parser.arguments.epochs,
            batch_size=cla_parser.arguments.batch_size,
            lr=cla_parser.arguments.lr,
            hidden_size=cla_parser.arguments.hidden_size,
            num_layers=cla_parser.arguments.num_layers,
            dropout=cla_parser.arguments.dropout,
            weight_pde=cla_parser.arguments.weight_pde,
            weight_data=cla_parser.arguments.weight_data,
            patience=cla_parser.arguments.patience,
            output_dir=cla_parser.arguments.output_dir,
        )

        print('Done. Exiting solver now...')
        return

    # create parameters
    params = swiftcfd.parameters()
    params.read_from_file(cla_parser.arguments.input)

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

    # create instance of the data manager class that handles daa I/O for the ML model
    data_manager = swiftcfd.ML_data_manager(params, mesh, eqm.field_manager)

    # loop over time, check for Ctrl+C to stop and write output
    try:
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
            if data_manager.should_train(runtime):
                data_manager.commit_training_data()

            if has_converged:
                break

    except KeyboardInterrupt:
        log.clear_screen()
        print("Ctrl+C detected, stopping simulation...")
        end_of_simulation(residuals, out_file, data_manager)
    
    # normal end to simulation, write out files as requested
    stats.timer_end()
    stats.write_statistics()
    end_of_simulation(residuals, out_file, data_manager)

def end_of_simulation(residuals, out_file, data_manager):
    # write residuals
    residuals.write()

    # write solution
    out_file.write_tecplot_file()
    out_file.plot_contours()
    out_file.plot_residuals()

    # ML training data
    data_manager.write()

if __name__ == '__main__':
    run()