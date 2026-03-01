import swiftcfd

from swiftcfd.enums import PrimitiveVariables as pv

def run():
    # read command line arguments
    cla_parser = swiftcfd.command_line_argument_parser()

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

    # create machine learning training data
    training = swiftcfd.ML_training_data(params, mesh, eqm.field_manager)

    # loop over time
    while (runtime.has_not_reached_final_time()):
        # copy solution
        eqm.field_manager.update_solution()

        # print time info to console
        log.print_time_info(runtime)

        # # linearisation step through picard iterations
        # while(runtime.has_not_reached_final_picard_iteration()):
        #     # update picard solution
        #     eqm.field_manager.update_picard_solution()

        #     # solve non-linear equations (e.q. momentum equations)
        #     eqm.solve_non_linear_equations(runtime, stats)

        #     # compute picard residuals
        #     has_converged = residuals.check_picard_convergence(runtime)

        #     # print time step statistics
        #     log.print_picard_iteration(runtime, eqm.equations, residuals)

        #     if has_converged:
        #         runtime.current_picard_iteration = 0
        #         break

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

if __name__ == '__main__':
    run()