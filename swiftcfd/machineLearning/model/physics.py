import torch


class HeatResidual:

    @staticmethod
    def compute(X_batch_raw, Y_prediction_raw, training_parameters):
        # training_parameters: [N, 6] = [dx, dy, dt, rho, nu, alpha]
        dx    = training_parameters[:, 0:1]
        dy    = training_parameters[:, 1:2]
        dt    = training_parameters[:, 2:3]
        alpha = training_parameters[:, 5:6]

        # inputs
        phi_center         = X_batch_raw[:, 0:1]
        phi_east           = X_batch_raw[:, 1:2]
        phi_west           = X_batch_raw[:, 2:3]
        phi_north          = X_batch_raw[:, 3:4]
        phi_south          = X_batch_raw[:, 4:5]
        phi_center_n_minus_1 = X_batch_raw[:, 5:6]
        phi_center_n_minus_2 = X_batch_raw[:, 6:7]

        # predictions
        phi_center_n_plus_1 = Y_prediction_raw[:, 0:1]
        phi_east_n_plus_1   = Y_prediction_raw[:, 1:2]
        phi_west_n_plus_1   = Y_prediction_raw[:, 2:3]
        phi_north_n_plus_1  = Y_prediction_raw[:, 3:4]
        phi_south_n_plus_1  = Y_prediction_raw[:, 4:5]

        phi_t = 2 * (phi_center - phi_center_n_minus_1) / dt - (phi_center_n_minus_1 - phi_center_n_minus_2) / dt
        phi_xx_n       = (phi_east       - 2 * phi_center       + phi_west)       / (dx * dx)
        phi_yy_n       = (phi_north      - 2 * phi_center       + phi_south)      / (dy * dy)
        phi_xx_n_plus_1 = (phi_east_n_plus_1 - 2 * phi_center_n_plus_1 + phi_west_n_plus_1)  / (dx * dx)
        phi_yy_n_plus_1 = (phi_north_n_plus_1 - 2 * phi_center_n_plus_1 + phi_south_n_plus_1) / (dy * dy)

        residual = phi_t - (alpha / 2) * (phi_xx_n + phi_yy_n + phi_xx_n_plus_1 + phi_yy_n_plus_1)
        return residual


class NSPressureResidual:

    @staticmethod
    def compute(X_batch_raw, Y_prediction_raw, training_parameters):
        # training_parameters: [N, 6] = [dx, dy, dt, rho, nu, alpha]
        dx  = training_parameters[:, 0:1]
        dy  = training_parameters[:, 1:2]
        dt  = training_parameters[:, 2:3]
        rho = training_parameters[:, 3:4]

        # u-velocity stencil at time n  (features 0–6)
        u_center         = X_batch_raw[:, 0:1]
        u_east           = X_batch_raw[:, 1:2]
        u_west           = X_batch_raw[:, 2:3]
        u_north          = X_batch_raw[:, 3:4]
        u_south          = X_batch_raw[:, 4:5]
        u_center_n_minus_1 = X_batch_raw[:, 5:6]
        u_center_n_minus_2 = X_batch_raw[:, 6:7]

        # v-velocity stencil at time n  (features 7–13)
        v_center         = X_batch_raw[:, 7:8]
        v_east           = X_batch_raw[:, 8:9]
        v_west           = X_batch_raw[:, 9:10]
        v_north          = X_batch_raw[:, 10:11]
        v_south          = X_batch_raw[:, 11:12]
        v_center_n_minus_1 = X_batch_raw[:, 12:13]
        v_center_n_minus_2 = X_batch_raw[:, 13:14]

        # pressure stencil at time n  (features 14–20)
        p_center         = X_batch_raw[:, 14:15]
        p_east           = X_batch_raw[:, 15:16]
        p_west           = X_batch_raw[:, 16:17]
        p_north          = X_batch_raw[:, 17:18]
        p_south          = X_batch_raw[:, 18:19]
        p_center_n_minus_1 = X_batch_raw[:, 19:20]
        p_center_n_minus_2 = X_batch_raw[:, 20:21]

        # predictions
        p_center_n_plus_1 = Y_prediction_raw[:, 0:1]
        p_east_n_plus_1   = Y_prediction_raw[:, 1:2]
        p_west_n_plus_1   = Y_prediction_raw[:, 2:3]
        p_north_n_plus_1  = Y_prediction_raw[:, 3:4]
        p_south_n_plus_1  = Y_prediction_raw[:, 4:5]

        # pressure Poisson: ∇²p^{n+1} = (rho/dt) * ∇·u^n
        p_xx_n_plus_1 = (p_east_n_plus_1  - 2 * p_center_n_plus_1 + p_west_n_plus_1)  / (dx * dx)
        p_yy_n_plus_1 = (p_north_n_plus_1 - 2 * p_center_n_plus_1 + p_south_n_plus_1) / (dy * dy)
        div_u_n       = (u_east - u_west) / (2 * dx) + (v_north - v_south) / (2 * dy)

        residual = (p_xx_n_plus_1 + p_yy_n_plus_1) - (rho / dt) * div_u_n
        return residual
