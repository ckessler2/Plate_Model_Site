import numpy as np

# ----------------------------
# Model: RHS translated from MATLAB
# State y = [v_xp, v_yp, omega, theta, x, y]
# Params p = (l, m, rho_f, a, b, s)
# Aero  opt = [C_L1, C_L2, C_D0, C_D1, C_D_pi_2, C_0_CP, C_1_CP, C_2_CP, C_R, e_x]
# ----------------------------



def nondim_freely_falling_plate(t, y, p, opt):
    v_xp, v_yp, omega, theta, x_pos, y_pos = y
    l, m, rho_f, a, b, s = p

    C_L1 = opt[0]
    C_L2 = opt[1]
    C_D0 = opt[2]
    C_D1 = opt[3]
    C_D_pi_2 = opt[4]
    C_0_CP = opt[5]
    C_1_CP = opt[6]
    C_2_CP = opt[7]
    C_R = opt[8]
    e_x = opt[9]
    
    # Derived quantities
    rho_s = m / (np.pi * a * b)
    Inertia = (m * (a**2 + b**2) / (rho_f * l**4)) + 1.0 / 32.0 + e_x**2
    l_CM = e_x * l
    m_prime = 4.0 * m / (np.pi * rho_f * l * l * s)
    gamma = rho_f / (rho_s - rho_f)

    alpha = np.arctan2((v_yp - omega * l_CM), v_xp)

    alpha0 = np.deg2rad(14.0)
    delta = np.deg2rad(6.0)


    # Activation functions
    Falpha1 = (1.0 - np.tanh((np.pi - np.abs(alpha) - alpha0) / delta)) / 2.0
    Falpha2 = (1.0 - np.tanh((np.abs(alpha) - alpha0) / delta)) / 2.0
    Falpha3 = (1.0 - np.tanh((alpha - alpha0) / delta)) / 2.0
    Falpha4 = (1.0 - np.tanh((np.pi - alpha - alpha0) / delta)) / 2.0

    # Lift coefficient (piecewise like MATLAB)
    C_Lalpha1 = Falpha1 * C_L1 * np.sin(np.pi - np.abs(alpha)) + (1.0 - Falpha1) * C_L2 * np.sin(
        2.0 * (np.pi - np.abs(alpha))
    )
    C_Lalpha2 = Falpha2 * C_L1 * np.sin(np.abs(alpha)) + (1.0 - Falpha2) * C_L2 * np.sin(2.0 * np.abs(alpha))
    C_Lalpha3 = Falpha3 * C_L1 * np.sin(alpha) + (1.0 - Falpha3) * C_L2 * np.sin(2.0 * alpha)
    C_Lalpha4 = Falpha4 * C_L1 * np.sin(np.pi - alpha) + (1.0 - Falpha4) * C_L2 * np.sin(2.0 * (np.pi - alpha))

    if (-np.pi <= alpha) and (alpha <= -np.pi / 2.0):
        C_Lalpha = C_Lalpha1
    elif (-np.pi / 2.0 < alpha) and (alpha <= 0.0):
        C_Lalpha = -C_Lalpha2
    elif (0.0 < alpha) and (alpha <= np.pi / 2.0):
        C_Lalpha = C_Lalpha3
    else:
        C_Lalpha = -C_Lalpha4

    # Drag coefficient (piecewise like MATLAB; all positive combination)
    C_Dalpha1 = Falpha1 * (C_D0 + C_D1 * np.sin((np.pi - np.abs(alpha)) ** 2)) + (1.0 - Falpha1) * C_D_pi_2 * (
        np.sin(np.pi - np.abs(alpha)) ** 2
    )
    C_Dalpha2 = Falpha2 * (C_D0 + C_D1 * np.sin((np.abs(alpha)) ** 2)) + (1.0 - Falpha2) * C_D_pi_2 * (
        np.sin(np.abs(alpha)) ** 2
    )
    C_Dalpha3 = Falpha3 * (C_D0 + C_D1 * np.sin((alpha) ** 2)) + (1.0 - Falpha3) * C_D_pi_2 * (np.sin(alpha) ** 2)
    C_Dalpha4 = Falpha4 * (C_D0 + C_D1 * np.sin((np.pi - alpha) ** 2)) + (1.0 - Falpha4) * C_D_pi_2 * (
        np.sin(np.pi - alpha) ** 2
    )

    if (-np.pi <= alpha) and (alpha < -np.pi / 2.0):
        C_Dalpha = C_Dalpha1
    elif (-np.pi / 2.0 <= alpha) and (alpha < 0.0):
        C_Dalpha = C_Dalpha2
    elif (0.0 <= alpha) and (alpha < np.pi / 2.0):
        C_Dalpha = C_Dalpha3
    else:
        C_Dalpha = C_Dalpha4

    # Centre of pressure term (epsilon_alpha), piecewise like MATLAB
    l_CP_alpha_l1 = Falpha1 * (C_0_CP - C_1_CP * (np.pi - np.abs(alpha)) ** 2) + (1.0 - Falpha1) * C_2_CP * (
        1.0 - (np.pi - np.abs(alpha)) / (np.pi / 2.0)
    )
    l_CP_alpha_l2 = Falpha2 * (C_0_CP - C_1_CP * (np.abs(alpha)) ** 2) + (1.0 - Falpha2) * C_2_CP * (
        1.0 - np.abs(alpha) / (np.pi / 2.0)
    )
    l_CP_alpha_l3 = Falpha3 * (C_0_CP - C_1_CP * (alpha) ** 2) + (1.0 - Falpha3) * C_2_CP * (
        1.0 - alpha / (np.pi / 2.0)
    )
    l_CP_alpha_l4 = Falpha4 * (C_0_CP - C_1_CP * (np.pi - alpha) ** 2) + (1.0 - Falpha4) * C_2_CP * (
        1.0 - (np.pi - alpha) / (np.pi / 2.0)
    )

    if (-np.pi <= alpha) and (alpha <= -np.pi / 2.0):
        epsilon_alpha = -l_CP_alpha_l1
    elif (-np.pi / 2.0 <= alpha) and (alpha <= 0.0):
        epsilon_alpha = l_CP_alpha_l2
    elif (0.0 <= alpha) and (alpha <= np.pi / 2.0):
        epsilon_alpha = l_CP_alpha_l3
    else:
        epsilon_alpha = -l_CP_alpha_l4

    # Aerodynamic terms
    speed = np.sqrt(v_xp**2 + (v_yp - omega * e_x) ** 2)

    L_Txy = (2.0 / np.pi) * C_Lalpha * speed
    L_Tx = L_Txy * (v_yp - omega * e_x)
    L_Ty = L_Txy * (-v_xp)

    L_Rxy = (-2.0 / np.pi) * C_R * omega
    L_Rx = L_Rxy * (v_yp - omega * e_x)
    L_Ry = L_Rxy * (-v_xp)

    D_xy = (-2.0 / np.pi) * C_Dalpha * speed
    D_x = D_xy * v_xp
    D_y = D_xy * (v_yp - omega * e_x)

    # plus_minus logic
    e_xplus = (2.0 * e_x + 1.0) ** 4 + (2.0 * e_x - 1.0) ** 4
    e_xminus = (2.0 * e_x + 1.0) ** 4 - (2.0 * e_x - 1.0) ** 4
    if 2.0 * e_x <= 1.0:
        plus_minus = e_xplus + e_xminus
    else:
        plus_minus = e_xplus - e_xminus

    domegadt = (
        (-(C_D_pi_2 / (32.0 * np.pi)) * omega * np.abs(omega) * plus_minus)
        + ((L_Ty + D_y) * (epsilon_alpha - e_x))
        - gamma * e_x * np.cos(theta)
    ) / Inertia

    dv_xpdt = (1.0 / m_prime) * (
        (m_prime + 1.0) * omega * v_yp - e_x * omega**2 + (L_Tx + L_Rx) + D_x - np.sin(theta)
    )

    dv_ypdt = (1.0 / (m_prime + 1.0)) * (
        -m_prime * omega * v_xp + domegadt * e_x + (L_Ty + L_Ry) + D_y - np.cos(theta)
    )

    dthetadt = omega

    dx_dt = v_xp * np.cos(theta) - v_yp * np.sin(theta)
    dy_dt = v_xp * np.sin(theta) + v_yp * np.cos(theta)

    return [dv_xpdt, dv_ypdt, domegadt, dthetadt, dx_dt, dy_dt]