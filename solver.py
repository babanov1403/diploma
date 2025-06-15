import numpy as np

kTolerance = .5

def solve_adaptive(matrix_c : np.ndarray, matrix_a : np.ndarray, matrix_lb : np.ndarray,
                   matrix_ub : np.ndarray, matrix_ld : np.ndarray, matrix_ud : np.ndarray, x = None):
    if x is None:
        x = (matrix_ld + matrix_ud) / 2

    # x = np.array([6, 10, 9, 10, 9, 10, 10, 6, 2, 0])
    # x = np.array([1, 2])

    basis_i = np.array([], dtype = np.int64)
    basis_j = np.array([], dtype = np.int64)

    basis_i_residual = np.array([], dtype = np.int64) # from 0 to matrix_a.rows
    basis_j_residual = np.array([], dtype = np.int64) # from 0 to matrix_a.cols
    for idx in range(matrix_a.shape[0]):
        basis_i_residual = np.append(basis_i_residual, idx)

    for jdx in range(matrix_a.shape[1]):
        basis_j_residual = np.append(basis_j_residual, jdx)

    iter = 0

    while True:
        print("================================")
        print("RUNNING ON ITERATION: ", iter)
        iter+=1

        basis_i.sort()
        basis_j.sort()
        basis_i_residual.sort()
        basis_j_residual.sort()
        print("i", "i_res", "j", "j_res")
        print(basis_i, basis_i_residual, basis_j, basis_j_residual)

        opora_dim = basis_i.shape[0]

        if opora_dim > 0:
            a_op = matrix_a[basis_i, :][:, basis_j]
            a_op_inv = np.linalg.inv(a_op)
        omega_lower = matrix_lb - matrix_a @ x.transpose()
        omega_upper = matrix_ub - matrix_a @ x.transpose()

        # STEP 1: compute u and delta! if we know basis_i and basis_j, otherwise it can be zeros

        u = np.zeros(matrix_a.shape[0])
        delta = np.zeros(matrix_a.shape[1])
        u = u.transpose()
        delta = delta.transpose()
        if opora_dim > 0:
            u[basis_i] = a_op_inv.transpose() @ matrix_c[basis_j]
            a_iop_jn = matrix_a[basis_i].transpose()
            delta = a_iop_jn @ u[basis_i] - matrix_c
        else:
            delta = -matrix_c

        # STEP 2: optimum criterion!

        z = matrix_a @ x
        broken_idxes_i = []
        broken_idxes_j = []
        for idx in basis_i:
            if u[idx] <= 0 and z[idx] == matrix_lb[idx]:
                continue
            if u[idx] >= 0 and z[idx] == matrix_ub[idx]:
                continue
            if u[idx] == 0 and z[idx] < matrix_ub[idx] and z[idx] > matrix_lb[idx]:
                continue
            broken_idxes_i.append(idx)

        for idx in basis_j_residual:
            if delta[idx] <= 0 and x[idx] == matrix_ud[idx]:
                continue
            if delta[idx] >= 0 and x[idx] == matrix_ld[idx]:
                continue
            if delta[idx] == 0 and x[idx] < matrix_ud[idx] and x[idx] > matrix_ld[idx]:
                continue
            broken_idxes_j.append(idx)

        if len(broken_idxes_i) + len(broken_idxes_j) + 1 == 0:
            print("ret in optimum criterion")
            return x

        # STEP 3: calculate beta
        
        # u consists of basis_i
        # delta constits of whole J

        beta = 0
        for jdx in basis_j_residual:
            if delta[jdx] > 0:
                beta += delta[jdx] * (x[jdx] - matrix_ld[jdx])
            else:
                beta += delta[jdx] * (x[jdx] - matrix_ud[jdx])
        
        for idx in basis_i:
            if u[idx] < 0:
                beta += u[idx] * omega_lower[idx]
            else:
                beta += u[idx] * omega_upper[idx]
        print(beta)
        if beta <= kTolerance:
            print("ret in first beta check")
            return x
        
        # STEP 4: calculate l

        l = np.zeros(x.shape[0], dtype=np.float64)

        print("delta:\n")
        print(delta)

        for jdx in basis_j_residual:
            if delta[jdx] < 0:
                l[jdx] = matrix_ud[jdx] - x[jdx]
            elif delta[jdx] > 0:
                l[jdx] = matrix_ld[jdx] - x[jdx]
            else:
                l[jdx] = 0
        
        omega = np.zeros(matrix_a.shape[0])
        
        for idx in basis_i:
            if u[idx] < 0:
                omega[idx] = omega_lower[idx]
            elif u[idx] > 0:
                omega[idx] = omega_upper[idx]
            else:
                omega[idx] = 0
        if opora_dim > 0:
            print(omega)
            l[basis_j] = a_op_inv @ omega[basis_i] - a_op_inv @ matrix_a[basis_i, :][:, basis_j_residual] @ l[basis_j_residual]

        # STEP 5: computing theta!!!

        theta = 1
        bad_index = -1
        bad_index_type = -1 # 0 - from basis_j, 1 - from basis_i_residual

        theta_j = np.inf
        theta_j_idx = -1

        theta_i = np.inf
        theta_i_idx = -1

        for jdx in basis_j:
            if l[jdx] < 0:
                val = (matrix_ld[jdx] - x[jdx]) / l[jdx]
                if val < theta_j:
                    theta_j = val
                    theta_j_idx = jdx
            elif l[jdx] > 0:
                val = (matrix_ud[jdx] - x[jdx]) / l[jdx]
                if val < theta_j:
                    theta_j = val
                    theta_j_idx = jdx

        print(matrix_a)
        print("l:", l)
        print("x:", x)
        print(basis_i, basis_i_residual, basis_j, basis_j_residual)
        rows_dir = matrix_a @ l.transpose()
        print("rows dir:")
        print(rows_dir)
        print("omegas:")
        print(omega_upper)
        print(omega_lower)
        print("x:")
        print(x)

        for idx in basis_i_residual:
            if rows_dir[idx] < 0:
                val = omega_lower[idx] / rows_dir[idx]
                if val < theta_i:
                    theta_i = val
                    theta_i_idx = idx
            elif rows_dir[idx] > 0:
                val = omega_upper[idx] / rows_dir[idx]
                if val < theta_i:
                    theta_i = val
                    theta_i_idx = idx
        
        if theta > theta_i:
            theta = theta_i
            bad_index = theta_i_idx
            bad_index_type = 1
        
        if theta > theta_j:
            theta = theta_j
            bad_index = theta_j_idx
            bad_index_type = 0

        # STEP 6: compute x_bar, beta_bar
        print("theta:", theta)
        print(theta_i, theta_j)

        x = x + theta * l
        beta_bar = (1 - theta) * beta

        print("beta_bar:", beta_bar)

        if beta_bar <= kTolerance:
            print("ret in second beta check")
            return x

        # STEP 7: compute ksi and alpha

        ksi_j = np.zeros(matrix_a.shape[1], dtype=np.float64)
        ksi_i = np.zeros(matrix_a.shape[0], dtype=np.float64)

        matrix_j_res = np.zeros_like(matrix_a, dtype=np.float64)
        matrix_i = np.zeros_like(matrix_a, dtype=np.float64)

        alpha = 0
        print(basis_i, basis_j, basis_i_residual, basis_j_residual)
        print("bad index type:", bad_index, bad_index_type)
        # A = A(I, J)
        if bad_index_type == 0:
            sign = 1 if np.allclose(x[bad_index], matrix_ld[bad_index]) else -1
            tmp_bad_index = np.where(basis_j == bad_index)[0]
            print(sign)
            if opora_dim > 0:
                matrix_j_res[np.ix_(basis_i, basis_j_residual)] = a_op_inv @ matrix_a[basis_i, :][:, basis_j_residual]
                matrix_i[np.ix_(basis_i, basis_j)] = -a_op_inv
                print(ksi_j[basis_j_residual])
                print(matrix_j_res.shape)
                print (matrix_j_res)
                ksi_j[basis_j_residual] = sign * matrix_j_res[tmp_bad_index][0][basis_j_residual]
                ksi_i[basis_i] = sign * matrix_i[tmp_bad_index][0][basis_i]
            print(matrix_j_res)
            if sign == 1:
                alpha = x[bad_index] + l[bad_index] - matrix_ld[bad_index]
            else:
                alpha = matrix_ud[bad_index] - x[bad_index] - l[bad_index]
        else:
            sign = 1 if np.allclose(matrix_a[bad_index, :] @ x, matrix_ub[bad_index]) else -1
            tmp_bad_index = np.where(basis_i_residual == bad_index)[0]
            print(matrix_a[bad_index, :] @ x, matrix_ub[bad_index])
            if opora_dim > 0:
                matrix_j_res[np.ix_(basis_i_residual, basis_j_residual)] = matrix_a[basis_i_residual, :][:, basis_j_residual] - matrix_a[basis_i_residual, :][:, basis_j] @ a_op_inv @ matrix_a[basis_i, :][:, basis_j_residual]
                matrix_i[np.ix_(basis_i_residual, basis_j)] = matrix_a[basis_i_residual, :][:, basis_j] @ a_op_inv
                ksi_j[basis_j_residual] = sign * matrix_j_res[bad_index][basis_j_residual]
                ksi_i[basis_i] = sign * matrix_i[bad_index][basis_j]
            else:
                matrix_j_res = matrix_a[basis_i_residual, :][:, basis_j_residual]
                print(matrix_j_res)
                print(matrix_j_res[bad_index])
                print(bad_index)
                ksi_j[basis_j_residual] = sign * matrix_j_res[bad_index, :][basis_j_residual]
            
            print(matrix_j_res)
            print(ksi_j)
            
            if sign == 1:
                alpha = matrix_ub[bad_index] - matrix_a[bad_index] @ (x + l)
            else:
                alpha = matrix_a[bad_index] @ (x + l) - matrix_lb[bad_index]
        print("alpha right after:", alpha)
        
        ksi_j = ksi_j.transpose()
        ksi_i = ksi_i.transpose()

        # TODO: step 11 in algo, u can delete all the lower code!

        # setting up kappa, delta, upper_d, lower_d...

        delta_j = np.zeros(matrix_a.shape[1], dtype=np.float64)
        delta_i = np.zeros(matrix_a.shape[0], dtype=np.float64)

        delta_j[basis_j_residual] = delta[basis_j_residual]
        if opora_dim > 0:
            delta_i[basis_i] = -u[basis_i]

        kappa_j = np.zeros(matrix_a.shape[1], dtype=np.float64)
        kappa_i = np.zeros(matrix_a.shape[0], dtype=np.float64)

        kappa_j[basis_j_residual] = x[basis_j_residual] + l[basis_j_residual]
        if opora_dim > 0:
            kappa_i[basis_i] = matrix_a[basis_i] @ (x + l)

        # STEP 8: iterate over alpha and compute sigma!

        sigma = 0
        bad_index_2 = -1
        bad_index_2_type = -1 # 0 in j, 1 in i

        print(ksi_j, ksi_i, kappa_j, kappa_i)

        while True:
            sigma_tmp = 1e8
            for idx in range(ksi_j.shape[0]):
                if ksi_j[idx] * delta_j[idx] < 0:
                    val = - delta_j[idx] / ksi_j[idx]
                    if val < sigma_tmp:
                        sigma_tmp = val
                        bad_index_2 = idx
                        bad_index_2_type = 0
            
            for idx in range(ksi_i.shape[0]):
                if ksi_i[idx] * delta_i[idx] < 0:
                    val = -delta_i[idx] / ksi_i[idx]
                    if val < sigma_tmp:
                        sigma_tmp = val
                        bad_index_2 = idx
                        bad_index_2_type = 1

            # if its true, then all of delta are zeros (i think)
            # because we need to do step delta = delta + sigma * ksi, and one of the delta will be zero eventually!
            if sigma_tmp > 1e7:
                break
            for idx in range(ksi_j.shape[0]):
                delta_j[idx] += sigma_tmp * ksi_j[idx]
                if abs(delta_j[idx]) < 1e-6:
                    if ksi_j[idx] < 0:
                        alpha += ksi_j[idx] * (kappa_j[idx] - matrix_ud[idx])
                    else:
                        alpha += ksi_j[idx] * (kappa_j[idx] - matrix_ld[idx])


            for idx in range(ksi_i.shape[0]):
                delta_i[idx] += sigma_tmp * ksi_i[idx]
                if abs(delta_i[idx]) < 1e-6:
                    if ksi_i[idx] < 0:
                        alpha += ksi_i[idx] * (kappa_i[idx] - matrix_ub[idx])
                    else:
                        alpha += ksi_i[idx] * (kappa_i[idx] - matrix_lb[idx])
            
            sigma += sigma_tmp

            print("alpha:", alpha)

            if alpha >= 0:
                break
        
        # STEP 9: four variants

        if bad_index_2 == -1:
            print('bad_index_2:', bad_index_2)
            continue

        # bad_index_type : 0 - from J_op, 1 - from I_n
        # bad_index_type_2 : 0 p from J_n, 1 - from I_op 

        print(basis_i, basis_j, basis_i_residual, basis_j_residual)

        print(bad_index, bad_index_2)
        print(bad_index_type, bad_index_2_type)
        
        if bad_index_type == 1 and bad_index_2_type == 1:
            basis_i = np.append(basis_i, bad_index)
            basis_i = basis_i[basis_i != bad_index_2]

            basis_i_residual = np.append(basis_i_residual, bad_index_2)
            basis_i_residual = basis_i_residual[basis_i_residual != bad_index]
        elif bad_index_type == 1 and bad_index_2_type == 0:
            basis_i = np.append(basis_i, bad_index)
            basis_j = np.append(basis_j, bad_index_2)

            
            basis_i_residual = basis_i_residual[basis_i_residual != bad_index]
            basis_j_residual = basis_j_residual[basis_j_residual != bad_index_2]
        elif bad_index_type == 0 and bad_index_2_type == 1:
            basis_i = basis_i[basis_i != bad_index_2]
            basis_j = basis_j[basis_j != bad_index]

            basis_i_residual = np.append(basis_i_residual, bad_index_2)
            basis_j_residual = np.append(basis_j_residual, bad_index)
        else:
            basis_j = np.append(basis_j, bad_index_2)
            basis_j = basis_j[basis_j != bad_index]

            basis_j_residual = basis_j_residual[basis_j_residual != bad_index_2]
            basis_j_residual = np.append(basis_j_residual, bad_index)

if __name__ == "__main__":
    # matrix_c = np.array([2, 3])
    # matrix_c = matrix_c.transpose()
    
    # matrix_a = np.array([[2, 5], [1, 1]])
    # matrix_lb = np.array([10, 3]).transpose()
    # matrix_ub = np.array([10, 3]).transpose()
    # matrix_ld = np.array([0, 0]).transpose()
    # matrix_ud = np.array([10, 10]).transpose()

    # solution = solve_adaptive(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud)
    # print("solution is\n")
    # print(solution)

    matrix_c = np.array([2, 5, 0, 0, 0])
    matrix_c = matrix_c.transpose()
    
    matrix_a = np.array([[0, 1, 1, 0, 0], [1, 4, 0, 1, 0], [1, 1, 0, 0, 1]])
    matrix_lb = np.array([-100, -100, -100]).transpose()
    matrix_ub = np.array([7, 29, 11]).transpose()
    matrix_ld = np.array([0, 0, 0, 0, 0]).transpose()
    matrix_ud = np.array([100, 100, 100, 100, 100]).transpose()

    solution = solve_adaptive(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud)
    print("solution is\n")
    print(solution)
    print(matrix_a @ solution)
    print(matrix_c.transpose() @ solution)

    # matrix_c = np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,])
    # matrix_c = matrix_c.transpose()
    
    # matrix_a = np.array([[0.05122510239, -0.0785930863, 0.1024886449, -0.1014611958, 0.05270966528, 0.05208317585, -0.2215331957, 0.5058842243, -0.9656126594, 1.754174755, ],
    #                      [0.09281285026, -0.1455465708, 0.195032618, -0.2262270293, 0.1998878894, -0.03075050927, -0.3870405994, 1.108915758, -2.129797237, 3.181471988,],
    #                      [0.1056268387, -0.1868267921, 0.278314929, -0.3421464972, 0.3126629425, -0.09920806516, -0.4186780639, 1.387792617, -2.907709477, 4.986782492, ],
    #                      [0.002239931401, -0.02179522895, 0.03908560243, -0.05101718315, 0.07248127277, -0.1143556508, 0.1393765937, -0.07665132908, -0.03246616959, -0.4379510637,],
    #                      [-0.01978755747, 0.00941389334, 0.03037143911, -0.1057756223, 0.1942489587, -0.2690003401, 0.3519698635, -0.4904998978, 0.5638793848, 0.2779376797, ],
    #                      [-0.04085427791, 0.03016455026, 0.005159948216, -0.08393876944, 0.2345873246, -0.4705323305, 0.7441648037, -0.9279663192, 0.8707036225, -0.6129519725, ]])
    # matrix_lb = np.array([0.001808905247, 0.01414121308, 0.01932218309, -0.01069983326, -0.006767646502, -0.003292886014, ]).transpose()
    # matrix_lb -= 0.5
    # matrix_ub = np.array([0.001808905247, 0.01414121308, 0.01932218309, -0.01069983326, -0.006767646502, -0.003292886014,]).transpose()
    # matrix_ub += 0.5
    # matrix_ld = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]).transpose()
    # matrix_ud = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10, 10]).transpose()

    # solution = solve_adaptive(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud)
    # print("solution is\n")
    # print(solution)
    # print("origin solution:\n")
    # x_or = np.array([6.96856, 10, 9.67614, 10, 9.77465, 10, 10, 6.31539, 1.9775, 0.198165])
    # print(x_or)
    # print(matrix_a @ x_or)
    # print(matrix_c.transpose() @ solution)
    # print(matrix_a @ solution)
    # print(matrix_lb)
    # print(matrix_ub)