import numpy as np

kTolerance = .5

def solve_adaptive(matrix_c : np.ndarray, matrix_a : np.ndarray, matrix_lb : np.ndarray,
                   matrix_ub : np.ndarray, matrix_ld : np.ndarray, matrix_ud : np.ndarray):

    x = (matrix_ld + matrix_ud) / 2

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

        print("RUNNING ON ITERATION: ", iter)
        iter+=1

        basis_i.sort()
        basis_j.sort()
        basis_i_residual.sort()
        basis_j_residual.sort()

        print(basis_i, basis_i_residual, basis_j_residual, basis_j)

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
        print(z, matrix_ub, matrix_lb)
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

        print(basis_i, basis_j)
        if len(broken_idxes_i) + len(broken_idxes_j) == 0:
            print("ret in optimum criterion")
            return x

        # STEP 3: calculate beta
        
        # u consists of basis_i
        # delta constits of whole J

        beta = 0
        print(delta)
        print(matrix_ud)
        print(x)
        print(delta[0] * (x[0] - matrix_ud[0]))
        print(delta[1] * (x[1] - matrix_ud[1]))
        for jdx in range(delta.shape[0]):
            if delta[jdx] > 0:
                beta += delta[jdx] * (x[jdx] - matrix_ld[jdx])
            else:
                beta += delta[jdx] * (x[jdx] - matrix_ud[jdx])
        
        for idx in basis_i:
            if u[idx] < 0:
                beta += u[idx] * omega_lower[idx]
            else:
                beta += u[idx] * omega_upper[idx]
        
        if beta <= kTolerance:
            print (beta)
            print("ret in first beta check")
            return x
        
        # STEP 4: calculate l

        l = np.zeros(x.shape[0])

        for jdx in basis_j_residual:
            if delta[jdx] < 0:
                l[jdx] = matrix_ud[jdx] - x[jdx]
            elif delta[jdx] > 0:
                l[jdx] = matrix_ld[jdx] - x[jdx]
            else:
                l[jdx] = 0
        
        omega = np.zeros(matrix_a.shape[0])
        print("opora dim:", opora_dim)
        print ("u:", u)
        
        for idx in basis_i:
            if u[idx] < 0:
                omega[idx] = omega_lower[idx]
            elif u[idx] > 0:
                print("omega: ", omega, "omega_upper: ", omega_upper)
                omega[idx] = omega_upper[idx]
            else:
                omega[idx] = 0
        if opora_dim > 0:
            l[basis_j] = a_op_inv @ omega[basis_j] - a_op_inv @ matrix_a[basis_i, :][:, basis_j_residual] @ l[basis_j_residual]

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
        print(matrix_a[basis_i_residual])
        print(l)
        rows_dir = matrix_a[basis_i_residual] @ l.transpose()
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
                if val < theta_j:
                    theta_j = val
                    theta_j_idx = idx
            elif rows_dir[idx] > 0:
                val = omega_upper[idx] / rows_dir[idx]
                if val < theta_j:
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

        last_x = x.copy()
        x = x + theta * l
        beta_bar = (1 - theta) * beta

        print(x, matrix_c.transpose() * x)
        print("beta_bar:", beta_bar)

        if beta_bar <= kTolerance:
            print("ret in second beta check")
            return x

        # STEP 7: compute ksi and alpha

        ksi_j = np.zeros(matrix_a.shape[1])
        ksi_i = np.zeros(matrix_a.shape[0])

        alpha = 0
        print("bad index type:", bad_index)
        if bad_index_type == 0:
            sign = 1 if x[bad_index] == matrix_ld[bad_index] else -1
            if opora_dim > 0:
                matrix_j_res = a_op_inv @ matrix_a[basis_i, :][:, basis_j_residual]
                matrix_i = -a_op_inv
                ksi_j[basis_j_residual] = sign * matrix_j_res[bad_index, :]
                ksi_i[basis_i] = sign * matrix_i[bad_index, :]

            if sign == 1:
                alpha = x[bad_index] + l[bad_index] - matrix_ld[bad_index]
            else:
                alpha = matrix_ud[bad_index] - x[bad_index] - l[bad_index]
        else:
            print("matrix_a @ x:", matrix_a[bad_index, :] @ x)
            sign = 1 if matrix_a[bad_index, :] @ x == matrix_ub[bad_index] else -1
            if opora_dim > 0:
                matrix_j_res = matrix_a[basis_i_residual, :][:, basis_j_residual] - matrix_a[basis_i_residual, :][:, basis_j] @ a_op_inv @ matrix_a[basis_i, :][:, basis_j_residual]
                matrix_i = matrix_a[basis_i_residual, :][:, basis_j] @ a_op_inv
                ksi_j[basis_j_residual] = sign * matrix_j_res[bad_index, :]
                ksi_i[basis_i] = sign * matrix_i[bad_index, :]
            else:
                matrix_j_res = matrix_a[basis_i_residual, :][:, basis_j_residual]
                ksi_j[basis_j_residual] = sign * matrix_j_res[bad_index, :]
            
            if sign == 1:
                alpha = matrix_ub[bad_index] - matrix_a[bad_index] @ (x + l)
            else:
                alpha = matrix_a[bad_index] @ (x + l) - matrix_lb[bad_index]
        print("alpha right after:", alpha)
        
        ksi_j = ksi_j.transpose()
        ksi_i = ksi_i.transpose()

        # TODO: step 11 in algo, u can delete all the lower code!

        # setting up kappa, delta, upper_d, lower_d...

        delta_j = np.zeros(matrix_a.shape[1])
        delta_i = np.zeros(matrix_a.shape[0])

        delta_j[basis_j_residual] = delta[basis_j_residual]
        if opora_dim > 0:
            delta_i[basis_i] = -u[basis_i]

        kappa_j = np.zeros(matrix_a.shape[1])
        kappa_i = np.zeros(matrix_a.shape[0])

        kappa_j[basis_j_residual] = x[basis_j_residual] + l[basis_j_residual]
        if opora_dim > 0:
            kappa_i[basis_i] = matrix_a[basis_i] @ (x + l)

        # STEP 8: iterate over alpha and compute sigma!

        sigma = 0
        bad_index_2 = -1
        bad_index_2_type = -1 # 0 in j, 1 in i

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
            return x

        # bad_index_type : 0 - from J_op, 1 - from I_n
        # bad_index_type_2 : 0 p from J_n, 1 - from I_op 
        
        if bad_index_type == 1 and bad_index_2_type == 1:
            basis_i = np.append(basis_i, bad_index)
            basis_i = np.delete(basis_i, bad_index_2)

            basis_i_residual = np.append(basis_i_residual, bad_index_2)
            basis_i_residual = np.delete(basis_i_residual, bad_index)
        elif bad_index_type == 1 and bad_index_2_type == 0:
            basis_i = np.append(basis_i, bad_index)
            basis_j = np.append(basis_j, bad_index_2)

            basis_i_residual = np.delete(basis_i_residual, bad_index)
            basis_j_residual = np.delete(basis_j_residual, bad_index_2)
        elif bad_index_type == 0 and bad_index_2_type == 1:
            basis_i = np.delete(basis_i, bad_index_2)
            basis_j = np.delete(basis_j, bad_index)

            basis_i_residual = np.append(basis_i_residual, bad_index_2)
            basis_j_residual = np.append(basis_j_residual, bad_index)
        else:
            basis_j = np.append(basis_j, bad_index_2)
            basis_j = np.delete(basis_j, bad_index)

            basis_j_residual = np.delete(basis_j_residual, bad_index_2)
            basis_j_residual = np.append(basis_j_residual, bad_index)

if __name__ == "__main__":
    matrix_c = np.array([-2, -1])
    matrix_c = matrix_c.transpose()
    
    matrix_a = np.array([[1, 1], [1, 3]])
    matrix_lb = np.array([6, 12]).transpose()
    matrix_ub = np.array([1000, 1000]).transpose()
    matrix_ld = np.array([0, 0]).transpose()
    matrix_ud = np.array([100, 100]).transpose()

    solution = solve_adaptive(matrix_c, matrix_a, matrix_lb, matrix_ub, matrix_ld, matrix_ud)
    print("solution is\n")
    print(solution)
    print(matrix_c @ solution)
    print(matrix_a @ solution)