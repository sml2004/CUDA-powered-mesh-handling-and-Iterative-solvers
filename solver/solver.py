from elem import *
from shell import *
import torch



#################################################################################################
#########################################   SOLVER   ############################################
#################################################################################################

def static_structure_solver(coords, force, fixed, c3d4=None, c3d6=None, c3d8=None, s3=None, s4=None, material=None, u_init=None, tol=1e-10, max_iter=1000, device="cuda:0", dtype=torch.float64, eps=1e-30):
    """
    Args:
        coords (torch.Tensor): Nodal coordinates [N, 3]
        force (torch.Tensor): External force vector [N, 6]
        fixed (torch.Tensor): Indices of fixed nodes [n_fixed]
        u_init (torch.Tensor, optional): Initial guess for displacement [N, 6]
        tol (float, optional): Convergence tolerance
        max_iter (int, optional): Maximum number of iterations
        device (str, optional): "cuda:X" or "cpu"
        dtype (torch.dtype, optional): Floating precision
        eps (float, optional): Small epsilon to avoid division by zero
    
    Returns:
        torch.Tensor: Displacement vector u [N, 6]
    """

    coords = coords.to(device=device, dtype=dtype)
    force = force.to(device=device, dtype=dtype)
    fixed = fixed.to(device=device)

    N = coords.shape[0]

    if c3d4 is not None:
        c3d4 = c3d4.to(device)
        K_c3d4 = compute_c3d4_K_matrix(coords, c3d4, material['E'], material["nu"], device=device, dtype=dtype)
    if c3d8 is not None:
        c3d8 = c3d8.to(device)
        K_c3d8 = compute_c3d8_K_matrix(coords, c3d8, material['E'], material["nu"], device=device, dtype=dtype)
    if c3d6 is not None:
        c3d6 = c3d6.to(device)
        K_c3d6 = compute_c3d6_K_matrix(coords, c3d6, material['E'], material["nu"], device=device, dtype=dtype)
    if s3 is not None:
        s3 = s3.to(device)
        K_s3 = compute_s3_K_matrix(coords, s3, material['membrane'], material['bending'], device=device, dtype=dtype)
        unit_s3 = compute_s3_local_unitvector(coords, s3, device=device)
    if s4 is not None:
        s4 = s4.to(device)
        K_s4 = compute_s4_K_matrix(coords, s4, material['membrane'], material['bending'], device=device, dtype=dtype)
        unit_s4 = compute_s4_local_unitvector(coords, s4, device=device)
    
    print("Preprocessing done.")

    if u_init is None:
        u = torch.zeros((N, 6), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    
    u[fixed] = 0.0

    Ap = torch.zeros((N, 6), device=device, dtype=dtype)
    if c3d4 is not None:
        Ap[:,:3] += compute_nodal_forces(K_c3d4, c3d4, u[:,:3], device=device, dtype=dtype)
    if c3d8 is not None:
        Ap[:,:3] += compute_nodal_forces(K_c3d8, c3d8, u[:,:3], device=device, dtype=dtype)
    if c3d6 is not None:
        Ap[:,:3] += compute_nodal_forces(K_c3d6, c3d6, u[:,:3], device=device, dtype=dtype)
    if s3 is not None:
        Ap += compute_shell_nodal_forces(K_s3, s3, u, unit_s3, device=device, dtype=dtype)
    if s4 is not None:
        Ap += compute_shell_nodal_forces(K_s4, s4, u, unit_s4, device=device, dtype=dtype)
    r = force - Ap

    r[fixed] = 0.0

    p = r.clone()

    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        if i >0:
            print("current iteration: ", i+1, "/", max_iter, "current residual:", human_readable_number(int(torch.sqrt(rs_new))), end="\r")
        Ap = torch.zeros((N, 6), device=device, dtype=dtype)
        if c3d4 is not None:
            Ap[:,:3] += compute_nodal_forces(K_c3d4, c3d4, p[:,:3], device=device, dtype=dtype)
        if c3d8 is not None:
            Ap[:,:3] += compute_nodal_forces(K_c3d8, c3d8, p[:,:3], device=device, dtype=dtype)
        if c3d6 is not None:
            Ap[:,:3] += compute_nodal_forces(K_c3d6, c3d6, p[:,:3], device=device, dtype=dtype)
        if s3 is not None:
            Ap += compute_shell_nodal_forces(K_s3, s3, p, unit_s3, device=device, dtype=dtype)
        if s4 is not None:
            Ap += compute_shell_nodal_forces(K_s4, s4, p, unit_s4, device=device, dtype=dtype)
        pAp = torch.sum(p * Ap)

        if pAp.abs() < eps or pAp < 0.0:
            print("\n",f"Terminating early at iteration {i+1}: "f"p^T K p = {pAp.item():.3e}, which is invalid for SPD systems.")
            break

        alpha = rs_old / (pAp + eps)

        if torch.isnan(alpha) or torch.isinf(alpha):
            print("\n",f"Terminating at iteration {i+1}: alpha is NaN or Inf.")
            break

        u += alpha * p

        u[fixed] = 0.0

        r -= alpha * Ap

        r[fixed] = 0.0

        rs_new = torch.sum(r * r)

        if torch.sqrt(rs_new) < tol:
            print("\n",f"Converged after {i+1} iterations. Residual norm: {rs_new.item():.3e}")
            break

        beta = rs_new / (rs_old + eps)

        if torch.isnan(beta) or torch.isinf(beta):
            print("\n",f"Terminating at iteration {i+1}: beta is NaN or Inf.")
            break

        p = r + beta * p

        p[fixed] = 0.0

        rs_old = rs_new

    else:
        print("\n","CG did not converge within the maximum number of iterations.")

    return u




#################################################################################################
#######################################   CG solver   ###########################################
#################################################################################################

def stable_conjugate_gradient_solver(K,elements, F, rbe2, u_init=None, tol=1e-10, max_iter=1000, device="cuda:0", dtype=torch.float64, eps=1e-30):
    """
    Args:
        K (torch.Tensor): Local Stiffness Matrices of shape [M, dofs, dofs].
        elements (torch.Tensor): Connectivity [M, node_per_element].
        F (torch.Tensor): External load vector [N, 3].
        rbe2 (torch.Tensor): Indices of constrained (fixed) nodes, shape [n_fixed].
        u_init (torch.Tensor, optional): Initial guess for displacement [N, 3]. Defaults to zeros.
        tol (float, optional): Convergence tolerance. Default: 1e-10.
        max_iter (int, optional): Maximum number of CG iterations. Default: 1000.
        device (str, optional): "cuda:X" or "cpu". Default: "cuda:0".
        dtype (torch.dtype, optional): Floating precision. Default: torch.float64.
        eps (float, optional): Small epsilon to avoid division by zero. Default: 1e-30.

    Returns:
        torch.Tensor: Displacement vector u of shape [N, 3].
    """

    K = K.to(device=device, dtype=dtype)
    elements = elements.to(device=device)

    N = F.shape[0] 
    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)

    u[rbe2] = 0.0

    Ku_init = compute_nodal_forces(K, elements, u, device=device, dtype=dtype)
    r = F - Ku_init

    r[rbe2] = 0.0

    p = r.clone()

    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        Ap = compute_nodal_forces(K, elements, p, device=device, dtype=dtype)

        pAp = torch.sum(p * Ap)

        if pAp.abs() < eps or pAp < 0.0:
            print(
                f"Terminating early at iteration {i+1}: "
                f"p^T K p = {pAp.item():.3e}, which is invalid for SPD systems."
            )
            break

        alpha = rs_old / (pAp + eps)

        if torch.isnan(alpha) or torch.isinf(alpha):
            print(f"Terminating at iteration {i+1}: alpha is NaN or Inf.")
            break

        u += alpha * p

        u[rbe2] = 0.0

        r -= alpha * Ap

        r[rbe2] = 0.0

        rs_new = torch.sum(r * r)

        if torch.sqrt(rs_new) < tol:
            print(f"Converged after {i+1} iterations. Residual norm: {rs_new.item():.3e}")
            break

        beta = rs_new / (rs_old + eps)

        if torch.isnan(beta) or torch.isinf(beta):
            print(f"Terminating at iteration {i+1}: beta is NaN or Inf.")
            break

        p = r + beta * p

        p[rbe2] = 0.0

        rs_old = rs_new

    else:
        print("CG did not converge within the maximum number of iterations.")

    return u

def final_solver(K,elements, F, rbe2, u_init=None, tol=1e-10, max_iter=1000, device="cuda:0", dtype=torch.float64, eps=1e-30):

    K = K.to(device=device, dtype=dtype)  
    elements = elements.to(device=device)       
    F = F.to(device=device, dtype=dtype)       

    N = F.shape[0]
    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype, requires_grad=True)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    mask = torch.ones_like(u)
    mask[rbe2] = 0.0 

    u = u * mask

    Ku_init = compute_nodal_forces(K, elements, u, device=device, dtype=dtype)
    r = F - Ku_init
    # Zero out constrained DOFs
    r = r * mask

    p = r.clone()
    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        Ap = compute_nodal_forces(K, elements, p, device=device, dtype=dtype)
        pAp = torch.sum(p * Ap)

        if pAp.abs() < eps or pAp < 0.0:
            print(
                f"Terminating early at iteration {i+1}: "
                f"p^T K p = {pAp.item():.3e}, which is invalid for SPD systems."
            )
            break

        alpha = rs_old / (pAp + eps)

        if torch.isnan(alpha) or torch.isinf(alpha):
            print(f"Terminating at iteration {i+1}: alpha is NaN or Inf.")
            break

        # Out-of-place update for u
        u = u + alpha * p
        # Enforce constraint
        u = u * mask

        r = r - alpha * Ap
        r = r * mask

        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            print(f"Converged after {i+1} iterations. Residual norm: {rs_new.item():.3e}")
            break

        beta = rs_new / (rs_old + eps)
        if torch.isnan(beta) or torch.isinf(beta):
            print(f"Terminating at iteration {i+1}: beta is NaN or Inf.")
            break

        p = r + beta * p
        p = p * mask

        rs_old = rs_new

    return u

def stable_conjugate_gradient_shell_solver(K, elements, F, rbe2, coords=None, unit=None, u_init=None, tol=1e-10, max_iter=1000, device="cuda:0", dtype=torch.float64, eps=1e-30):
    """

    Args:
        K (torch.Tensor): Local Stiffness Matrices of shape [M, dofs, dofs].
        elements (torch.Tensor): Connectivity [M, node_per_element].
        F (torch.Tensor): External load vector [N, 3].
        rbe2 (torch.Tensor): Indices of constrained (fixed) nodes, shape [n_fixed].
        u_init (torch.Tensor, optional): Initial guess for displacement [N, 3]. Defaults to zeros.
        tol (float, optional): Convergence tolerance. Default: 1e-10.
        max_iter (int, optional): Maximum number of CG iterations. Default: 1000.
        device (str, optional): "cuda:X" or "cpu". Default: "cuda:0".
        dtype (torch.dtype, optional): Floating precision. Default: torch.float64.
        eps (float, optional): Small epsilon to avoid division by zero. Default: 1e-30.

    Returns:
        torch.Tensor: Displacement vector u of shape [N, 3].
    """

    K = K.to(device=device, dtype=dtype)
    elements = elements.to(device=device)
    F = F.to(device)
    rbe2 = rbe2.to(device)
    if unit is None:
        if coords is not None:
            unit = compute_s3_local_unitvector(coords, elements, device=device)
        else:
            raise ValueError("Neither coords or units data were provided")

    N = F.shape[0] 
    if u_init is None:
        u = torch.zeros((N, 6), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)

    u[rbe2] = 0.0

    Ku_init = compute_shell_nodal_forces(K, elements, u, unit, device=device, dtype=dtype)
    r = F - Ku_init

    r[rbe2] = 0.0

    p = r.clone()

    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        Ap = compute_shell_nodal_forces(K, elements, p, unit, device=device, dtype=dtype)
        pAp = torch.sum(p * Ap)

        if pAp.abs() < eps or pAp < 0.0:
            print(
                f"Terminating early at iteration {i+1}: "
                f"p^T K p = {pAp.item():.3e}, which is invalid for SPD systems."
            )
            break

        alpha = rs_old / (pAp + eps)

        if torch.isnan(alpha) or torch.isinf(alpha):
            print(f"Terminating at iteration {i+1}: alpha is NaN or Inf.")
            break

        u += alpha * p

        u[rbe2] = 0.0

        r -= alpha * Ap

        r[rbe2] = 0.0

        rs_new = torch.sum(r * r)

        if torch.sqrt(rs_new) < tol:
            print(f"Converged after {i+1} iterations. Residual norm: {rs_new.item():.3e}")
            break

        beta = rs_new / (rs_old + eps)

        if torch.isnan(beta) or torch.isinf(beta):
            print(f"Terminating at iteration {i+1}: beta is NaN or Inf.")
            break

        p = r + beta * p

        p[rbe2] = 0.0

        rs_old = rs_new

    else:
        print("CG did not converge within the maximum number of iterations.")

    return u


#################################################################################################
######################################   Constraint   ###########################################
#################################################################################################

def parse_spc_list(spc_list, device="cuda:0", dtype=torch.float64):
    """
    Converts a list of SPC constraints in dict form to flattened PyTorch tensors.
    Input Example:
      spc_list = [
         {
           'node': 10,
           'dofs': [0,1,2],
           'value': 0.0
         },
         {
           'node': 20,
           'dofs': [0],
           'value': 0.01
         }
         ...
      ]

    Returns:
      spc_nodes  (torch.Tensor) : shape [S], each entry is a node index
      spc_dofs   (torch.Tensor) : shape [S], each entry is a DOF index
      spc_values (torch.Tensor) : shape [S], displacement values
    """
    spc_nodes_list  = []
    spc_dofs_list   = []
    spc_values_list = []

    for spc in spc_list:
        node  = spc['node']
        value = spc['value']
        dofs  = spc['dofs']
        for d in dofs:
            spc_nodes_list.append(node)
            spc_dofs_list.append(d)
            spc_values_list.append(value)

    spc_nodes  = torch.tensor(spc_nodes_list,  device=device, dtype=torch.int32)
    spc_dofs   = torch.tensor(spc_dofs_list,   device=device, dtype=torch.int32)
    spc_values = torch.tensor(spc_values_list, device=device, dtype=dtype)
    return spc_nodes, spc_dofs, spc_values

def parse_rbe2_list(rbe2_list, device="cuda:0"):
    """
    Converts a list of RBE2 constraints in dict form to flattened PyTorch tensors.
    Input Example:
      rbe2_list = [
        {
          'master': 15,
          'slaves': [21, 22, 23],
          'dofs': [0,1,2]
        },
        {
          'master': 50,
          'slaves': [51,52],
          'dofs': [0,1,2]
        },
        ...
      ]

    Returns:
      rbe2_slaves  (torch.Tensor) : shape [R], each entry is a slave node index
      rbe2_masters (torch.Tensor) : shape [R], each entry is the matching master node
      rbe2_dofs    (torch.Tensor) : shape [R], each entry is the DOF index
    """
    rbe2_slaves_list  = []
    rbe2_masters_list = []
    rbe2_dofs_list    = []

    for rbe2 in rbe2_list:
        master = rbe2['master']
        dofs   = rbe2['dofs']
        for slave in rbe2['slaves']:
            for d in dofs:
                rbe2_slaves_list.append(slave)
                rbe2_masters_list.append(master)
                rbe2_dofs_list.append(d)

    rbe2_slaves  = torch.tensor(rbe2_slaves_list,  device=device, dtype=torch.int32)
    rbe2_masters = torch.tensor(rbe2_masters_list, device=device, dtype=torch.int32)
    rbe2_dofs    = torch.tensor(rbe2_dofs_list,    device=device, dtype=torch.int32)
    return rbe2_slaves, rbe2_masters, rbe2_dofs

def enforce_constraints(
    u: torch.Tensor,
    r: torch.Tensor,
    spc_nodes: torch.Tensor,
    spc_dofs: torch.Tensor,
    spc_values: torch.Tensor,
    rbe2_slaves: torch.Tensor,
    rbe2_masters: torch.Tensor,
    rbe2_dofs: torch.Tensor
):
    """
    Enforce all SPC and RBE2 constraints in a single pass on GPU.

    Args:
        u (torch.Tensor): Displacement array, shape [N, dofsN].
        r (torch.Tensor): Residual array, same shape as `u`.
        spc_nodes (torch.Tensor): shape [S], node indices for SPC constraints.
        spc_dofs (torch.Tensor): shape [S], DOF indices for SPC constraints.
        spc_values (torch.Tensor): shape [S], displacement values for each constraint.
        rbe2_slaves (torch.Tensor): shape [R], slave node indices for RBE2.
        rbe2_masters(torch.Tensor): shape [R], the matching master node for each slave.
        rbe2_dofs (torch.Tensor): shape [R], DOF indices for RBE2 constraints.

    Returns: 
        None. (Modifies `u` and `r` in place)
    """
    if rbe2_slaves.numel() > 0:
        u[rbe2_slaves, rbe2_dofs] = u[rbe2_masters, rbe2_dofs]
        r[rbe2_slaves, rbe2_dofs] = 0.0

    if spc_nodes.numel() > 0:
        u[spc_nodes, spc_dofs] = spc_values
        r[spc_nodes, spc_dofs] = 0.0

def constrained_conjugate_gradient_solver(K,elements, F, rbe2_list, spc_list, u_init=None, tol=1e-10, max_iter=1000, device="cuda:0", dtype=torch.float64, eps=1e-30):
    """
    A CG solver that enforces constraints using vectorized GPU indexing.

    Args:
        K (torch.Tensor): Local stiffness matrices, shape [M, dofsE, dofsE].
        elements (torch.Tensor): Connectivity, shape [M, nodes_per_element].
        F (torch.Tensor): External load, shape [N, dofsN].
        rbe2_list (list): RBE2 constraints in dict form.
        spc_list (list): SPC constraints in dict form.
        u_init (torch.Tensor, optional): Initial displacement guess, shape [N, dofsN].
        tol (float): Convergence tolerance on residual norm.
        max_iter (int): Max iterations.
        device (str): "cuda:0" or "cpu".
        dtype (torch.dtype): e.g. torch.float64 or torch.float32.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: Displacement array, shape [N, dofsN], with constraints enforced.
    """
    K = K.to(device=device, dtype=dtype)
    elements = elements.to(device=device, dtype=torch.int32)
    F = F.to(device=device, dtype=dtype)

    N, dofsN = F.shape
    if u_init is None:
        u = torch.zeros((N, dofsN), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)

    spc_nodes, spc_dofs, spc_values = parse_spc_list(spc_list, device, dtype)
    rbe2_slaves, rbe2_masters, rbe2_dofs = parse_rbe2_list(rbe2_list, device)

    Ku_init = compute_nodal_forces(K, elements, u, device, dtype)
    r = F - Ku_init

    enforce_constraints(
        u, r,
        spc_nodes, spc_dofs, spc_values,
        rbe2_slaves, rbe2_masters, rbe2_dofs
    )

    p = r.clone()
    rs_old = torch.sum(r * r)

    for i in range(max_iter):
        Ap = compute_nodal_forces(K, elements, p, device, dtype)
        pAp = torch.sum(p * Ap)

        if pAp.abs() < eps or pAp < 0.0:
            print(
                f"[CG] Early terminate @ iter {i+1}: p^T K p = {pAp.item():.3e}, not valid for SPD."
            )
            break

        alpha = rs_old / (pAp + eps)
        if torch.isnan(alpha) or torch.isinf(alpha):
            print(f"[CG] Terminate @ iter {i+1}: alpha is NaN/Inf.")
            break

        u += alpha * p
        r -= alpha * Ap

        enforce_constraints(
            u, r,
            spc_nodes, spc_dofs, spc_values,
            rbe2_slaves, rbe2_masters, rbe2_dofs
        )

        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            print(f"[CG] Converged @ iter {i+1}, residual norm = {rs_new.item():.3e}")
            break

        beta = rs_new / (rs_old + eps)
        if torch.isnan(beta) or torch.isinf(beta):
            print(f"[CG] Terminate @ iter {i+1}: beta is NaN/Inf.")
            break

        p = r + beta * p
        rs_old = rs_new
    else:
        print("[CG] Did not converge within max_iter.")

    return u


#################################################################################################
#########################################   +RBE3   #############################################
#################################################################################################

def parse_rbe3_list(rbe3_list, device="cuda:0", dtype=torch.float64):
    """
    Parses RBE3 constraints from dict format into flattened PyTorch tensors.
    Input Example:
      rbe3_list = [
        {
          'master': 15,
          'slaves': [21, 22, 23],
          'dofs': [0,1,2],
          'weights': [1.0, 2.0, 1.0]
        },
        {
          'master': 50,
          'slaves': [51,52],
          'dofs': [0,1,2],
          'weights': [1.0, 3.0]
        },
        ...
      ]
    Returns:
      rbe3_masters (torch.Tensor) : shape [R], each entry is the matching master node
      rbe3_slaves  (torch.Tensor) : shape [R], each entry is a slave node index
      rbe3_dofs    (torch.Tensor) : shape [R], each entry is the DOF index
      rbe3_inds    (torch.Tensor) : shape [num RBE3], running index of each RBE3
      weighted_sum (torch.Tensor) : shape [num RBE3], sum of weights
    Each dict has 'master', 'slaves', 'dofs', 'weights'. Returns flattened tensors plus index offsets.
    """
    rbe3_master_list, rbe3_slaves_list, rbe3_dofs_list, rbe3_wts_list = [], [], [], []
    weight_sums_list, master_index_offsets = [], [0]
    running_count = 0
    for rbe3 in rbe3_list:
        m, ss, dd, ww = rbe3['master'], rbe3['slaves'], rbe3['dofs'], rbe3['weights']
        for i, s in enumerate(ss):
            for d in dd:
                rbe3_master_list.append(m)
                rbe3_slaves_list.append(s)
                rbe3_dofs_list.append(d)
                rbe3_wts_list.append(ww[i])
        count_sub = len(ss)*len(dd)
        running_count += count_sub
        weight_sums_list.append(sum(ww))
        master_index_offsets.append(running_count)
    rbe3_master  = torch.tensor(rbe3_master_list,  device=device, dtype=torch.int32)
    rbe3_slaves  = torch.tensor(rbe3_slaves_list,  device=device, dtype=torch.int32)
    rbe3_dofs    = torch.tensor(rbe3_dofs_list,    device=device, dtype=torch.int32)
    rbe3_wts     = torch.tensor(rbe3_wts_list,     device=device, dtype=dtype)
    weight_sums  = torch.tensor(weight_sums_list,  device=device, dtype=dtype)
    rbe3_inds    = torch.tensor(master_index_offsets, device=device, dtype=torch.int64)
    return rbe3_master, rbe3_slaves, rbe3_dofs, rbe3_wts, rbe3_inds, weight_sums

def apply_loads_to_F(F, load_list):
    """
    Applies 3D force vectors from dict-based load_list into the global force array F of shape [N,3].
    Each dict has 'node' and 'force'=[Fx, Fy, Fz]. Modifies F in place.
    """
    for ld in load_list:
        node = ld['node']
        fx, fy, fz = ld['force']
        F[node, 0] += fx
        F[node, 1] += fy
        F[node, 2] += fz

def new_enforce_constraints(
    u, r,
    spc_nodes, spc_dofs, spc_values,
    rbe2_slaves, rbe2_masters, rbe2_dofs,
    rbe3_master, rbe3_slaves, rbe3_dofs, rbe3_weights, rbe3_inds, weight_sums
):
    """
    Applies SPC constraints first, then RBE2, then RBE3, modifying u and r in place.
    """
    ### SPC
    if spc_nodes.numel() > 0:
        u[spc_nodes, spc_dofs] = spc_values
        r[spc_nodes, spc_dofs] = 0.0
    ### rbe2
    if rbe2_slaves.numel() > 0:
        u[rbe2_slaves, rbe2_dofs] = u[rbe2_masters, rbe2_dofs]
        r[rbe2_slaves, rbe2_dofs] = 0.0
    ### rbe3
    num_rbe3 = rbe3_inds.shape[0] - 1
    for i in range(num_rbe3):
        start = rbe3_inds[i].item()
        end   = rbe3_inds[i+1].item()
        w_sum = weight_sums[i]
        m_sub = rbe3_master[start:end]
        s_sub = rbe3_slaves[start:end] 
        d_sub = rbe3_dofs[start:end]
        w_sub = rbe3_weights[start:end]
        master_node = m_sub[0].item() 
        unique_dofs = d_sub.unique()
        for dval in unique_dofs:
            mask = (d_sub == dval)
            slave_nodes = s_sub[mask]
            slave_wts   = w_sub[mask]
            disp_sum = torch.sum(slave_wts * u[slave_nodes, dval])
            u_val = disp_sum / (w_sum + 1e-30)
            u[master_node, dval] = u_val

def new_constrained_conjugate_gradient_solver(
    K, elements, N,
    rbe2_list, rbe3_list, spc_list, load_list,
    u_init=None, tol=1e-10, max_iter=1000,
    device="cuda:0", dtype=torch.float64, eps=1e-30
):
    """
    Builds a force array F from 3D loads, parses RBE2/RBE3/SPC constraints, then solves using CG.
    Enforces SPC, RBE2, RBE3 in a single combined function after each update.
    """
    K = K.to(device=device, dtype=dtype)
    elements = elements.to(device=device, dtype=torch.int32)
    F = torch.zeros((N, 3), device=device, dtype=dtype)
    apply_loads_to_F(F, load_list)

    rbe2_slaves, rbe2_masters, rbe2_dofs = parse_rbe2_list(rbe2_list, device)
    rbe3_master, rbe3_slaves, rbe3_dofs, rbe3_weights, rbe3_inds, weight_sums = parse_rbe3_list(rbe3_list, device, dtype)
    spc_nodes, spc_dofs, spc_values = parse_spc_list(spc_list, device, dtype)

    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    Ku_init = compute_nodal_forces(K, elements, u, device, dtype)
    r = F - Ku_init
    new_enforce_constraints(u, r, spc_nodes, spc_dofs, spc_values, rbe2_slaves, rbe2_masters, rbe2_dofs,
                        rbe3_master, rbe3_slaves, rbe3_dofs, rbe3_weights, rbe3_inds, weight_sums)
    p = r.clone()
    rs_old = torch.sum(r * r)
    for i in range(max_iter):
        Ap = compute_nodal_forces(K, elements, p, device, dtype)
        pAp = torch.sum(p * Ap)
        if pAp.abs() < eps or pAp < 0.0:
            print(
                f"[CG] Early terminate @ iter {i+1}: p^T K p = {pAp.item():.3e}, not valid for SPD."
            )
            break
        alpha = rs_old / (pAp + eps)
        if torch.isnan(alpha) or torch.isinf(alpha):
            print(f"[CG] Terminate @ iter {i+1}: alpha is NaN/Inf.")
            break
        u += alpha * p
        r -= alpha * Ap
        new_enforce_constraints(u, r, spc_nodes, spc_dofs, spc_values, rbe2_slaves, rbe2_masters, rbe2_dofs,
                            rbe3_master, rbe3_slaves, rbe3_dofs, rbe3_weights, rbe3_inds, weight_sums)
        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            print(f"[CG] Converged @ iter {i+1}, residual norm = {rs_new.item():.3e}")
            break
        beta = rs_new / (rs_old + eps)
        if torch.isnan(beta) or torch.isinf(beta):
            print(f"[CG] Terminate @ iter {i+1}: beta is NaN/Inf.")
            break
        p = r + beta * p
        rs_old = rs_new
    else:
        print("[CG] Did not converge within max_iter.")
    return u


#################################################################################################
#######################################   PreCG solver   ########################################
#################################################################################################

def preconditioned_conjugate_gradient_solver(K, elements, F, M_inv, u_init=None, tol=1e-8, max_iter=1000, device="cuda:0", dtype=torch.float32):
    """
    Preconditioned Conjugate Gradient solver for K u = F.
    
    Args:
        K (torch.Tensor): Local stiffness matrices [M, dofs, dofs]
        elements (torch.Tensor): Element connectivity [M, nodes per element]
        F (torch.Tensor): Load vector [N, 3]
        M_inv (torch.Tensor): Preconditioner (inverse of the diagonal of K)
        u_init (torch.Tensor, optional): Initial displacement vector [N, 3]. Defaults to zeros.
        tol (float, optional): Tolerance for convergence
        max_iter (int, optional): Maximum number of iterations
        
    Returns:
        torch.Tensor: Displacement vector u [N, 3]
    """
    N = F.shape[0]
    K = K.to(device)
    elements = elements.to(device)
    F = F.to(device, dtype=dtype)
    M_inv = M_inv.to(device, dtype=dtype)
    
    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    
    r = F - compute_nodal_forces(K, elements, u, device=device, dtype=dtype)
    z = M_inv * r
    p = z.clone()
    rs_old = torch.sum(r * z)
    
    for i in range(max_iter):
        Ap = compute_nodal_forces(K, elements, p, device=device, dtype=dtype)
        alpha = rs_old / torch.sum(p * Ap)
        u += alpha * p
        r -= alpha * Ap
        z = M_inv * r
        rs_new = torch.sum(r * z)
        if torch.sqrt(rs_new) < tol:
            print(f'Converged after {i+1} iterations.')
            break
        p = z + (rs_new / rs_old) * p
        rs_old = rs_new
    else:
        print('Preconditioned CG did not converge within the maximum number of iterations.')
    return u

def compute_diagonal_preconditioner(K, elements, N, device="cuda:0", dtype=torch.float32):
    """
    Computes the inverse of the diagonal of the global stiffness matrix K without assembling K.
    
    Args:
        K (torch.Tensor): Local stiffness matrices [M, dofs, dofs]
        elements (torch.Tensor): Element connectivity [M, nodes per element]
        N (int): Number of nodes
    """
    diag_K = torch.zeros((N * 3,), device=device, dtype=dtype)
    dofs_per_node = 3
    dofs = elements.unsqueeze(-1) * dofs_per_node + torch.arange(dofs_per_node, device=device).view(1, 1, -1)
    dofs = dofs.view(-1)
    
    K_diagonal_entries = K.view(-1, K.shape[-1])[:, ::K.shape[-1] + 1].view(-1)
    diag_K.index_add_(0, dofs, K_diagonal_entries)
    M_inv = 1.0 / diag_K
    M_inv[M_inv == float('inf')] = 0.0 
    M_inv = M_inv.view(N, 3)
    return M_inv


#################################################################################################
#######################################   BCG solver   ##########################################
#################################################################################################

def bicgstab_solver(K, elements, F, rbe2, u_init=None, tol=1e-8, max_iter=1000, device="cuda:0", dtype=torch.float32):
    """
    Bi-Conjugate Gradient Stabilized (BiCGStab) solver for K u = F.
    
    Args:
        K (torch.Tensor): Local stiffness matrices [M, dofs, dofs]
        elements (torch.Tensor): Element connectivity [M, nodes per element]
        F (torch.Tensor): Load vector [N, 3]
        rbe2 (torch.Tensor) : rbe2 mask
        u_init (torch.Tensor, optional): Initial displacement vector [N, 3]. Defaults to zeros.
        tol (float, optional): Tolerance for convergence
        max_iter (int, optional): Maximum number of iterations
        
    Returns:
        torch.Tensor: Displacement vector u [N, 3]
    """
    N = F.shape[0]
    K = K.to(device)
    elements = elements.to(device)
    rbe2 = rbe2.to(device)
    F = F.to(device, dtype=dtype)
    
    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    
    # Initial residual
    r = F - compute_nodal_forces(K, elements, u, device=device, dtype=dtype)
    r_hat = r.clone()
    rho_old = alpha = omega = 1.0
    v = torch.zeros_like(u)
    p = torch.zeros_like(u)
    
    rs_old = torch.sum(r * r)
    tol_squared = (tol * torch.sqrt(rs_old)).item() ** 2
    
    for i in range(max_iter):
        u[rbe2] = torch.tensor([0,0,0], dtype=torch.float32).to(device)
        rho_new = torch.sum(r_hat * r)
        if rho_new == 0:
            print('Breakdown in BiCGStab: rho_new = 0')
            break
        if i == 0:
            p = r.clone()
        else:
            beta = (rho_new / rho_old) * (alpha / omega)
            p = r + beta * (p - omega * v)
        v = compute_nodal_forces(K, elements, p, device=device, dtype=dtype)
        alpha = rho_new / torch.sum(r_hat * v)
        s = r - alpha * v
        t = compute_nodal_forces(K, elements, s, device=device, dtype=dtype)
        omega = torch.sum(t * s) / torch.sum(t * t)
        u += alpha * p + omega * s
        r = s - omega * t
        rho_old = rho_new
        
        rs_new = torch.sum(r * r)
        if rs_new < tol_squared:
            print(f'Converged after {i+1} iterations.')
            break
        u[rbe2] = torch.tensor([0,0,0], dtype=torch.float32).to(device)
    else:
        print('BiCGStab did not converge within the maximum number of iterations.')
    return u


#################################################################################################
#######################################   GMRES solver   ########################################
#################################################################################################

def gmres_solver(K, elements, F, u_init=None, tol=1e-8, max_iter=1000, restart=50, device="cuda:0", dtype=torch.float32):
    """
    Generalized Minimal Residual (GMRES) solver for K u = F with restarts.
    
    Args:
        K (torch.Tensor): Local stiffness matrices [M, dofs, dofs]
        elements (torch.Tensor): Element connectivity [M, nodes per element]
        F (torch.Tensor): Load vector [N, 3]
        u_init (torch.Tensor, optional): Initial displacement vector [N, 3]. Defaults to zeros.
        tol (float, optional): Tolerance for convergence
        max_iter (int, optional): Maximum number of iterations
        restart (int, optional): Number of iterations before restart
        
    Returns:
        torch.Tensor: Displacement vector u [N, 3]
    """
    N = F.shape[0]
    K = K.to(device)
    elements = elements.to(device)
    F = F.to(device, dtype=dtype)
    
    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    
    r = F - compute_nodal_forces(K, elements, u, device=device, dtype=dtype)
    beta = torch.norm(r)
    if beta < tol:
        print('Converged at initial guess.')
        return u
    
    for outer_iter in range(0, max_iter, restart):
        V = [r / beta]  # List of orthonormal basis vectors
        H = torch.zeros((restart + 1, restart), device=device, dtype=dtype)
        
        for i in range(restart):
            # Arnoldi process
            w = compute_nodal_forces(K, elements, V[i], device=device, dtype=dtype)
            for k in range(i + 1):
                H[k, i] = torch.dot(V[k].flatten(), w.flatten())
                w -= H[k, i] * V[k]
            H[i + 1, i] = torch.norm(w)
            if H[i + 1, i] != 0:
                V.append(w / H[i + 1, i])
            else:
                break  
            e1 = torch.zeros((i + 2,), device=device, dtype=dtype)
            e1[0] = beta
            H_small = H[:i + 2, :i + 1]
            y, _ = torch.lstsq(e1.unsqueeze(1), H_small)
            u = u + sum(y[j] * V[j] for j in range(i + 1))
            r = F - compute_nodal_forces(K, elements, u, device=device, dtype=dtype)
            if torch.norm(r) < tol:
                print(f'Converged after {outer_iter + i + 1} iterations.')
                return u
    else:
        print('GMRES did not converge within the maximum number of iterations.')
    return u


#################################################################################################
###################################  Newton-Rhapson Method  #####################################
#################################################################################################

##### 비선형성이 아주 강한 경우를 제외하고는 필요 없음! 비선형이 강한 경우에도 충분한 iteration과 좋은 초기조건 모델이 있다면 마찬가지로 필요 X

def newton_raphson_solver(K_func, elements, F_ext, u_init=None, tol=1e-8, max_iter=50, device="cuda:0", dtype=torch.float32):
    """
    Newton-Raphson solver for nonlinear systems K(u) * u = F_ext.
    
    Args:
        K_func (callable): Function that computes local stiffness matrices K(u) given displacements u.
                           Should return K_local [M, dofs, dofs].
        elements (torch.Tensor): Element connectivity [M, nodes per element].
        F_ext (torch.Tensor): External load vector [N, 3].
        u_init (torch.Tensor, optional): Initial displacement vector [N, 3]. Defaults to zeros.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of iterations. Defaults to 50.
        device (str, optional): Device to perform computations on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to torch.float32.
    
    Returns:
        torch.Tensor: Displacement vector u [N, 3].
    """
    N = F_ext.shape[0]
    elements = elements.to(device=device)
    F_ext = F_ext.to(device=device, dtype=dtype)

    if u_init is None:
        u = torch.zeros((N, 3), device=device, dtype=dtype)
    else:
        u = u_init.clone().to(device=device, dtype=dtype)
    
    for iteration in range(max_iter):
        K_local = K_func(u)
        
        F_int = compute_nodal_forces(K_local, elements, u, device=device, dtype=dtype)
        
        R = F_ext - F_int
        norm_R = torch.norm(R)
        print(f"Iteration {iteration+1}, Residual Norm: {norm_R.item()}")
        
        if norm_R < tol:
            print(f'Converged after {iteration+1} iterations.')
            break
        
        def compute_Ku(delta_u):
            return compute_nodal_forces(K_local, elements, delta_u, device=device, dtype=dtype)
        
        delta_u = conjugate_gradient_solver_Ku(compute_Ku, R, tol=tol, max_iter=100, device=device, dtype=dtype)
        
        u += delta_u
    else:
        print('Newton-Raphson did not converge within the maximum number of iterations.')
    
    return u

def conjugate_gradient_solver_Ku(compute_Ku, R, tol=1e-8, max_iter=1000, device="cuda:0", dtype=torch.float32):
    """
    Conjugate Gradient solver for solving K(u) * delta_u = R.
    
    Args:
        compute_Ku (callable): Function to compute K(u) * delta_u.
        R (torch.Tensor): Residual vector [N, 3].
        tol (float, optional): Tolerance for convergence. Defaults to 1e-8.
        max_iter (int, optional): Maximum number of iterations. Defaults to 1000.
        device (str, optional): Device to perform computations on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to torch.float32.
        
    Returns:
        torch.Tensor: Displacement increment delta_u [N, 3].
    """
    N = R.shape[0]
    delta_u = torch.zeros((N, 3), device=device, dtype=dtype)
    
    r = R - compute_Ku(delta_u)
    p = r.clone()
    rs_old = torch.sum(r * r)
    
    for i in range(max_iter):
        Ap = compute_Ku(p)
        alpha = rs_old / torch.sum(p * Ap)
        delta_u += alpha * p
        r -= alpha * Ap
        rs_new = torch.sum(r * r)
        if torch.sqrt(rs_new) < tol:
            # Optionally, print convergence info
            # print(f'CG converged after {i+1} iterations.')
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    else:
        print('CG did not converge within the maximum number of iterations.')
    return delta_u


















def vectorized_modal_solver(
    K_local,
    M_local,
    elements,
    rbe2_node_ids,
    num_nodes,
    num_eigs=5,
    max_iter=20,
    device="cuda:0",
    dtype=torch.float32
):
    """
    Vectorized subspace-iteration-based modal solver in pure PyTorch,
    that finds the 'num_eigs' smallest approximate eigenvalues/eigenvectors
    of the generalized problem K*u = lambda*M*u.
    
    Steps:
      1) Build lumped diagonal mass from M_local => Mdiag
      2) Use subspace iteration on X in R^(3*N x num_eigs):
         a) Y = M^-1 * K * X
         b) Orthonormalize Y
         c) Solve small (num_eigs x num_eigs) GEVP => update subspace
      3) Zero out boundary DOFs from rbe2_node_ids each iteration
      4) Output the final approximate eigenvalues (lam) and modes
    
    Warnings:
      - This is a naive approach (no shift-and-invert, no robust preconditioning).
      - We use Euclidean norms for orthonormalization, not M-norm.
      - Invert B_k with Gauss-Jordan, and do a simple Jacobi rotation to find eigenpairs.
    """

    # Move to device/dtype
    K_local = K_local.to(device=device, dtype=dtype)
    M_local = M_local.to(device=device, dtype=dtype)
    elements = elements.to(device=device, dtype=torch.long)
    rbe2_node_ids = rbe2_node_ids.to(device=device, dtype=torch.long)

    n_dof = num_nodes * 3

    # ----------------------------------------------------------------
    # 1) Build lumped diagonal mass array => Mdiag
    # ----------------------------------------------------------------
    Mdiag = torch.zeros(n_dof, device=device, dtype=dtype)
    dofs = elements.unsqueeze(-1)*3 + torch.tensor([0,1,2], device=device, dtype=torch.long).view(1,1,3)
    dofs = dofs.view(-1)  
    # We must use contiguous or reshape for diagonal
    m_diag_loc = M_local.diagonal(dim1=1, dim2=2).contiguous().view(-1)
    Mdiag.index_add_(0, dofs, m_diag_loc)
    eps = 1e-12
    Mdiag.clamp_(min=eps)          # Avoid zeros
    Minv_diag = 1.0 / Mdiag       # shape [n_dof]

    # ----------------------------------------------------------------
    # 2) Boundary DOFs
    # ----------------------------------------------------------------
    # For each node in rbe2_node_ids => 3 DOFs
    fix_dofs_list = (rbe2_node_ids.view(-1,1)*3 +
                     torch.tensor([0,1,2], device=device, dtype=torch.long)
                    ).view(-1)

    def clamp_boundary_dofs(matrix_2d):
        # matrix_2d: shape [n_dof, kdim]
        matrix_2d[fix_dofs_list, :] = 0.0

    # ----------------------------------------------------------------
    # 3) define apply_A(X) = M^-1 * K * X
    # ----------------------------------------------------------------
    def apply_A(X_2d):
        """
        X_2d: [n_dof, kdim].
        We'll apply K to each column, then multiply by Minv_diag.
        """
        kdim = X_2d.shape[1]
        # reshape to [num_nodes,3,kdim]
        X_resh = X_2d.view(num_nodes, 3, kdim)
        # partial vectorization: small loop over kdim
        forces_list = [
            compute_nodal_forces(
                K_local, elements, X_resh[..., i],
                device=device, dtype=dtype
            ).view(n_dof, 1)
            for i in range(kdim)
        ]
        KX_2d = torch.cat(forces_list, dim=1)  # [n_dof, kdim]
        return KX_2d * Minv_diag.unsqueeze(1)  # broadcast multiply

    # ----------------------------------------------------------------
    # 4) Gram-Schmidt Euclidean Orthonormalization (in-place)
    # ----------------------------------------------------------------
    def gram_schmidt_euclid(X_2d):
        kdim = X_2d.shape[1]
        for j in range(kdim):
            col_j = X_2d[:, j]
            norm_j = torch.norm(col_j, p=2)
            if norm_j < 1e-14:
                col_j[:] = 0.
                if j < col_j.numel():
                    col_j[j] = 1.
            else:
                col_j /= norm_j
            if j > 0:
                prev_cols = X_2d[:, :j]
                # dot with all previous columns
                dots = (prev_cols * col_j.unsqueeze(1)).sum(dim=0)
                col_j -= prev_cols @ dots
            norm2 = torch.norm(col_j, p=2)
            if norm2 < 1e-14:
                col_j[:] = 0.
                if j < col_j.numel():
                    col_j[j] = 1.
            else:
                col_j /= norm2

    # ----------------------------------------------------------------
    # 5) Initial subspace X => random
    # ----------------------------------------------------------------
    X = torch.randn(n_dof, num_eigs, device=device, dtype=dtype)
    clamp_boundary_dofs(X)
    gram_schmidt_euclid(X)

    # ----------------------------------------------------------------
    # 6) Build A_k, B_k => [kdim,kdim], solve small GEVP
    # ----------------------------------------------------------------
    def build_Ak_Bk(X_2d):
        kdim = X_2d.shape[1]
        X_resh = X_2d.view(num_nodes, 3, kdim)
        # KX
        KX_list = [
            compute_nodal_forces(K_local, elements, X_resh[..., i],
                                 device=device, dtype=dtype).view(n_dof,1)
            for i in range(kdim)
        ]
        KX_2d = torch.cat(KX_list, dim=1)
        # MX => lumps
        MX_2d = X_2d * Mdiag.unsqueeze(1)
        A_k = X_2d.transpose(0,1) @ KX_2d
        B_k = X_2d.transpose(0,1) @ MX_2d
        return A_k, B_k

    def invert_small_matrix(MM):
        """
        Gauss-Jordan with row clones to avoid memory overlap errors in PyTorch.
        """
        n = MM.shape[0]
        Aug = torch.cat([MM, torch.eye(n, device=device, dtype=dtype)], dim=1)
        for i in range(n):
            row_i = Aug[i].clone()
            pivot = row_i[i]
            if abs(pivot) < 1e-14:
                pivot = 1e-14
            row_i /= pivot
            Aug[i] = row_i
            for r in range(n):
                if r != i:
                    row_r = Aug[r].clone()
                    factor = row_r[i]
                    row_r -= factor * row_i
                    Aug[r] = row_r
        return Aug[:, n:]

    def naive_jacobi_symmetric(AA, max_sweeps=30, tol=1e-10):
        n = AA.shape[0]
        V = torch.eye(n, device=device, dtype=dtype)
        for _ in range(max_sweeps):
            diagA = torch.diagonal(AA)
            off = AA - torch.diag_embed(diagA)
            idxmax = torch.argmax(off.abs())
            i = idxmax // n
            j = idxmax % n
            if i == j:
                break
            if i>j:
                i, j = j, i
            val = AA[i,j]
            if abs(val) < tol:
                break
            theta = 0.5*(AA[j,j]-AA[i,i]) / val
            t = torch.sign(theta)/(abs(theta)+torch.sqrt(1+theta*theta))
            c = 1./torch.sqrt(1+t*t)
            s = t*c
            aii = AA[i,i]
            aij = AA[i,j]
            ajj = AA[j,j]
            AA[i,i] = aii - t*aij
            AA[j,j] = ajj + t*aij
            AA[i,j] = 0
            AA[j,i] = 0
            rowi = AA[i].clone()
            rowj = AA[j].clone()
            AA[i] = c*rowi - s*rowj
            AA[j] = s*rowi + c*rowj
            AA[:, i] = AA[i]
            AA[:, j] = AA[j]
            Vi = V[:, i].clone()
            Vj = V[:, j].clone()
            V[:, i] = c*Vi - s*Vj
            V[:, j] = s*Vi + c*Vj
        eigvals = torch.diagonal(AA)
        eigvals_s, idx = eigvals.sort()
        V_s = V[:, idx]
        return eigvals_s, V_s

    def solve_small_gevp(A_k, B_k):
        B_inv = invert_small_matrix(B_k)
        A_ = B_inv @ A_k
        A_copy = A_.clone()
        lam, Z = naive_jacobi_symmetric(A_copy)
        return lam, Z

    # ----------------------------------------------------------------
    # 7) Subspace iteration
    # ----------------------------------------------------------------
    for _iter in range(max_iter):
        Y = apply_A(X)
        clamp_boundary_dofs(Y)
        gram_schmidt_euclid(Y)

        A_k, B_k = build_Ak_Bk(Y)
        lam, Z = solve_small_gevp(A_k, B_k)

        Ynew = Y @ Z
        clamp_boundary_dofs(Ynew)
        gram_schmidt_euclid(Ynew)
        X = Ynew

    A_k, B_k = build_Ak_Bk(X)
    lam, Z = solve_small_gevp(A_k, B_k)
    modes = X @ Z

    return lam, modes