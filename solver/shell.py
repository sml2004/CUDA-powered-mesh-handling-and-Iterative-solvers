


import torch
import numpy as np





#################################################################################################
##########################################   통합 함수   ##########################################
#################################################################################################

def compute_kirchoff_D_matrix(membrane, bending, device="cuda:0", dtype=torch.float32):
    """
    Kirchoff plate bending stiffness matrix를 구하는 함수
    Membrane과 bending의 정보를 받아서 D matrix를 계산한다.

    Inputs:
        membrane (torch.Tensor): [E, nu, thickness]
        bending (torch.Tensor): [E, nu, thickness]
    Outputs:
        D (torch.Tensor): Kirchoff plate bending stiffness matrix
    """
    E_m, nu_m, t_m = membrane
    E_b, nu_b, t_b = bending

    a = E_m * t_m / (1-nu_m**2)
    b = E_b * t_b**3 / (12*(1-nu_b**2))

    D = torch.tensor([[a, nu_m*a, 0, 0, 0, 0],
                      [nu_m*a, a, 0, 0, 0, 0],
                      [0, 0, a*(1-nu_m)/2, 0, 0, 0],
                      [0, 0, 0, b, nu_b*b, 0],
                      [0, 0, 0, nu_b*b, b, 0],
                      [0, 0, 0, 0, 0, b*(1-nu_b)/2]], device=device, dtype=dtype) # [6, 6]
    
    return D

def compute_global_to_local_displacement(shell, displacement, unit, device="cuda:0"):
    shell = shell.to(device)
    displacement = displacement.to(device)
    unit = unit.to(device)

    global_displacement = displacement[shell]  # [M, num_nodes_per_element, 6]

    global_trans = global_displacement[..., :3]  # [M, num_nodes_per_element, 3]
    global_rot   = global_displacement[..., 3:]  # [M, num_nodes_per_element, 3]

    local_trans = torch.einsum('mij,mkj->mik', global_trans, unit) # [M, num_nodes_per_element, 3]
    local_rot   = torch.einsum('mij,mkj->mik', global_rot, unit)   # [M, num_nodes_per_element, 3]

    local = torch.cat([local_trans, local_rot], dim=-1)  # [M, num_nodes_per_element, 6]

    return local

def compute_shell_nodal_forces(K, shell, displacement, unit, device="cuda:0", dtype=torch.float32):
    """
    Computes global nodal forces for various shell types without assembling the global stiffness matrix.

    Args:
        K (torch.Tensor): Local stiffness matrices [M, DOF, DOF]
        shell (torch.Tensor): shell connectivity [M, num_nodes_per_element]
        displacement (torch.Tensor): Nodal displacements [N, 6]
        unit (torch.Tensor): Local unit vector [M, 3, 3]
        device (str, optional): Device to perform computations on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to torch.float32.

    Returns:
        torch.Tensor: Global nodal force vector [N, 6]
    """
    K = K.to(device=device, dtype=dtype)
    shell = shell.to(device=device, dtype=torch.long)
    displacement = displacement.to(device=device, dtype=dtype)
    unit = unit.to(device=device, dtype=dtype)

    M = K.shape[0]  
    N = displacement.shape[0] 

    dofs = shell.unsqueeze(-1) * 6 + torch.tensor([0, 1, 2, 3, 4, 5], device=device).view(1, 1, 6) # [M, num_nodes_per_element, 6]
    dofs = dofs.view(M, -1)  # [M, dof_per_element]

    u_local = compute_global_to_local_displacement(shell, displacement, unit, device=device)  # [M, num_nodes_per_element, 6]
    u_local = u_local.view(M, -1) # [M, dof]

    F_local = torch.bmm(K, u_local.unsqueeze(-1)).squeeze(-1)  # [M, dof]
    F_local = F_local.view(M, -1, 6)  # [M, num_nodes_per_element, 6]

    F_local_trans = F_local[...,0:3]  # [M, num_nodes_per_element, 3]
    F_local_rot   = F_local[...,3:6]  # [M, num_nodes_per_element, 3]
    unitT = unit.transpose(1,2)  # [M,3,3]

    F_trans_global = torch.einsum('mab,mnb->mna', unitT, F_local_trans)  # [M,num_nodes_per_element,3]
    F_rot_global   = torch.einsum('mab,mnb->mna', unitT, F_local_rot)  # [M,num_nodes_per_element,3]
    F_global_elem = torch.cat([F_trans_global, F_rot_global], dim=-1)  # [M,num_nodes_per_element,6]

    F_global = torch.zeros((N * 6,), device=device, dtype=dtype)  # [N*6]
    F_global = F_global.index_add(0, dofs.view(-1), F_global_elem.view(-1))  # [N*6]
    F_global = F_global.view(N, 6)  # [N, 6]

    return F_global 

def compute_shell_postprocess_values(NMQ, t, z=0, device="cuda:0", dtype=torch.float32):
    """
    Shell의 Element wise [N, M, (Q)] stress vector를 받아서 후처리 값들을 계산하는 함수

    Input:
        NMQ (torch.Tensor): stress vector [num_elements, 6] / [num_elements, 8]
        t (float): Shell thickness
        z (float): Through-thickness coordinate
        device (str, optional): Device to perform computations on. Defaults to "cuda:0".
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to torch.float32.
    Output:
        A dict of torch.Tensors, each shape [N, M], including:
            'sx'      => sigma_x at z
            'sy'      => sigma_y at z
            'txy'     => tau_xy at z
            's1'      => principal stress #1
            's2'      => principal stress #2
            'theta_p' => principal angle (radians)
            'tau_max' => max in-plane shear
            'sigma_vm'=> 2D "von Mises" stress
    """
    NMQ = NMQ.to(device=device, dtype=dtype)
    Nxx = NMQ[:, 0] # [num_elements]
    Nyy = NMQ[:, 1] # [num_elements]
    Nxy = NMQ[:, 2] # [num_elements]
    Mxx = NMQ[:, 3] # [num_elements]
    Myy = NMQ[:, 4] # [num_elements]
    Mxy = NMQ[:, 5] # [num_elements]

    factor1 = 1.0 / t
    factor2 = 6.0 * z / (t**2)
    sx = Nxx*factor1 + Mxx*factor2
    sy = Nyy*factor1 + Myy*factor2
    txy= Nxy*factor1 + Mxy*factor2

    half = 0.5*(sx+sy)
    diff = 0.5*(sx-sy)
    R = torch.sqrt(diff*diff + txy*txy)
    s1 = half + R
    s2 = half - R

    eps = 1.0e-30
    denom = (sx - sy).clamp(min=eps, max=None)
    angle_p = 0.5 * torch.atan2(2.0*txy, denom)  # [num_elements]

    tau_max = 0.5*(s1 - s2)

    sigma_vm = torch.sqrt(s1*s1 - s1*s2 + s2*s2 + 1.0e-30)

    return {
        'sx'      : sx,
        'sy'      : sy,
        'txy'     : txy,
        's1'      : s1,
        's2'      : s2,
        'theta_p' : angle_p,  
        'tau_max' : tau_max,
        'vm_stress': sigma_vm
    }



#################################################################################################
###########################################   Shell   ###########################################
#################################################################################################
'''
모든 Shell은 Element 단위의 자체적인 좌표계가 필요하다.
Node 0을 원점, Node 1을 x축, Node 2를 y축으로 하는 좌표계를 사용한다. y축의 경우 x축과 수직하게 바꿔준다.
'''

##### s3 노드순서 및 shape function (linear)
'''
Node 0:  (xi, eta) = (0, 0) 
Node 1:  (xi, eta) = (1, 0) 
Node 2:  (xi, eta) = (0, 1)

N_0 = 1 - ξ - η
N_1 = ξ
N_2 = η
'''
def compute_s3_normal(coords, shell, device="cuda:0"):
    """
    삼각형 면적에 대응되는 노멀 벡터를 계산합니다.

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        shell (torch.Tensor): Shell connectivity [M, 3] (3 node IDs for each triangle)
        
    Output:
        normal (torch.Tensor): Shell Normal [M, 3]
    """
    coords = coords.to(device)
    shell  = shell.to(device)

    a = coords[shell[:, 1]] - coords[shell[:, 0]] # [M,3]
    b = coords[shell[:, 2]] - coords[shell[:, 0]] # [M,3]

    normal = torch.cross(a, b, dim=1) * 0.5 # [M,3]

    return normal # [M,3]

def identify_s3_shared_edges(shell, device="cuda:0"):
    """
    Shell 정보가 주어지면 각 변이 공유되는 shell 정보를 구하는 함수
    아웃풋은 [S, 2, 2]이며, 총 S개의 변에 대해 [2, 2] 연결 정보를 제공한다.
    [2, 2]:[[shell, edge index],[shell id, edge index]]

    Inputs:
        shell (torch.Tensor): Shell connectivity [M, 3] 

    Outputs:
        shared_edge_indices (torch.Tensor): Indices of shells sharing each edge [S, 2, 2]
    """
    M = shell.shape[0]
    shell = shell.to(device)
    
    edge_node_indices = torch.tensor([
        [0, 1],  # Edge 0
        [1, 2],  # Edge 1
        [2, 0],  # Edge 2
    ], device=device)  # [3, 2]
    
    edge = shell[:, edge_node_indices]  # [M, 3, 2]
    edges_sorted, _ = torch.sort(edge, dim=2)  # [M, 3, 2]
    edges_flat = edges_sorted.view(-1, 2)  # [M*3, 2]
    
    shell_ids = torch.arange(M, device=device).repeat_interleave(3)  # [M*3]
    edge_indices = torch.tile(torch.arange(3, device=device), (M,))  # [M*3]
    
    _, inverse_indices, counts = torch.unique(
        edges_flat, return_inverse=True, return_counts=True, dim=0
    )
    
    shared_mask = counts == 2
    shared_edge_ids = torch.nonzero(shared_mask, as_tuple=True)[0]  # [S]
    
    if shared_edge_ids.numel() == 0:
        return torch.empty((0, 2, 2), dtype=torch.long, device=device)
    
    sorted_inverse, sorted_order = torch.sort(inverse_indices)
    sorted_shell_ids = shell_ids[sorted_order]
    sorted_edge_indices = edge_indices[sorted_order]
    
    positions = torch.searchsorted(sorted_inverse, shared_edge_ids)
    
    shell1 = sorted_shell_ids[positions]
    edge1 = sorted_edge_indices[positions]
    shell2 = sorted_shell_ids[positions + 1]
    edge2 = sorted_edge_indices[positions + 1]
    
    shared_edge_indices = torch.stack([
        torch.stack([shell1, edge1], dim=1),
        torch.stack([shell2, edge2], dim=1)
    ], dim=1)  # [S, 2, 2]
    
    return shared_edge_indices  # [S, 2, 2]

def compute_triangle_surface_faces_with_third_node(shell, device="cuda:0"):
    """
    삼각형 정보가 주어지면 1번만 등장하는 변(테두리)을 구하고, 이러한 테두리 삼각형의 테두리가 아닌 3번째 노드 번호 또한 구하는 함수
    3번째 노드 정보는 바같방향을 구하기 위해 사용된다.

    Input:
        shell (torch.Tensor): Shell connectivity [M, 3] 
    
    Returns:
        surfaces (torch.Tensor): surface connectivity [K, 2]
        3rd id (torch.Tensor): 3rd node id [K]
    """
    shell = shell.to(device)

    edges = torch.cat([
        shell[:, [0, 1]],  # edge between node 0 and node 1
        shell[:, [1, 2]],  # edge between node 1 and node 2
        shell[:, [2, 0]],  # edge between node 2 and node 0
    ], dim=0)  # [3*M, 2]

    third_nodes = torch.cat([
        shell[:, 2],  # third node for edge [0, 1]
        shell[:, 0],  # third node for edge [1, 2]
        shell[:, 1],  # third node for edge [2, 0]
    ], dim=0)  # [3*M]

    sorted_edges, _ = torch.sort(edges, dim=1)
    
    _, inverse_indices, counts = torch.unique(sorted_edges, dim=0, return_inverse=True, return_counts=True)
    
    surface_edge_mask = counts[inverse_indices] == 1
    surface_edges = edges[surface_edge_mask]
    surface_third_nodes = third_nodes[surface_edge_mask]
    
    return surface_edges, surface_third_nodes # [K, 2], [K]

def compute_s3_local_unitvector(coords, shell, device="cuda:0"):
    '''
    삼각형의 로컬 좌표계를 구하는 함수

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        shell (torch.Tensor): Shell connectivity [M, 3]
    
    Output:
        unit (torch.Tensor): Local unit vector [M, 3, 3]
    '''
    coords = coords.to(device)
    shell = shell.to(device)

    a = coords[shell[:, 1]] - coords[shell[:, 0]] # [M,3]
    b = coords[shell[:, 2]] - coords[shell[:, 0]] # [M,3]
    b = b - (torch.sum(a * b, dim=1, keepdim=True) / torch.sum(a * a, dim=1, keepdim=True)) * a  # Make b perpendicular to a

    a = a / torch.norm(a, dim=1, keepdim=True) # [M,3]
    b = b / torch.norm(b, dim=1, keepdim=True) # [M,3]
    c = torch.cross(a, b, dim=1) # [M,3]

    unit = torch.stack([a, b, c], dim=1) # [M,3,3]

    return unit

def compute_s3_global_to_local_coordinates(coords, shell, unit, device="cuda:0", dtype=torch.float32):
    '''
    Global 좌표계를 로컬 좌표계로 변환하는 함수

    Input:
        shell (torch.Tensor): Shell connectivity [M, 3]
        target (torch.Tensor): Global coordinates [N, 3]
        unit (torch.Tensor): Local unit vector [M, 3, 3]
    
    Output:
        local (torch.Tensor): Local coordinates [M, 3, 3]
    '''
    coords = coords.to(device)
    shell = shell.to(device)
    unit = unit.to(device)

    M = shell.shape[0]

    global_coordinates = coords[shell] # [M,3,3]
    origin = global_coordinates[:,0,:] # [M,3]
    v = global_coordinates - origin.unsqueeze(1) # [M,3,3]

    local_coordinates = torch.einsum('mna,mda->mnd', v, unit) # [M,3,3]

    return local_coordinates # [M, 3, 3]

def compute_s3_jacobian(coords, shell, device="cuda:0", dtype=torch.float32):
    """
    compute_s3_jacobian:
        coords:  shape [N, 3] (x,y,z of each node)
        shell:   shape [M, 3] (3 node IDs per triangular shell)
    Returns:
        J: shape [M, 2, 2], the partial derivatives:
           [ dx/dxi   dx/deta ]
           [ dy/dxi   dy/deta ]
    """
    coords = coords.to(device, dtype=dtype)
    shell  = shell.to(device)

    unit = compute_s3_local_unitvector(coords, shell, device=device) # [M,3,3]
    local_coordinates = compute_s3_global_to_local_coordinates(coords, shell, unit, device=device, dtype=dtype) # [M,3,3]

    dN_dxi  = torch.tensor([-1.0,  1.0,  0.0], device=device, dtype=dtype) 
    dN_deta = torch.tensor([-1.0,  0.0,  1.0], device=device, dtype=dtype)

    dN_dxi_  = dN_dxi.unsqueeze(0) # [1,3]
    dN_deta_ = dN_deta.unsqueeze(0) # [1,3]

    x = local_coordinates[...,0] # [M,3]
    y = local_coordinates[...,1] # [M,3]

    dx_dxi  = (dN_dxi_  * x).sum(dim=1)  # [M]
    dx_deta = (dN_deta_ * x).sum(dim=1)  # [M]
    dy_dxi  = (dN_dxi_  * y).sum(dim=1)  # [M]
    dy_deta = (dN_deta_ * y).sum(dim=1)  # [M]

    J = torch.stack([
        torch.stack([dx_dxi,  dx_deta], dim=-1),
        torch.stack([dy_dxi,  dy_deta], dim=-1),
    ], dim=1) # [M, 2, 2]

    return J

def compute_s3_shape_gradient(coords, shell, device="cuda:0", dtype=torch.float32):
    """
    Return the shape-fn gradients dN/dx, dN/dy for each element, [M,3,2].
    i.e. for each of M triangles, for each of 3 local nodes => (dN/dx, dN/dy).
    """
    J = compute_s3_jacobian(coords, shell, device=device, dtype=dtype) # [M,2,2]
    Jinv = torch.inverse(J)  # [M,2,2]

    dN_dxi  = torch.tensor([-1.0,  1.0,  0.0], device=device, dtype=dtype)
    dN_deta = torch.tensor([-1.0,  0.0, +1.0], device=device, dtype=dtype)
    dN_dxi_  = dN_dxi.unsqueeze(0)    # [1,3]
    dN_deta_ = dN_deta.unsqueeze(0)   # [1,3]

    big_dN_param = torch.stack([dN_dxi_, dN_deta_], dim=-1)  # [1,3,2]

    dN_xy = torch.einsum('mab,ncb->mca', Jinv, big_dN_param)
    return dN_xy

def compute_s3_B_matrix(coords, shell, device="cuda:0", dtype=torch.float32):
    """
    Returns the Kirchhoff B-matrix for each of M triangles at (xi, eta).
    Output shape: [M, 6, 18]
       6 strain components,  3 nodes x 6 dofs = 18 total columns.
       (We place 'drilling' rotation col=thz=0 for classical shells.)
    """
    M = shell.shape[0]

    dN_xy = compute_s3_shape_gradient(coords, shell,  device=device, dtype=dtype) # [M,3,2]
    dNdx = dN_xy[...,0]  # [M,3]
    dNdy = dN_xy[...,1]  # [M,3]

    def membrane_block_i(i):
        blk = torch.zeros(M,3,6, device=device, dtype=dtype)
        blk[:,0,0] = dNdx[:,i]
        blk[:,1,1] = dNdy[:,i]
        blk[:,2,0] = dNdy[:,i]
        blk[:,2,1] = dNdx[:,i]
        return blk  # [M,3,6]

    Bmem = torch.cat([membrane_block_i(i) for i in range(3)], dim=2)  # [M,3,18]

    def bending_block_i(i):
        blk = torch.zeros(M,3,6, device=device, dtype=dtype)
        blk[:,0,4] = - dNdx[:,i]
        blk[:,1,3] =   dNdy[:,i]
        blk[:,2,3] =   dNdy[:,i]
        blk[:,2,4] =   dNdx[:,i]
        return blk  # [M,3,6]

    Bbend = torch.cat([bending_block_i(i) for i in range(3)], dim=2)  # [M,3,18]

    B = torch.cat([Bmem, Bbend], dim=1)  # [M,6,18]
    return B

def compute_s3_K_matrix(coords, shell, membrane, bending, device="cuda:0", dtype=torch.float32):
    """
    Compute the Kirchhoff shell stiffness for each triangular element [M]
    """
    D = compute_kirchoff_D_matrix(membrane, bending, device=device, dtype=dtype) # [6,6]
    
    B = compute_s3_B_matrix(coords, shell, device=device, dtype=dtype) # [M,6,18]
    J = compute_s3_jacobian(coords, shell, device=device, dtype=dtype) # [M,2,2]
    detJ = torch.det(J)  # [M]

    K = torch.einsum('mai,ab,mbj->mij', B, D, B) # [M,18,18]
    K = K * detJ.unsqueeze(-1).unsqueeze(-1) * 0.5 # [M,18,18]

    return K

def compute_s3_shell_stress(coords, shell, membrane, bending, displacement, device="cuda:0", dtype=torch.float32):
    """
    Compute the Kirchhoff shell stress for each triangular element [M]
    """
    M = shell.shape[0]

    D = compute_kirchoff_D_matrix(membrane, bending, device=device, dtype=dtype) # [6,6]
    B = compute_s3_B_matrix(coords, shell, device=device, dtype=dtype) # [M,6,18]

    u = displacement[shell].reshape(M, -1)  # [M,18]
    strain = torch.bmm(B, u.unsqueeze(2)).squeeze(2)  # [M,6]
    stress = torch.matmul(strain, D.t())  # [M,6]
    
    return stress # [M,6]


##### s4 노드순서 및 shape function (linear)
'''
Node 0:  (xi, eta) = (-1, -1)
Node 1:  (xi, eta) = (1, -1) 
Node 2:  (xi, eta) = (1, 1) 
Node 3:  (xi, eta) = (-1, 1)

N_0 = (1 - ξ)(1 - η)/4
N_1 = (1 + ξ)(1 - η)/4
N_2 = (1 + ξ)(1 + η)/4
N_3 = (1 - ξ)(1 + η)/4 
'''
def compute_s4_normal(coords, shell, device="cuda:0"):
    """
    사각형 면적 구하기

    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        shell (torch.Tensor): Shell connectivity [M, 4]
        
    Output:
        Normal (torch.Tensor): Shell Normal [M, 3]
    """
    coords = coords.to(device)
    shell = shell.to(device)

    a = coords[shell[:, 1]] - coords[shell[:, 0]]
    b = coords[shell[:, 3]] - coords[shell[:, 0]]
    
    normal = torch.cross(a, b, dim=1)

    return normal

def identify_s4_shared_edges(shell, device="cuda:0"):
    """
    Shell 정보가 주어지면 각 변이 공유되는 shell 정보를 구하는 함수
    아웃풋은 [S, 2, 2]이며, 총 S개의 변에 대해 [2, 2] 연결 정보를 제공한다.
    [2, 2]:[[shell, edge index],[shell id, edge index]]

    Inputs:
        shell (torch.Tensor): Shell connectivity [M, 4] 

    Outputs:
        shared_edge_indices (torch.Tensor): Indices of shells sharing each edge [S, 2, 2]
    """
    M = shell.shape[0]
    shell = shell.to(device)
    
    edge_node_indices = torch.tensor([
        [0, 1],  # Edge 0
        [1, 2],  # Edge 1
        [2, 3],  # Edge 2
        [3, 0]   # Edge 3
    ], device=device)  # [4, 2]
    
    edge = shell[:, edge_node_indices]  # [M, 4, 2]
    edges_sorted, _ = torch.sort(edge, dim=2)  # [M, 4, 2]
    edges_flat = edges_sorted.view(-1, 2)  # [M * 4, 2]
    
    shell_ids = torch.arange(M, device=device).repeat_interleave(4)  # [M*4]
    edge_indices = torch.tile(torch.arange(4, device=device), (M,))  # [M*4]
    
    _, inverse_indices, counts = torch.unique(
        edges_flat, return_inverse=True, return_counts=True, dim=0
    )
    
    shared_mask = counts == 2
    shared_edge_ids = torch.nonzero(shared_mask, as_tuple=True)[0]  # [S]
    
    if shared_edge_ids.numel() == 0:
        return torch.empty((0, 2, 2), dtype=torch.long, device=device)
    
    sorted_inverse, sorted_order = torch.sort(inverse_indices)
    sorted_shell_ids = shell_ids[sorted_order]
    sorted_edge_indices = edge_indices[sorted_order]
    
    positions = torch.searchsorted(sorted_inverse, shared_edge_ids)
    
    shell1 = sorted_shell_ids[positions]
    edge1 = sorted_edge_indices[positions]
    shell2 = sorted_shell_ids[positions + 1]
    edge2 = sorted_edge_indices[positions + 1]
    
    shared_edge_indices = torch.stack([
        torch.stack([shell1, edge1], dim=1),
        torch.stack([shell2, edge2], dim=1)
    ], dim=1)  # [S, 2, 2]
    
    return shared_edge_indices  # [S, 2, 2]

def compute_square_surface_faces_with_fourth_node(shell, device="cuda:0"):
    '''
    사각형 정보가 주어지면 1번만 등장하는 변(테두리)을 구하고, 이러한 테두리 사각형의 테두리가 아닌 4번째 노드 번호 또한 구하는 함수
    4번째 노드 정보는 바같방향을 구하기 위해 사용된다.
    
    Input:
        shell (torch.Tensor): Shell connectivity [M, 4]
    
    Returns:
        surfaces (torch.Tensor): surface connectivity [K, 2]
        4th id (torch.Tensor): 4th node id [K]
    '''
    shell = shell.to(device)

    edges = torch.cat([
        shell[:, [0, 1]],  # edge between node 0 and node 1
        shell[:, [1, 2]],  # edge between node 1 and node 2
        shell[:, [2, 3]],  # edge between node 2 and node 3
        shell[:, [3, 0]],  # edge between node 3 and node 0
    ], dim=0)  # [4*M, 2]

    fourth_nodes = torch.cat([
        shell[:, 3],  # fourth node for edge [0, 1]
        shell[:, 0],  # fourth node for edge [1, 2]
        shell[:, 1],  # fourth node for edge [2, 3]
        shell[:, 2],  # fourth node for edge [3, 0]
    ], dim=0)  # [4*M]

    sorted_edges, _ = torch.sort(edges, dim=1)

    _, inverse_indices, counts = torch.unique(sorted_edges, dim=0, return_inverse=True, return_counts=True)

    surface_edge_mask = counts[inverse_indices] == 1
    surface_edges = edges[surface_edge_mask]
    surface_fourth_nodes = fourth_nodes[surface_edge_mask]

    return surface_edges, surface_fourth_nodes  # [K, 2], [K]

def compute_s4_local_unitvector(coords, shell, device="cuda:0"):
    '''
    사각형의 로컬 좌표계를 구하는 함수
    
    Input:
        coords (torch.Tensor): Node coordinates [N, 3]
        shell (torch.Tensor): Shell connectivity [M, 4]
    
    Output:
        unit (torch.Tensor): Local unit vector [M, 3, 3]
    '''
    coords = coords.to(device)
    shell = shell.to(device)

    a = coords[shell[:, 1]] - coords[shell[:, 0]] # [M,3]
    b = coords[shell[:, 3]] - coords[shell[:, 0]] # [M,3]

    a = a / torch.norm(a, dim=1, keepdim=True) # [M,3]
    b = b - (torch.sum(a * b, dim=1, keepdim=True) / torch.sum(a * a, dim=1, keepdim=True)) * a  # Make b perpendicular to a
    b = b / torch.norm(b, dim=1, keepdim=True) # [M,3]
    c = torch.cross(a, b, dim=1) # [M,3]

    unit = torch.stack([a, b, c], dim=1) # [M,3,3]

    return unit

def compute_s4_global_to_local_coordinates(coords, shell, unit, device="cuda:0", dtype=torch.float32):
    '''
    Global 좌표계를 로컬 좌표계로 변환하는 함수
    
    Input:
        shell (torch.Tensor): Shell connectivity [M, 4]
        target (torch.Tensor): Global coordinates [N, 3]
        unit (torch.Tensor): Local unit vector [M, 3, 3]
    
    Output:
        local (torch.Tensor): Local coordinates [M, 4, 3]
    '''
    coords = coords.to(device, dtype=dtype)
    shell = shell.to(device)
    unit = unit.to(device)

    M = shell.shape[0]

    global_coordinates = coords[shell] # [M,4,3]
    origin = global_coordinates[:,0,:] # [M,3]
    v = global_coordinates - origin.unsqueeze(1) # [M,4,3]

    local_coordinates = torch.einsum('mna,mda->mnd', v, unit) # [M,4,3]

    return local_coordinates # [M, 4, 3]

def s4_integration_points(device="cuda:0"):
    """
    Return the 4 Gauss points and weights for a 2x2 integration.
    Points in [-1/sqrt(3), +1/sqrt(3)], weights = 1.0 each.
    Shape of points: (4, 2)
    """
    sqrt3 = 1/np.sqrt(3)
    s4_points = torch.tensor([
        [sqrt3, -sqrt3],
        [sqrt3, sqrt3],
        [-sqrt3, sqrt3],
        [-sqrt3, -sqrt3]
    ], dtype=torch.float32)

    s4_weights = torch.tensor([
        1.0,  
        1.0,  
        1.0,
        1.0
    ], dtype=torch.float32)

    return s4_points.to(device), s4_weights.to(device)

def compute_s4_jacobian(coords, shell, xi, eta, device="cuda:0", dtype=torch.float32):
    """
    coords:  shape [N, 3]  (x,y,z for each of N global nodes)
    shell: shape [M, 4]  (4 node IDs for each of M shell)
    xi, eta: scalar float in [-1,1], the param coords for which we want the Jacobian
    
    Returns: J of shape [M, 2, 2] 
       the 2x2 'in-plane' partial derivatives:
         [ dx/dxi   dx/deta ]
         [ dy/dxi   dy/deta ]
       (We often only keep x,y for a shell mid-surface approach.)
    """
    coords = coords.to(device)
    shell = shell.to(device)

    unit = compute_s4_local_unitvector(coords, shell, device=device) # [M,3,3]
    local_coordinates = compute_s4_global_to_local_coordinates(coords, shell, unit, device=device, dtype=dtype) # [M,4,3]

    dN_dxi = 0.25 * torch.tensor([
        -(1-eta),  # dN1/dxi
         (1-eta),  # dN2/dxi
         (1+eta),  # dN3/dxi
        -(1+eta)   # dN4/dxi
    ], device=device, dtype=dtype)

    dN_deta = 0.25 * torch.tensor([
        -(1 - xi), 
        -(1 + xi),
         (1 + xi),
         (1 - xi)
    ], device=device, dtype=dtype)
    
    dN_dxi_ = dN_dxi.unsqueeze(0)   # [1,4]
    dN_deta_ = dN_deta.unsqueeze(0) # [1,4]

    x = local_coordinates[...,0] 
    y = local_coordinates[...,1] 

    dx_dxi   = (dN_dxi_   * x).sum(dim=1)
    dx_deta  = (dN_deta_  * x).sum(dim=1)
    dy_dxi   = (dN_dxi_   * y).sum(dim=1)
    dy_deta  = (dN_deta_  * y).sum(dim=1)

    J = torch.stack([
        torch.stack([dx_dxi,  dx_deta], dim=-1),
        torch.stack([dy_dxi,  dy_deta], dim=-1),
    ], dim=1)  # [M, 2, 2]
    return J

def compute_s4_shape_gradient(coords, shell, xi, eta, device="cuda:0", dtype=torch.float32):
    """
    Return the shape-fn gradients dN/dx, dN/dy for each element, all 4 nodes.
    Output shape: [M, 4, 2], i.e. for each of M shell,
      for each of 4 local nodes, the partial derivatives (dN/dx, dN/dy).
    """
    J = compute_s4_jacobian(coords, shell, xi, eta, device=device, dtype=dtype)  
    Jinv = torch.inverse(J)  # [M,2,2]

    dN_dxi = 0.25 * torch.tensor([
        -(1-eta), (1-eta), (1+eta), -(1+eta)
    ], device=device, dtype=dtype)
    dN_deta = 0.25 * torch.tensor([
        -(1 - xi), -(1 + xi), (1 + xi), (1 - xi)
    ], device=device, dtype=dtype)

    dN_dxi_ = dN_dxi.unsqueeze(0) # [1,4]
    dN_deta_ = dN_deta.unsqueeze(0) # [1,4]

    big_dN_param = torch.stack([dN_dxi_.expand(-1,4), dN_deta_.expand(-1,4)], dim=-1) # [1,4,2]

    dN_xy = torch.einsum('mab, ncb -> mca', Jinv, big_dN_param) # [M,4,2]

    return dN_xy

def compute_s4_B_matrix_single(coords, shell, xi, eta, device="cuda:0", dtype=torch.float32):
    """
    Returns the Kirchhoff B-matrix for each of M shell at (xi, eta).
    Output shape: [M, 6, 24]
       6 strain components, 24 = 4 nodes x 6 dofs (u,v,w,thx,thy,thz). 
       We typically ignore the 'drilling' rotation (thz) in classical shells,
       but let's just place a zero block for it.

    *not* integrate; this is just the B at a single integration pt.
    """
    M = shell.shape[0]

    dN_xy = compute_s4_shape_gradient(coords, shell, xi, eta, device=device, dtype=dtype)
    dNdx = dN_xy[...,0]  # [M,4]
    dNdy = dN_xy[...,1]  # [M,4]

    # B_membrane => shape [3, 24]
    # B_bending  => shape [3, 24]

    #--- Membrane part (epsilon_xx^m, epsilon_yy^m, gamma_xy^m) ---
    # row0 = [ dNdx, 0, 0, 0, 0, 0 ]         => epsilon_xx^m depends on u
    # row1 = [ 0, dNdy, 0, 0, 0, 0 ]         => epsilon_yy^m depends on v
    # row2 = [ dNdy, dNdx, 0, 0, 0, 0 ]      => gamma_xy^m depends on (u, v)
    
    def membrane_block_i(i):
        block = torch.zeros(M, 3, 6, device=device, dtype=dtype)
        block[:,0,0] = dNdx[:,i]  # row=0, col=0 => dNdx * u
        block[:,1,1] = dNdy[:,i]  # row=1, col=1 => dNdy * v
        block[:,2,0] = dNdy[:,i]  # row=2, col=0 => gamma_xy wrt u
        block[:,2,1] = dNdx[:,i]  # row=2, col=1 => gamma_xy wrt v
        return block # [M,3,6]
    
    Bmem = torch.cat([membrane_block_i(i) for i in range(4)], dim=2)  # [M,3,24]

    #--- Bending part (kappa_x, kappa_y, kappa_xy) ---
    # row kappa_x   => [0, 0, 0, 0, -dNdx, 0] for each node's (u,v,w,thx,thy,thz)
    # row kappa_y   => [0, 0, 0, dNdy, 0, 0]
    # row kappa_xy  => [0, 0, 0, dNdy, dNdx, 0]
    # sign for kappa_x is negative in many references: -dNdx for Theta_y.

    def bending_block_i(i):
        block = torch.zeros(M, 3, 6, device=device, dtype=dtype)
        block[:,0,4] = -dNdx[:,i]
        block[:,1,3] = dNdy[:,i]
        block[:,2,3] = dNdy[:,i]   
        block[:,2,4] = dNdx[:,i]   
        return block
    
    Bbend = torch.cat([bending_block_i(i) for i in range(4)], dim=2)  # [M,3,24]

    B = torch.cat([Bmem, Bbend], dim=1) # [M,6,24]

    return B

def compute_s4_B_matrix(coords, shell, integration_points=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Returns the Kirchhoff B-matrix for each of M shell at all integration points.
    Output shape: [M, 6, 24, 4]
       6 strain components, 24 = 4 nodes x 6 dofs (u,v,w,thx,thy,thz), 4 integration points.
       We typically ignore the 'drilling' rotation (thz) in classical shells,
       but let's just place a zero block for it.

    """
    if integration_points is None:
        integration_points, weights = s4_integration_points(device=device)

    for i, (xi, eta) in enumerate(integration_points):
        B = compute_s4_B_matrix_single(coords, shell, xi, eta, device=device, dtype=dtype)
        if i == 0:
            B_all = B.unsqueeze(-1) * weights[i]
        else:
            B_all = torch.cat([B_all, B.unsqueeze(-1)*weights[i]], dim=-1)
    if single: 
        B_all = B_all.sum(dim=-1)
    
    return B_all

def compute_s4_K_matrix(coords, shell, membrane, bending, integration_points=None, single=True, device="cuda:0", dtype=torch.float32):
    """
    Compute the Kirchhoff shell stiffness for each element [M], 
    either:
      single=True => [M,24,24] (summed over gauss points),
      single=False => [M,24,24,4].
    """
    D = compute_kirchoff_D_matrix(membrane, bending, device=device, dtype=dtype) # [6,6]

    if integration_points is None:
        gpts, gws = s4_integration_points(device=device)
    else:
        gpts, gws = integration_points
    K_list = []

    for f in range(gpts.shape[0]):
        xi  = gpts[f,0].item()
        eta = gpts[f,1].item()
        w_f = gws[f]

        Bf = compute_s4_B_matrix_single(coords, shell, xi, eta, device=device, dtype=dtype) # [M,6,24]
        Jf = compute_s4_jacobian(coords, shell, xi, eta, device=device, dtype=dtype) # [M,2,2]
        detJf = torch.det(Jf)  # [M]

        Kf = torch.einsum('mai,ab,mbj->mij', Bf, D, Bf) # [M,24,24]
        Kf = Kf * detJf.unsqueeze(-1).unsqueeze(-1) * w_f # [M,24,24]

        K_list.append(Kf.unsqueeze(-1))  # => [M,24,24,1]

    K_all = torch.cat(K_list, dim=-1)  # [M,24,24,4]

    if single:
        K_final = K_all.sum(dim=-1)
    else:
        K_final = K_all

    return K_final

def compute_s4_shell_stress(coords, shell, membrane, bending, displacement, device="cuda:0", dtype=torch.float32):
    """
    Compute the Kirchhoff shell stress for each element [M]
    """
    M = shell.shape[0]

    D = compute_kirchoff_D_matrix(membrane, bending, device=device, dtype=dtype) # [6,6]
    B = compute_s4_B_matrix(coords, shell, device=device, dtype=dtype) # [M,6,24]

    u = displacement[shell].reshape(M, -1)  # [M,24]
    strain = torch.bmm(B, u.unsqueeze(2)).squeeze(2)  # [M,6]
    stress = torch.matmul(strain, D.t())  # [M,6]
    
    return stress # [M,6]




#################################################################################################
#########################################   to Wedge   ##########################################
#################################################################################################

def shell_extrude(coords, tri, quad, thickness, device="cuda:0", dtype=torch.float32):
    """
    This function combines all steps: computing node normals, extruding a shell,
    and building 3D volume connectivity from an input mid-surface mesh.
    """

    def compute_node_normals(coords, tri, quad, eps=1e-8, device="cuda:0", dtype=torch.float32):
        """
        Compute per-node normals from triangle and quad connectivity.

        Parameters:
          coords: (N, 3) tensor of node coordinates.
          tri: (T, 3) tensor (dtype torch.long) of triangle connectivity.
          quad: (S, 4) tensor (dtype torch.long) of quad connectivity.

        Returns:
          node_normals: (N, 3) tensor of unit normals computed at each node.
        """
        N = coords.shape[0]

        tri_coords = coords[tri]  
        v1 = tri_coords[:, 1] - tri_coords[:, 0]  # (T,3)
        v2 = tri_coords[:, 2] - tri_coords[:, 0]  # (T,3)
        n_tri = torch.cross(v1, v2, dim=1)          # (T,3)
        n_tri = n_tri / (n_tri.norm(dim=1, keepdim=True) + eps)  # Normalize

        quad_coords = coords[quad]  # (S,4,3)
        v1q = quad_coords[:, 1] - quad_coords[:, 0]
        v2q = quad_coords[:, 2] - quad_coords[:, 0]
        n_quad1 = torch.cross(v1q, v2q, dim=1)
        n_quad1 = n_quad1 / (n_quad1.norm(dim=1, keepdim=True) + eps)
        v1q2 = quad_coords[:, 2] - quad_coords[:, 0]
        v2q2 = quad_coords[:, 3] - quad_coords[:, 0]
        n_quad2 = torch.cross(v1q2, v2q2, dim=1)
        n_quad2 = n_quad2 / (n_quad2.norm(dim=1, keepdim=True) + eps)

        node_normals = torch.zeros((N, 3), dtype=dtype, device=device)
        counts = torch.zeros(N, dtype=dtype, device=device)

        tri_flat = tri.reshape(-1)  # (3*T,)
        n_tri_rep = n_tri.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)  # (3*T,3)
        node_normals.index_add_(0, tri_flat, n_tri_rep)
        counts.index_add_(0, tri_flat, torch.ones_like(tri_flat, dtype=dtype))

        quad_tri1 = quad[:, :3].reshape(-1)  
        n_quad1_rep = n_quad1.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        node_normals.index_add_(0, quad_tri1, n_quad1_rep)
        counts.index_add_(0, quad_tri1, torch.ones_like(quad_tri1, dtype=dtype))
        quad_tri2 = quad[:, [0, 2, 3]].reshape(-1)
        n_quad2_rep = n_quad2.unsqueeze(1).expand(-1, 3, -1).reshape(-1, 3)
        node_normals.index_add_(0, quad_tri2, n_quad2_rep)
        counts.index_add_(0, quad_tri2, torch.ones_like(quad_tri2, dtype=dtype))

        node_normals = node_normals / (counts.unsqueeze(1) + eps)
        node_normals = node_normals / (node_normals.norm(dim=1, keepdim=True) + eps)

        return node_normals

    def extrude_shell(coords, node_normals, thickness):
        """
        Given mid-surface coordinates and per-node normals, extrude to form 
        top and bottom layers.

        Parameters:
          coords: (N,3) tensor of mid-surface coordinates.
          node_normals: (N,3) tensor of per-node unit normals.
          thickness: scalar, total thickness to extrude.

        Returns:
          coords_3d: (2*N,3) tensor with bottom nodes followed by top nodes.
        """
        coords_bottom = coords - 0.5 * thickness * node_normals
        coords_top    = coords + 0.5 * thickness * node_normals
        coords_3d = torch.cat([coords_bottom, coords_top], dim=0)
        return coords_3d

    def build_volume_connectivity(tri, quad, N):
        """
        Build volume connectivity for wedges (from triangles) and hexahedra (from quads).

        Parameters:
          tri: (T,3) tensor of triangle connectivity.
          quad: (S,4) tensor of quad connectivity.
          N: number of original nodes.

        Returns:
          wedge_elems: (T,6) tensor for wedge elements (c3d6).
          hexa_elems: (S,8) tensor for hexahedral elements (c3d8).
        """
        wedge_elems = torch.cat([tri, tri + N], dim=1)  # (T, 6)
        hexa_elems = torch.cat([quad, quad + N], dim=1)   # (S, 8)
        return wedge_elems, hexa_elems

    coords = coords.to(device, dtype=dtype)

    node_normals = compute_node_normals(coords, tri, quad, device=device, dtype=dtype)
    coords_3d = extrude_shell(coords, node_normals, thickness)
    wedge_elems, hexa_elems = build_volume_connectivity(tri, quad, coords.shape[0])

    return coords_3d, wedge_elems, hexa_elems