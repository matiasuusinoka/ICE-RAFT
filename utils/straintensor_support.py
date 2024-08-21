import numpy as np

def gradshape(xi):
    x,y = tuple(xi)
    return .25 * np.array([[-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)],[-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]])

def kinematic_matrix(m, scaling_factor):
    
    # Length scale of the cells in meters.
    l = scaling_factor * (m-1)
    
    # Vertices of the quadrilateral shape in global coordinates. Since there is only two coordinates (x,y) for each
    # point, we can use the weight factor of +- 1 / sqrt(3) based on the rules of Gaussian quadrature.
    q4 = [[x/np.sqrt(3.0),y/np.sqrt(3.0)] for y in [-1.0,1.0] for x in [-1.0,1.0]]
    
    # Defining the natural coordinates. We consider all deformable cells separately and thus simplify the natural
    # coordinates to the lenght scale of the cells.
    xIe = np.array([[0,l,l,0],[0,0,-l,-l]]).reshape(4,2)
    
    B = np.zeros((3,8))

    # Estimating parts of B matrix (B(m) = M = J^-1 dN). The strains are estimated with B(m) \times U.
    for q in q4:
        dN = gradshape(q)           # Gradient of shape functions
        J  = np.dot(dN, xIe).T      # Jacobian matrix  J = dNx
        dN = np.dot(np.linalg.inv(J), dN)
        B[0,0::2], B[1,1::2], B[2,0::2], B[2,1::2] = dN[0,:], dN[1,:], dN[1,:], dN[0,:]

    return B

def DeformationGradient_quadrilateral(u_i,B):

    u_i = np.array([u_i[:,0,0], u_i[:,0,1], u_i[:,1,1], u_i[:,1,0]]).reshape(8,1)
    dudx, dvdy, dudy_dvdx = B@u_i

    return np.array([[dudx[0],dudy_dvdx[0]],[dvdy[0],dudy_dvdx[0]]]) + np.identity(2)

# Transforming Green-Lagrangian strain back to rotation-corrected infinitesimal strains
def infinit_rot(E):
    erot_x = np.sqrt(1+2*E[0,0])-1
    erot_y = np.sqrt(1+2*E[1,1])-1
    erot_xy = .5*np.arcsin(2*E[0,1]/np.sqrt((1+2*E[0,0])*(1+2*E[1,1])))
    epsilon_rot = [[erot_x,erot_xy],[erot_xy,erot_y]]-np.identity(2)
    return epsilon_rot

def tensor_formulation(F, rotation_fixed = True):
    
    # For rotation_corrected infinitesimal strains
    if rotation_fixed == True:
        E = .5*(F.transpose(0,1,3,2) @ F - np.identity(2))
        epsilon = np.array([[infinit_rot(E[x,y]) for y in range(E.shape[0])] for x in range(E.shape[1])])
    
    # For general infinitesimal strains
    else: epsilon = .5*(F+F.transpose(0,1,3,2))-np.identity(2)
        
    return epsilon

# Calculating the areas of polygons / quadrilateral elements
def PolyArea(coordinates):
    x, y = (np.array([measure[0,0],measure[0,-1],measure[-1,-1],measure[-1,0]]) for measure in [coordinates[0],coordinates[1]])
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# Extracting deformation estimates below the velocity threshold defined in Uusinoka et al. (2024)
def velocity_filter(u_i): 
    if any(vert > .3 for vert in [u_i[0,0], u_i[0,1], u_i[1,1], u_i[1,0]]): return True
    return False
