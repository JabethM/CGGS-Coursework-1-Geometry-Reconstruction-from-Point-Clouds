import os
import numpy as np
from scipy import spatial
from scipy import linalg
from scipy.sparse import csr_matrix, hstack, vstack


def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces

def compute_Q(l, coordinates):
    length = np.arange(l + 1)

    X, Y, Z = np.where(np.add.outer(np.add.outer(length, length), length) <= l)

    coords = coordinates[..., np.newaxis]

    powers = np.power(coords, np.array([X,Y,Z]))
    Q = np.prod(powers,axis=1)
    m = X.size
    return Q, m

def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=np.array([]), useOffPoints=True,
                        sparsify=False, l=-1):
    
    n = inputPoints.shape[0]

    if RBFCentreIndices.any():
        points = inputPoints[RBFCentreIndices]
        normals = inputNormals[RBFCentreIndices]
    else:
        points = inputPoints
        normals = inputNormals

    if useOffPoints:
        epsilon_Plus = points + epsilon * normals
        epsilon_Minus = points - epsilon * normals
        RBFCentres = np.vstack((points, epsilon_Plus, epsilon_Minus))
        d_vector = np.hstack((np.zeros(n), np.ones(n) * epsilon, np.ones(n) * (- epsilon)))
    else:
        RBFCentres = points
        d_vector = np.zeros(n)

    if sparsify:
        solution_vector= sparse_RBF(RBFCentres, RBFFunction, d_vector, l)

    else:
        r_matrix = spatial.distance.cdist(RBFCentres, RBFCentres)
        A_matrix = RBFFunction(r_matrix)

        if l < 0:
            LHS = A_matrix
            RHS = d_vector
        else:
            Q, m = compute_Q(l, RBFCentres)

            a = np.zeros(m)

            LHS = np.block([[A_matrix, Q],
                            [Q.T, np.zeros((m,m))]])
            RHS = np.hstack((d_vector, a))

        
        lu, piv = linalg.lu_factor(LHS, overwrite_a=True, check_finite=False)
        solution_vector = linalg.lu_solve((lu, piv), RHS, overwrite_b=True, check_finite=False)
        
    weights = solution_vector[:3*n]
    a = solution_vector[3*n:]

    return weights, RBFCentres, a


def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):
    num_of_points = np.shape(xyz)[0]

    r_matrix = spatial.distance.cdist(xyz, centres)
    A_matrix = RBFFunction(r_matrix)

    if l < 0:
        left_product = A_matrix
        right_product = w
    else:
        Q, _ = compute_Q(l, xyz)
        
        left_product = np.block([[A_matrix, Q]])
        right_product = np.hstack((w, a))

    values = np.dot(left_product, right_product)
    return values[:num_of_points]

def sparse_RBF(RBFCentres, RBFFunction, d_vector, l=-1):
    # TODO: Compute sparse matrix before computing dense matrix
    dense = spatial.distance.cdist(RBFCentres, RBFCentres)
    dense_A = RBFFunction(dense)
    sparse_A = csr_matrix(dense_A)
    
    if l < 0:
        LHS = sparse_A
        RHS = d_vector
    else:
        Q, m = compute_Q(l, RBFCentres)
        a = np.zeros(m)

        first_row = hstack([sparse_A, Q])
        second_row = hstack([Q.T, np.zeros((m, m))])
        LHS = vstack([first_row, second_row])
        
        RHS = hstack((d_vector, a))
    
    lu = linalg.splu(LHS)
    solution_vector = lu.solve(RHS)
    return solution_vector.toarray()

def wendland(b, r):
    return 1 / 12 * np.power(np.maximum((1 - b * r), 0), 3) * (1 - 3 * b * r)


def biharmonic(r):
    return r


def polyharmonic(r):
    return r ** 3

