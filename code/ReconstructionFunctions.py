import os
import numpy as np
from scipy import spatial
from scipy import linalg
from scipy.sparse import csr_matrix, coo_matrix, csc_matrix, hstack, vstack
from scipy.sparse import linalg as splinalg
import time
import resource

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

    powers = np.power(coords, np.array([X, Y, Z]))
    Q = np.prod(powers, axis=1)
    m = X.size
    return Q, m


def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=np.array([]),
                        useOffPoints=True,
                        sparsify=False, l=-1):
    
    epsilon_Plus = inputPoints + epsilon * inputNormals    
    epsilon_Minus = inputPoints - epsilon * inputNormals
    all_centres = np.vstack((inputPoints, epsilon_Plus, epsilon_Minus))
    
    overdetermined = False
    if RBFCentreIndices.any():
        overdetermined = True
        centre_points = inputPoints[RBFCentreIndices]
        centre_normals = inputNormals[RBFCentreIndices]
    else:
        centre_points= inputPoints
        centre_normals = inputNormals

    N3 = all_centres.shape[0]

    if useOffPoints:
        N1 = N3 // 3
        epsilon_Plus = centre_points + epsilon * centre_normals
        epsilon_Minus = centre_points - epsilon * centre_normals
        RBFCentres = np.vstack((centre_points, epsilon_Plus, epsilon_Minus))
        d_vector = np.hstack((np.zeros(N1), np.ones(N1) * epsilon, np.ones(N1) * (- epsilon)))

    else:
        overdetermined = True
        RBFCentres = centre_points
        d_vector = np.zeros(N3)  
    
    M = RBFCentres.shape[0]
    
    
    del epsilon_Plus
    del epsilon_Minus
    del inputNormals
    del centre_normals

    if sparsify:
        solution_vector = sparse_RBF(RBFCentres, RBFFunction, epsilon, d_vector, l)

    else:
        A_matrix = spatial.distance.cdist(all_centres, RBFCentres)
        A_matrix = RBFFunction(A_matrix)

        if l < 0:
            LHS = A_matrix
            RHS = d_vector
        else:
            Q, m = compute_Q(l, RBFCentres)

            a = np.zeros(m)

            LHS = np.block([[A_matrix, Q],
                            [Q.T, np.zeros((m, m))]])
            RHS = np.hstack((d_vector, a))

        if overdetermined:
            solution_vector = linalg.lstsq(LHS, RHS)[0]
        else:
            lu, piv = linalg.lu_factor(LHS, overwrite_a=True, check_finite=False)
            solution_vector = linalg.lu_solve((lu, piv), RHS, overwrite_b=True, check_finite=False)
 
    weights = solution_vector[:M]
    a = solution_vector[M:]

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


def sparse_RBF(RBFCentres, RBFFunction, epsilon, d_vector, l=-1):
    # On my machine the following is slower and more memory intensive than using scipy functions to calculate dense first
    sparse_A = compute_sparse_A(RBFCentres, RBFFunction, epsilon)

    dense = spatial.distance.cdist(RBFCentres, RBFCentres)
    sparse_A = csr_matrix(RBFFunction(dense))

    

    if l < 0:
        LHS = sparse_A
        RHS = d_vector
    else:
        Q, m = compute_Q(l, RBFCentres)
        a = np.zeros(m)
        zero_array = csc_matrix((m,m), dtype=float)
        first_row = hstack([sparse_A, Q])
        second_row = hstack([Q.T, np.zeros((m, m))])
        LHS = vstack([first_row, second_row])

        RHS = hstack((d_vector, a))

    lu = splinalg.splu(LHS)
    solution_vector = lu.solve(RHS)
    return solution_vector


def compute_sparse_A(RBFCentres, RBFFunction, threshold):

    size = np.shape(RBFCentres)[0]
    sq_differences = spatial.distance.cdist(RBFCentres, RBFCentres)

    sq_differences[np.tril_indices(size, k=0)] = 0
    
    boundary = np.inf
    for i in range(size//2):
        row = sq_differences[i][i+1:]

        threshold_idx = binary_search(row, RBFFunction, threshold)
        bound = sq_differences[i][threshold_idx + i + 1]
        if bound < boundary:
            boundary = bound
    
    row_indices, col_indices = np.where(sq_differences <= boundary)
    data = sq_differences[row_indices, col_indices]

    del sq_differences
    del row
    del boundary
    del threshold_idx
    del threshold

    data = RBFFunction(data)

    full_data = np.tile(data, 2)
    full_row_indices = np.concatenate((row_indices, col_indices))
    full_col_indices = np.concatenate((col_indices, row_indices))

    del row_indices
    del col_indices

    num_rows = size
    return coo_matrix((full_data, (full_row_indices, full_col_indices)), shape=(num_rows, num_rows))



def binary_search(arr, RBFFunction, threshold):
    order = np.argsort(arr)
    left = 0
    right = len(arr) - 1

    if RBFFunction(arr[order[right]]) > threshold:
        return order[right]

    while left < right:
        mid = (left + right) // 2
        mid_val = RBFFunction(arr[order[mid]])
        if mid_val < threshold:
            right = mid - 1
        elif mid_val > threshold:
            left = mid + 1
    return order[left]



def wendland(b, r):
    return 1 / 12 * np.power(np.maximum((1 - b * r), 0), 3) * (1 - 3 * b * r)


def biharmonic(r):
    return r


def polyharmonic(r):
    return r ** 3
