import os
import numpy as np
from scipy import spatial
from scipy import linalg

def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces


def compute_RBF_weights(inputPoints, inputNormals, RBFFunction, epsilon, RBFCentreIndices=[], useOffPoints=True,
                        sparsify=False, l=-1):
    
    point_plus = inputPoints + epsilon * inputNormals
    point_minus = inputPoints - epsilon * inputNormals
    RBFCentres = np.vstack((inputPoints, point_plus, point_minus))

    r_matrix = spatial.distance.cdist(RBFCentres, RBFCentres)
    F_matrix = RBFFunction(r_matrix)

    n = inputPoints.shape[0]
    d_vector = np.hstack((np.zeros(n), np.ones(n) * epsilon, np.ones(n) * (- epsilon)))
    if l >= 0:
        length = np.arange(l + 1)
        X, Y, Z = np.where(np.add.outer(np.add.outer(length, length), length) <= l)

        coords = RBFCentres[..., np.newaxis]

        powers = np.power(coords, np.array([X,Y,Z]))
        Q = np.prod(powers,axis=1)

        m = X.size
        zero_vector = np.zeros(m)

        LHS = np.block([[F_matrix, Q],
                        [Q.T, np.zeros((m,m))]])
        RHS = np.hstack((d_vector, zero_vector))
        a = 2
    else:
        LHS = F_matrix
        RHS = d_vector

    
    lu, piv = linalg.lu_factor(LHS, overwrite_a=True, check_finite=False)
    solution_vector = linalg.lu_solve((lu, piv), RHS, overwrite_b=True, check_finite=False)
    
    weights = solution_vector[:3*n]
    a = solution_vector[3*n:]
    return weights, RBFCentres, a


def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):

    r_matrix = spatial.distance.cdist(xyz, centres)

    base_results = RBFFunction(r_matrix)
    values = np.dot(base_results, w)

    ###Complate RBF evaluation here
    return values


def wendland(b, r):
    return 1 / 12 * (1 - b * r) * (1 - 3 * b * r)


def biharmonic(r):
    return r


def polyharmonic(r):
    return r ** 3
