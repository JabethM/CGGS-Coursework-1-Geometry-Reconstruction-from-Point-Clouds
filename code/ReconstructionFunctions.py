import os
import numpy as np


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

    differences = RBFCentres[:, np.newaxis] - RBFCentres
    r_matrix = np.linalg.norm(differences, axis=2)
    F_matrix = RBFFunction(r_matrix)

    n = inputPoints.shape[0]
    d_vector = np.hstack((np.zeros(n), np.ones(n) * epsilon, np.ones(n) * (- epsilon)))
    weights = np.linalg.solve(F_matrix, d_vector)
    a = []  # polynomial coefficients (for Section 2)

    return weights, RBFCentres, a


def evaluate_RBF(xyz, centres, RBFFunction, w, l=-1, a=[]):
    values = np.zeros(xyz.shape[0])

    differences = xyz[:, np.newaxis] - centres
    r_matrix = np.linalg.norm(differences, axis=2)
    base_results = RBFFunction(r_matrix)
    expanded_implicit = base_results * w

    values = np.sum(expanded_implicit, axis=1)

    ###Complate RBF evaluation here
    return values


def wendland(b, r):
    return 1 / 12 * (1 - b * r) * (1 - 3 * b * r)


def biharmonic(r):
    return r


def polyharmonic(r):
    return r ** 3
