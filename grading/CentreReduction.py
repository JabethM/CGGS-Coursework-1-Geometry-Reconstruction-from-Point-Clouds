import os
import sys
import pickle
import numpy as np

sys.path.append(os.path.join('..', 'code'))
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF, wendland, biharmonic, polyharmonic

if __name__ == '__main__':

    data_path = os.path.join('..', 'data')  # Replace with the path to your folder

    epsilon = 1e-3

    # Get a list of all files in the folder with the ".off" extension
    off_files = [file for file in os.listdir(data_path) if file.endswith(".off")]

    for currFileIndex in range(len(off_files)):
        print("Processing mesh ", off_files[currFileIndex])
        off_file_path = os.path.join(data_path, off_files[currFileIndex])

        inputPointNormals, _ = load_off_file(off_file_path)
        inputPoints = inputPointNormals[:, 0:3]
        if inputPoints.shape[0] > 4500:
            continue  # not doing these in the basic version

        root, old_extension = os.path.splitext(off_file_path)
        pickle_file_path = root + '-centre-reduction-nooffpoints.data'
        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        w, RBFCentres, _ = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                               polyharmonic,
                                               loaded_data['epsilon'], useOffPoints=False)

        RBFValues = evaluate_RBF(loaded_data['xyz'], RBFCentres, polyharmonic, w)

        print("No-off-points checks:")
        print("w error: ", np.amax(loaded_data['w'] - w))
        print("RBFCentres error: ", np.amax(loaded_data['RBFCentres'] - RBFCentres))
        print("RBFValues error: ", np.amax(loaded_data['RBFValues'] - RBFValues))

        pickle_file_path = root + '-centre-reduction-subset.data'
        with open(pickle_file_path, 'rb') as pickle_file:
            loaded_data = pickle.load(pickle_file)

        w, RBFCentres, _ = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                               polyharmonic,
                                               loaded_data['epsilon'],
                                               RBFCentreIndices=loaded_data['RBFCentreIndices'])

        RBFValues = evaluate_RBF(loaded_data['xyz'], RBFCentres, polyharmonic, w)

        print("Centre subset checks:")
        print("w error: ", np.amax(loaded_data['w'] - w))
        print("RBFCentres error: ", np.amax(loaded_data['RBFCentres'] - RBFCentres))
        print("RBFValues error: ", np.amax(loaded_data['RBFValues'] - RBFValues))
