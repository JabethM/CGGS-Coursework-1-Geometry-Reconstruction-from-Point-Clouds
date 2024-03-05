import os
import sys
import pickle
import numpy as np
from functools import partial


sys.path.append(os.path.join('code'))
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF, wendland, biharmonic, polyharmonic

if __name__ == '__main__':

    wendy = partial(wendland, 1)
    func = wendy

    data_path = os.path.join('data')  # Replace with the path to your folder

    epsilon = 1e-3
    epsilonRange = [1e-4, 1e-3, 1e-2, 1e-1]
    LRange = np.arange(0, 3)

    # Get a list of all files in the folder with the ".off" extension
    off_files = [file for file in os.listdir(data_path) if file.endswith(".off")]

    for currFileIndex in range(len(off_files)):
        print("Processing mesh ", off_files[currFileIndex])
        off_file_path = os.path.join(data_path, off_files[currFileIndex])

        inputPointNormals, _ = load_off_file(off_file_path)
        inputPoints = inputPointNormals[:, 0:3]

        if (inputPoints.shape[0] > 4500):
            continue
        
        print("Basic:")
        for currEpsilonIndex in range(len(epsilonRange)):
            root, old_extension = os.path.splitext(off_file_path)
            pickle_file_path = root + '-basic-version-eps-' + str(currEpsilonIndex) + '.data'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_data = pickle.load(pickle_file)

            w, RBFCentres, _ = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                                   func,
                                                   loaded_data['currEpsilon'])

            w_sparse, RBFCentres_sparse, _ = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                                   func,
                                                   loaded_data['currEpsilon'], sparsify=True)
            print("w_reg error: ", np.amax(loaded_data['w'] - w))
            print("w_sparse error: ", np.amax(loaded_data['w'] - w_sparse))

        print()
        print("Polynomial:")
        for LIndex in range(len(LRange)):
            root, old_extension = os.path.splitext(off_file_path)
            pickle_file_path = root + '-polynomial-precision-L-' + str(LIndex) + '.data'
            with open(pickle_file_path, 'rb') as pickle_file:
                loaded_data = pickle.load(pickle_file)
            
            

            w, RBFCentres, a = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                                   func,
                                                   loaded_data['epsilon'], l=loaded_data['l'])
            
            w_sparse, RBFCentre_sparse, a_sparse = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                                   func,
                                                   loaded_data['epsilon'],sparsify=True, l=loaded_data['l'])
            
            print("w_reg error: ", np.amax(loaded_data['w'] - w))
            print("w_sparse error: ", np.amax(loaded_data['w'] - w_sparse))
            
         
            print("a_reg error: ", np.amax(loaded_data['a'] - a))
            print("a_sparse error: ", np.amax(loaded_data['a'] - a_sparse))
            
