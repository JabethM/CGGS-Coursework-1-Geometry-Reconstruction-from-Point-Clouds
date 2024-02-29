import polyscope as ps
import os
import sys
import pickle
from scipy.spatial import Delaunay

sys.path.append(os.path.join('..', 'code'))
from ReconstructionFunctions import load_off_file, compute_RBF_weights, evaluate_RBF, wendland, biharmonic, polyharmonic


ps.init()

data_path = os.path.join('..', 'data')  # Replace with the path to your folder
off_files = [file for file in os.listdir(data_path) if file.endswith(".off")]
epsilonRange = [1e-4, 1e-3, 1e-2, 1e-1]

currFileIndex = 1
currEpsilonIndex = 0

off_file_path = os.path.join(data_path, off_files[currFileIndex])

inputPointNormals, _ = load_off_file(off_file_path)
inputPoints = inputPointNormals[:, 0:3]

root, old_extension = os.path.splitext(off_file_path)
pickle_file_path = root + '-basic-version-eps-' + str(currEpsilonIndex) + '.data'
with open(pickle_file_path, 'rb') as pickle_file:
    loaded_data = pickle.load(pickle_file)

w, RBFCentres, _ = compute_RBF_weights(loaded_data['inputPoints'], loaded_data['inputNormals'],
                                                   polyharmonic,
                                                   loaded_data['currEpsilon'])

RBFValues = evaluate_RBF(loaded_data['xyz'], RBFCentres, polyharmonic, w)



# Add the mesh to Polyscope
ps_mesh = ps.register_point_cloud("RBF Mesh", inputPoints)
ps_vf = ps.register_surface_mesh("reconstruction", loaded_data['xyz'], RBFValues)
# Add the normals to visualization
# Color the mesh based on RBF values
#ps_mesh.add_scalar_quantity("RBF Values", RBFValues, defined_on='vertices')

# Show the visualization
ps.show()

"""vertices, faces = load_off_file(os.path.join('..', 'data', 'Kitten.off'))
ps_mesh = ps.register_surface_mesh("my mesh", vertices, faces, smooth_shade=True)

faceNormals, faceAreas = compute_areas_normals(vertices, faces)
ps_mesh.add_scalar_quantity("face areas", faceAreas, defined_on='faces')
ps_mesh.add_vector_quantity("face normals", faceNormals, defined_on='faces')
"""
