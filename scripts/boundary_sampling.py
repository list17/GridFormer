import os
import numpy as np
import glob
from scipy.spatial import KDTree
from multiprocessing import Pool
import time


def calculate_constrasitve(input_folder):
    start_time = time.time()
    radiuses = [0.08]
    for radius in radiuses:
        points_path = os.path.join(input_folder, 'points.npz')
        
        all_points = np.load(points_path)
        occ = np.unpackbits(all_points['occupancies']).astype(np.int64)
        points = all_points['points']

        contra = np.zeros(occ.shape)
        # find if there is a point within the radius which has the different occupancy
        # if there is a point with the different occupancy, then the point is a contrastive point
        tree = KDTree(points)
        for i in range(occ.shape[0]):
            dist, ind = tree.query(points[i], k=100, distance_upper_bound=radius)
            # select the ind < occ.shape[0] to remove the unvalid index
            ind = ind[ind < occ.shape[0]]
            if np.sum(occ[ind] != occ[i]) > 0:
                contra[i] = 1

        # save the boundary points 
        colors = [
            [255, 0, 0],
            [0, 0, 255]
        ]
        points = np.concatenate([points, np.array([colors[i] for i in occ])], axis=1)
        output_path = os.path.join(input_folder, f'boundary{radius}')
        os.makedirs(output_path, exist_ok=True)
        
        np.savetxt(os.path.join(output_path, 'boundary.xyz'), points[contra == 1])
        all_points = dict(all_points)
        all_points['contrastive'] = contra.astype(np.uint8)
        np.savez(os.path.join(output_path, 'boundary_points.npz'), **all_points)
        np.save(os.path.join(output_path, 'boundary_value.npy'), contra.astype(np.uint8))
    end_time = time.time()
    print(f'processing {input_folder} time: {end_time - start_time}')
    
    
if __name__ == '__main__':
    lst_paths = glob.glob(os.path.join('../data/ShapeNet/', '*','*.lst'))
    all_folder = []
    for lst_path in lst_paths:
        with open(lst_path) as f:
            lines = f.readlines()
            all_folder.extend([os.path.join(os.path.dirname(lst_path), line.strip()) for line in lines])
    
    print(f'total number of models: {len(all_folder)}')
    
    with Pool(32) as p:
        p.map(calculate_constrasitve, all_folder)
