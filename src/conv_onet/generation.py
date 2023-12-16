import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange, tqdm
import trimesh
from src.utils import libmcubes
from src.common import make_3d_grid, normalize_coord, add_key, coord2index
from src.utils.libsimplify import simplify_mesh
from src.utils.libmise import MISE
import time
import math
###
from skimage import measure
import open3d as o3d

counter = 0


class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        refinement_step (int): number of refinement steps
        device (device): pytorch device
        resolution0 (int): start resolution for MISE
        upsampling steps (int): number of upsampling steps
        with_normals (bool): whether normals should be estimated
        padding (float): how much padding should be used for MISE
        sample (bool): whether z should be sampled
        input_type (str): type of input
        vol_info (dict): volume infomation
        vol_bound (dict): volume boundary
        simplify_nfaces (int): number of faces the mesh should be simplified to
    '''

    def __init__(self, model, points_batch_size=1000000,
                 threshold=0.5, refinement_step=0, device=None,
                 resolution0=16, upsampling_steps=3,
                 with_normals=False, padding=0.1, sample=False,
                 input_type = None,
                 vol_info = None,
                 vol_bound = None,
                 simplify_nfaces=None):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.refinement_step = refinement_step
        self.threshold = threshold
        self.device = device
        self.resolution0 = resolution0
        self.upsampling_steps = upsampling_steps
        self.with_normals = with_normals
        self.input_type = input_type
        self.padding = padding
        self.sample = sample
        self.simplify_nfaces = simplify_nfaces
        
        # for pointcloud_crop
        self.vol_bound = vol_bound
        if vol_info is not None:
            self.input_vol, _, _ = vol_info

    def generate_mesh(self, data, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}

        inputs = data.get('inputs', torch.empty(1, 0)).to(device)
        kwargs = {}

        t0 = time.time()

        # obtain features for all crops
        if self.vol_bound is not None:
            self.get_crop_bound(inputs)
            c_plane = self.encode_crop(inputs, device)
            c_final = c_plane
        else: # input the entire volume
            inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            # inputs = add_key(inputs, data.get('inputs.ind'), 'points', 'index', device=device)
            t0 = time.time()
            with torch.no_grad():
                #c = self.model.encode_inputs(inputs) ### original 
               
                '''
                c_plane, point_feature = self.model.encoder(inputs, 0) ###  
                # else:
                #     #print('####see pi size:', pi.size())
                #     print('####see pi shape:', pi.shape)
                #     c = self.model.encoder(pi, n, c) ###?

                c = self.model.decode(inputs, c_plane, 0, point_feature) ###
                c_final, _ = self.model.encoder(inputs, 1, c, c_plane)
                '''

                point_feature=None
                c_plane = self.model.module.encoder(inputs)
                c_final = c_plane

        stats_dict['time (encode inputs)'] = time.time() - t0

        mesh = self.generate_from_latent(c_final, inputs, stats_dict=stats_dict, **kwargs) ### add inputs for poco mc try

        if return_stats:
            return mesh, stats_dict
        else:
            return mesh


    def generate_from_latent(self, c_final, inputs, stats_dict={}, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''
        inputs = inputs[0].cpu().numpy() #.detach().cpu().numpy() ###
        #print('inputs shape======',inputs.shape) #(10000,3)
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)

        t0 = time.time()
        # Compute bounding box size
        box_size = 1 + self.padding

        # Shortcut
        if self.upsampling_steps == 0:
            nx = self.resolution0
            pointsf = box_size * make_3d_grid(
                (-0.5,) * 3, (0.5,) * 3, (nx,) * 3
            )

            values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()
            value_grid = values.reshape(nx, nx, nx)
        else:
            # input_points (10000,3)
            input_points = inputs
            bmin = input_points.min()
            bmax = input_points.max()

            ########################## hard-code paramters for now #########################
            step = None
            resolution = 256
            padding = 1
            dilation_size = 2
            device = self.device
            num_pts = 50000
            out_value = 1
            mc_value = 0
            return_volume = False
            refine_iter = 10
            simplification_target = None
            refine_threshold = None
            ###############################################################################


            if step is None:
                step = (bmax - bmin) / (resolution - 1)  # 0.0039886895348044005
                resolutionX = resolution  # 256
                resolutionY = resolution  # 256
                resolutionZ = resolution  # 256
            else:
                bmin = input_points.min(axis=0)
                bmax = input_points.max(axis=0)
                resolutionX = math.ceil((bmax[0] - bmin[0]) / step)
                resolutionY = math.ceil((bmax[1] - bmin[1]) / step)
                resolutionZ = math.ceil((bmax[2] - bmin[2]) / step)

            bmin_pad = bmin - padding * step
            bmax_pad = bmax + padding * step

            pts_ids = (input_points - bmin) / step + padding
            pts_ids = pts_ids.astype(np.int)  # (10000,3)

            # create the volume
            volume = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), np.nan,
                             dtype=np.float64)
            mask_to_see = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding),
                                  True, dtype=bool)
            while (pts_ids.shape[0] > 0):

                # print("Pts", pts_ids.shape)

                # creat the mask
                mask = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding), False,
                               dtype=bool)
                mask[pts_ids[:, 0], pts_ids[:, 1], pts_ids[:, 2]] = True

                # dilation
                for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
                    xc = int(pts_ids[i, 0])
                    yc = int(pts_ids[i, 1])
                    zc = int(pts_ids[i, 2])
                    mask[max(0, xc - dilation_size):xc + dilation_size,
                    max(0, yc - dilation_size):yc + dilation_size,
                    max(0, zc - dilation_size):zc + dilation_size] = True

                # get the valid points
                valid_points_coord = np.argwhere(mask).astype(np.float32)
                valid_points = valid_points_coord * step + bmin_pad
                #print('valid_points===',valid_points.shape)

                # get the prediction for each valid points
                z = []
                near_surface_samples_torch = torch.tensor(valid_points, dtype=torch.float, device=device)
                for pnts in tqdm(torch.split(near_surface_samples_torch, num_pts, dim=0), ncols=100, disable=True):

                    ### our decoder
                    occ_hat = self.eval_points(pnts, c_final, **kwargs).cpu().numpy()
                    occ_hat_pos = torch.tensor(occ_hat) #[0,1]
                    occ_hat_neg = occ_hat - 1 #[-1,0]
                    outputs = -(occ_hat_pos + occ_hat_neg) #[-1,1]
                    z.append(outputs)

                z = np.concatenate(z, axis=0)
                z = z.astype(np.float64)

                # update the volume
                volume[mask] = z

                # create the masks
                mask_pos = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding),
                                   False, dtype=bool)
                mask_neg = np.full((resolutionX + 2 * padding, resolutionY + 2 * padding, resolutionZ + 2 * padding),
                                   False, dtype=bool)

                # dilation
                for i in tqdm(range(pts_ids.shape[0]), ncols=100, disable=True):
                    xc = int(pts_ids[i, 0])
                    yc = int(pts_ids[i, 1])
                    zc = int(pts_ids[i, 2])
                    mask_to_see[xc, yc, zc] = False
                    if volume[xc, yc, zc] <= 0:
                        mask_neg[max(0, xc - dilation_size):xc + dilation_size,
                        max(0, yc - dilation_size):yc + dilation_size,
                        max(0, zc - dilation_size):zc + dilation_size] = True
                    if volume[xc, yc, zc] >= 0:
                        mask_pos[max(0, xc - dilation_size):xc + dilation_size,
                        max(0, yc - dilation_size):yc + dilation_size,
                        max(0, zc - dilation_size):zc + dilation_size] = True

                # get the new points

                new_mask = (mask_neg & (volume >= 0) & mask_to_see) | (mask_pos & (volume <= 0) & mask_to_see)
                pts_ids = np.argwhere(new_mask).astype(np.int)

            volume[0:padding, :, :] = out_value
            volume[-padding:, :, :] = out_value
            volume[:, 0:padding, :] = out_value
            volume[:, -padding:, :] = out_value
            volume[:, :, 0:padding] = out_value
            volume[:, :, -padding:] = out_value

            # volume[np.isnan(volume)] = out_value
            maxi = volume[~np.isnan(volume)].max()
            mini = volume[~np.isnan(volume)].min()

            if not (maxi > mc_value and mini < mc_value):
                return None

            if return_volume:
                return volume

            # compute the marching cubes
            verts, faces, _, _ = measure.marching_cubes(
                volume=volume.copy(),
                level=mc_value,
            )

            # removing the nan values in the vertices
            values = verts.sum(axis=1)
            o3d_verts = o3d.utility.Vector3dVector(verts)
            o3d_faces = o3d.utility.Vector3iVector(faces)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)
            mesh.remove_vertices_by_mask(np.isnan(values))
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)

            if refine_iter > 0:

                dirs = verts - np.floor(verts)
                dirs = (dirs > 0).astype(dirs.dtype)

                mask = np.logical_and(dirs.sum(axis=1) > 0, dirs.sum(axis=1) < 2)
                v = verts[mask]
                dirs = dirs[mask]

                # initialize the two values (the two vertices for mc grid)
                v1 = np.floor(v)
                v2 = v1 + dirs

                # get the predicted values for both set of points
                v1 = v1.astype(int)
                v2 = v2.astype(int)
                preds1 = volume[v1[:, 0], v1[:, 1], v1[:, 2]]
                preds2 = volume[v2[:, 0], v2[:, 1], v2[:, 2]]

                # get the coordinates in the real coordinate system
                v1 = v1.astype(np.float32) * step + bmin_pad
                v2 = v2.astype(np.float32) * step + bmin_pad

                # tmp mask
                mask_tmp = np.logical_and(
                    np.logical_not(np.isnan(preds1)),
                    np.logical_not(np.isnan(preds2))
                )
                v = v[mask_tmp]
                dirs = dirs[mask_tmp]
                v1 = v1[mask_tmp]
                v2 = v2[mask_tmp]
                mask[mask] = mask_tmp

                # initialize the vertices
                verts = verts * step + bmin_pad
                v = v * step + bmin_pad

                # iterate for the refinement step
                for iter_id in tqdm(range(refine_iter), ncols=50, disable=True):

                    preds = []
                    pnts_all = torch.tensor(v, dtype=torch.float, device=device)
                    for pnts in tqdm(torch.split(pnts_all, num_pts, dim=0), ncols=100, disable=True):
                        occ_hat = self.eval_points(pnts, c_final, **kwargs).cpu().numpy()
                        occ_hat_pos = torch.tensor(occ_hat)  # [0,1]
                        occ_hat_neg = occ_hat - 1  # [-1,0]
                        outputs = -(occ_hat_pos + occ_hat_neg)  # [-1,1]
                        preds.append(outputs)


                    preds = np.concatenate(preds, axis=0)

                    mask1 = (preds * preds1) > 0
                    v1[mask1] = v[mask1]
                    preds1[mask1] = preds[mask1]

                    mask2 = (preds * preds2) > 0
                    v2[mask2] = v[mask2]
                    preds2[mask2] = preds[mask2]

                    v = (v2 + v1) / 2

                    verts[mask] = v

                    # keep only the points that needs to be refined
                    if refine_threshold is not None:
                        mask_vertices = (np.linalg.norm(v2 - v1, axis=1) > refine_threshold)
                        # print("V", mask_vertices.sum() , "/", v.shape[0])
                        v = v[mask_vertices]
                        preds1 = preds1[mask_vertices]
                        preds2 = preds2[mask_vertices]
                        v1 = v1[mask_vertices]
                        v2 = v2[mask_vertices]
                        mask[mask] = mask_vertices

                        if v.shape[0] == 0:
                            break
                        # print("V", v.shape[0])

            else:
                verts = verts * step + bmin_pad

            o3d_verts = o3d.utility.Vector3dVector(verts)
            o3d_faces = o3d.utility.Vector3iVector(faces)
            mesh = o3d.geometry.TriangleMesh(o3d_verts, o3d_faces)

            if simplification_target is not None and simplification_target > 0:
                mesh = o3d.geometry.TriangleMesh.simplify_quadric_decimation(mesh, simplification_target)

            return mesh



    def eval_points(self, p, c=None, point_feature=None, n=None, N=None, vol_bound=None, **kwargs): ### add n, N
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
                pi = pi.unsqueeze(0).to(self.device)
                chunk_size = 5000
                pi_chunks = torch.split(pi, chunk_size, 1)
                with torch.no_grad():
                    p_r = self.model.module.decode(pi, c, logits=True)
                    occ_hat = p_r.probs
                occ_hats.append(occ_hat.squeeze(0).detach().cpu())
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat



    def extract_mesh(self, occ_hat, c=None, stats_dict=dict()):
        ''' Extracts the mesh from the predicted occupancy grid.

        Args:
            occ_hat (tensor): value grid of occupancies
            c (tensor): encoded feature volumes
            stats_dict (dict): stats dictionary
        '''
        # Some short hands
        n_x, n_y, n_z = occ_hat.shape
        box_size = 1 + self.padding
        threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        # Make sure that mesh is watertight
        t0 = time.time()
        occ_hat_padded = np.pad(
            occ_hat, 1, 'constant', constant_values=-1e6)
        vertices, triangles = libmcubes.marching_cubes(
            occ_hat_padded, threshold)
        stats_dict['time (marching cubes)'] = time.time() - t0
        # Strange behaviour in libmcubes: vertices are shifted by 0.5
        vertices -= 0.5
        # # Undo padding
        vertices -= 1
        
        if self.vol_bound is not None:
            # Scale the mesh back to its original metric
            bb_min = self.vol_bound['query_vol'][:, 0].min(axis=0)
            bb_max = self.vol_bound['query_vol'][:, 1].max(axis=0)
            mc_unit = max(bb_max - bb_min) / (self.vol_bound['axis_n_crop'].max() * self.resolution0*2**self.upsampling_steps)
            vertices = vertices * mc_unit + bb_min
        else: 
            # Normalize to bounding box
            vertices /= np.array([n_x-1, n_y-1, n_z-1])
            vertices = box_size * (vertices - 0.5)
        
        # Estimate normals if needed
        if self.with_normals and not vertices.shape[0] == 0:
            t0 = time.time()
            normals = self.estimate_normals(vertices, c)
            stats_dict['time (normals)'] = time.time() - t0

        else:
            normals = None


        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles,
                               vertex_normals=normals,
                               process=False)
        


        # Directly return if mesh is empty
        if vertices.shape[0] == 0:
            return mesh

        # TODO: normals are lost here
        if self.simplify_nfaces is not None:
            t0 = time.time()
            mesh = simplify_mesh(mesh, self.simplify_nfaces, 5.)
            stats_dict['time (simplify)'] = time.time() - t0

        # Refine mesh
        if self.refinement_step > 0:
            t0 = time.time()
            self.refine_mesh(mesh, occ_hat, c)
            stats_dict['time (refine)'] = time.time() - t0

        return mesh

    def estimate_normals(self, vertices, c=None):
        ''' Estimates the normals by computing the gradient of the objective.

        Args:
            vertices (numpy array): vertices of the mesh
            c (tensor): encoded feature volumes
        '''
        device = self.device
        vertices = torch.FloatTensor(vertices)
        vertices_split = torch.split(vertices, self.points_batch_size)

        normals = []
        c = c.unsqueeze(0)
        for vi in vertices_split:
            vi = vi.unsqueeze(0).to(device)
            vi.requires_grad_()
            occ_hat = self.model.decode(vi, c).logits
            out = occ_hat.sum()
            out.backward()
            ni = -vi.grad
            ni = ni / torch.norm(ni, dim=-1, keepdim=True)
            ni = ni.squeeze(0).cpu().numpy()
            normals.append(ni)

        normals = np.concatenate(normals, axis=0)
        return normals

    def refine_mesh(self, mesh, occ_hat, c=None):
        ''' Refines the predicted mesh.

        Args:   
            mesh (trimesh object): predicted mesh
            occ_hat (tensor): predicted occupancy grid
            c (tensor): latent conditioned code c
        '''

        self.model.eval()

        # Some shorthands
        n_x, n_y, n_z = occ_hat.shape
        assert(n_x == n_y == n_z)
        # threshold = np.log(self.threshold) - np.log(1. - self.threshold)
        threshold = self.threshold

        # Vertex parameter
        v0 = torch.FloatTensor(mesh.vertices).to(self.device)
        v = torch.nn.Parameter(v0.clone())

        # Faces of mesh
        faces = torch.LongTensor(mesh.faces).to(self.device)

        # Start optimization
        optimizer = optim.RMSprop([v], lr=1e-4)

        for it_r in trange(self.refinement_step):
            optimizer.zero_grad()

            # Loss
            face_vertex = v[faces]
            eps = np.random.dirichlet((0.5, 0.5, 0.5), size=faces.shape[0])
            eps = torch.FloatTensor(eps).to(self.device)
            face_point = (face_vertex * eps[:, :, None]).sum(dim=1)

            face_v1 = face_vertex[:, 1, :] - face_vertex[:, 0, :]
            face_v2 = face_vertex[:, 2, :] - face_vertex[:, 1, :]
            face_normal = torch.cross(face_v1, face_v2)
            face_normal = face_normal / \
                (face_normal.norm(dim=1, keepdim=True) + 1e-10)
            face_value = torch.sigmoid(
                self.model.decode(face_point.unsqueeze(0), c).logits
            )
            normal_target = -autograd.grad(
                [face_value.sum()], [face_point], create_graph=True)[0]

            normal_target = \
                normal_target / \
                (normal_target.norm(dim=1, keepdim=True) + 1e-10)
            loss_target = (face_value - threshold).pow(2).mean()
            loss_normal = \
                (face_normal - normal_target).pow(2).sum(dim=1).mean()

            loss = loss_target + 0.01 * loss_normal

            # Update
            loss.backward()
            optimizer.step()

        mesh.vertices = v.data.cpu().numpy()

        return mesh
