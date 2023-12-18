import os

import numpy as np
from numba import cuda
from numba import njit

import random

from simple_3dviz import Mesh, Lines
from simple_3dviz.window import show

import trimesh
from trimesh.voxel.creation import local_voxelize, voxelize

import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
CACHE_DIR = 'cache'

class VoxelSlicer():
    def __init__(self, path, pitch=1):
        self.filename = os.path.basename(path)[:-4]
        self.path = path
        self.pitch = pitch
        self.dir = f'{CACHE_DIR}\\{self.filename}\\p{self.pitch}'
        self.out_dir = f'{OUTPUT_DIR}\\{self.filename}\\p{self.pitch}'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir) 
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
       
        self.dense_path = f'{self.dir}\dense_{self.pitch}.npy'
        self.planar_mask_path = f'{self.dir}\planar_mask_{self.pitch}.npy'
        self.nonplanar_mask_path = f'{self.dir}\\nonplanar_mask_{self.pitch}.npy'
        self.renderables_path = None

        self.mesh = trimesh.load_mesh(path)
        self.bounds = self.mesh.bounds
        self.size_mm = self.bounds[1] - self.bounds[0]

        self.dense = self.get_dense()

        self.shape = self.dense.shape
        self.size_vg = self.dense.size
        self.number_of_voxels = np.count_nonzero(self.dense)

        self.scale = 1 / (self.shape[1]*self.pitch)

        print(f'\nInitialized {path} with {pitch} mm pitch, {self.number_of_voxels:,} voxels')

    def get_dense(self):
        try:
            dense = np.load(self.dense_path)
            print('Using cached dense tensor')
            return dense
        except:
            dense = self.mesh_to_dense()
            print('Saving dense tensor')
            np.save(self.dense_path, dense)
            return dense           

    def mesh_to_dense(self):
        print('Generating dense tensor')
        radius = int(np.max(self.size_mm)/self.pitch * 0.5)+2
        voxels = local_voxelize(self.mesh, self.mesh.centroid, self.pitch, radius, fill=True)
        dense = voxels.matrix
        return dense

    
    def split_voxelgrid(self, voxelgrid, n):
        z_indices = np.arange(0, self.shape[2], 1, dtype=int)
        split_z_indices = np.array_split(z_indices, n)
        partial_voxelgrids = []
        for indices in split_z_indices:
            start, stop = indices[0], indices[-1]+1
            partial_voxelgrid = np.zeros_like(voxelgrid)
            partial_voxelgrid[:,:, start:stop] = voxelgrid[:,:, start:stop]
            partial_voxelgrids.append(partial_voxelgrid)
        return partial_voxelgrids  
    
    def planar_mask(self):
        try:
            planar_mask = np.load(self.planar_mask_path)
            print('Using cached planar mask')
        except:
            # Create a mask where integers represents layer number, starting from 0
            planar_mask = np.zeros_like(self.dense, dtype=np.uint16)    
            x, y, z = self.shape
            n = 1
            for i in range(z):
                # n: current layer
                # i: current index (voxel grid is reversed in z-dir)
                if np.any(self.dense[:, :, i]):
                    planar_mask[:, :, i] = self.dense[:, :, i] * n
                    n += 1
            np.save(self.planar_mask_path, planar_mask)              
        return planar_mask 

    
    def create_renderables(self, layer_mask, voxelgrid, section=False, contours=True, layer_numbers=None, show_only_listed=False, show_stl=False):
        print('Creating visualization')
        renderables = []       
        maxdim = max(self.shape)
        num_planar_layers = np.max(layer_mask)

        if section:
            i, j, k = np.array(section) 
            i = int(i * self.shape[0])
            j = int(j * self.shape[1])
            k = int(k * self.shape[2])

            voxelgrid[:i, :, :] = False
            voxelgrid[:, :j, :] = False
            voxelgrid[:, :, :k] = False

        surface_mask = self.dense_filled_to_surface(voxelgrid)

        if isinstance(layer_numbers, int):
            layer_numbers = [layer_numbers]

        if isinstance(layer_numbers, list):
            layer_numbers = np.asarray(layer_numbers, dtype=int)
        else:
            layer_numbers = np.arange(1, num_planar_layers+1, dtype=int)
        
        if show_only_listed:
            # Remove all voxels that are not part of layers to render
            layers_only_mask = np.zeros_like(layer_mask, dtype=bool)
            for n in layer_numbers:
                layers_only_mask += (layer_mask == n)
            voxelgrid = voxelgrid * (layers_only_mask)

        color_mask = np.zeros(shape=(maxdim, maxdim, maxdim,4))
        color_mask[:, :, :, 0:3] = 0.3    # Set all voxel colors to grey
        color_mask[:, :, :, 3] = 0.05     # Set alpha
        
        for  n in  layer_numbers:
            r, g, b = random_rgb()
            a = 1
            single_layer_mask = (layer_mask == n)
            color_mask[:, :, :, 0] += single_layer_mask * r
            color_mask[:, :, :, 1] += single_layer_mask * g
            color_mask[:, :, :, 2] += single_layer_mask * b
            color_mask[:, :, :, 3] += single_layer_mask * a
        
        max_size = 3e6  # Maximum number of filled voxels per visualization mesh (errors occur if >~ 8e6)  

        layer_mask = layer_mask * surface_mask
        render_layer_mask = np.zeros_like(layer_mask, dtype=bool)
        for n in layer_numbers:
            render_layer_mask += (layer_mask == n)
        
        # Create multiple renderables to avoid gl errors
        if max_size < self.size_vg:
            print(f'Splitting data for rendering')
            n = np.ceil(self.size_vg / max_size)
            partial_voxelgrids = self.split_voxelgrid(voxelgrid, n)
            
            # First create renderables for the layers to visualize
            for pvg in partial_voxelgrids:
                if not np.any(pvg):     # Check if layer is empty
                    continue
                if np.any(pvg*render_layer_mask):
                    if contours:
                        renderables.append(Lines.from_voxel_grid(pvg*render_layer_mask, colors=(0.1, 0.1, 0.1)))
                    renderables.append(Mesh.from_voxel_grid(pvg*render_layer_mask, colors=color_mask))
            
            # Then create renderables from the remaining voxels
            for pvg in partial_voxelgrids:
                if np.any(pvg):
                    renderables.append(Mesh.from_voxel_grid(pvg, colors=color_mask))

        else:
            if contours:
                linewidth = 0.0015 + (self.pitch-0.25)/200
                renderables.append(Lines.from_voxel_grid(voxelgrid*render_layer_mask, colors=(0.1, 0.1, 0.1), width=linewidth))
            renderables.append(Mesh.from_voxel_grid(voxelgrid*render_layer_mask, colors=color_mask))
            renderables.append(Mesh.from_voxel_grid(voxelgrid, colors=color_mask))
        if show_stl:
            stl_mesh = trimesh.load_mesh(self.path)
            bbox = np.array([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
            scale = (bbox[1] - bbox[0]) / (self.shape[1]*self.pitch)  
            stl_mesh.apply_scale(scale)
            stl_mesh.apply_translation([-0.5/self.shape[0], -0.5/self.shape[1], -stl_mesh.centroid[2]])
            vertices = stl_mesh.vertices
            faces = stl_mesh.faces
            stl_renderable = Mesh.from_faces(vertices, faces, colors=(0, 0, 0, 0.5))
            renderables.append(stl_renderable)
        
        return renderables  
    
    def show_planar(self):
        voxelgrid = np.copy(self.dense)
        layer_mask = self.planar_mask()
        renderables = self.create_renderables(layer_mask, voxelgrid, section=[0, 0.5, 0])  
        
        show(renderables, 
            size=(1200, 1200),
            background=(1, 1, 1, 1), 
            title='Non-planar voxel layers', 
            camera_position=(0.5, -1.5, 0.5), 
            camera_target=(0, 0, 0), 
            up_vector=(0, 0, 1), 
            light=None, 
            behaviours=[])
    

    def show_nonplanar(self, max_t, section=False, layer_numbers=None, show_only_listed=False , contours=True, save=False, force_render=False, show_stl=False):
        if force_render:
            print('Forcing visualization recompute')

        # Generate output path
        self.renderables_path = f'{self.dir}\\renderables_{self.pitch}'
        
        if section:
            self.renderables_path += f'_section{section}'
        if contours:
            self.renderables_path += f'_contour'
        if show_stl:
            self.renderables_path += f'_stl'
        if layer_numbers:
            if isinstance(layer_numbers, int):
                self.renderables_path += f'layer_{layer_numbers}'
            else:
                layer_numbers_hash = hash(frozenset(layer_numbers))
                self.renderables_path += f'_lnh{layer_numbers_hash}'
        if show_only_listed:
            self.renderables_path += '_only'
        self.renderables_path += '.npy'
        
        renderables = None
        if not force_render:
            try:
                renderables = list(np.load(self.renderables_path, allow_pickle=True))
                print('Using cached visualization data')
            except:
                print('Failed to load visualization data')
            
        if renderables is None:
            
            layer_mask = self.nonplanar(max_t)

            voxelgrid = np.copy(self.dense)
            renderables = self.create_renderables(layer_mask, 
                                                  voxelgrid, 
                                                  section=section, 
                                                  contours=contours, 
                                                  layer_numbers=layer_numbers, 
                                                  show_only_listed=show_only_listed,
                                                  show_stl=show_stl)
            
            if save:
                print('Saving renderables')
                np.save(self.renderables_path, renderables, allow_pickle=True)

        show(renderables, 
             size=(1200, 1200),
             background=(1, 1, 1, 1), 
             title='Non-planar voxel layers', 
             camera_position=(0.5, -1.5, 0.5),
             camera_target=(0, 0, 0), 
             up_vector=(0, 0, 1), 
             light=None, 
             behaviours=[])

    def export_layers(self, max_t=None, layer_numbers=None,):
        nonplanar_mask = self.nonplanar(max_t)
        

        if isinstance(layer_numbers, int):
            layer_numbers = [layer_numbers]
            print(f'Exporting layer {layer_numbers}')
        if layer_numbers is None:   # Export all layers
            layer_numbers = [i for i in range(1, np.max(nonplanar_mask)+1)]
            print(f'Exporting {np.max(layer_numbers)} layers')
        
        # Generate output path
        self.layer_export_path= self.out_dir + f'\\{self.filename}_p{self.pitch}'
        export_layer_mask = np.zeros_like(self.dense, dtype=bool)
        
        use_trimesh = True
        for n in layer_numbers:
            print('', end='\r')
            print(f'Exporting layer {n}', end='\r')
            export_layer_mask = (nonplanar_mask == n)
            if use_trimesh:
                export_mesh = trimesh.voxel.VoxelGrid(export_layer_mask).as_boxes()             
                export_mesh.export(self.layer_export_path + f'_layer_{n}' + '.stl')
            else:
                export_mesh = Mesh.from_voxel_grid(export_layer_mask)
                vertices, faces = export_mesh.to_points_and_faces()
                export_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                export_mesh.apply_scale(1/self.scale)
                export_mesh.export(self.layer_export_path + f'_layer_{n}' + '.stl')
        print(f'Exported {len(layer_numbers)} layers')    

    def compute_distance_mask(self, padded_mask):
        x_dim, y_dim, z_dim = padded_mask.shape
       
        overhang_mask = np.zeros_like(padded_mask)
        for z in range(1, z_dim):
            overhang_mask[:, :, z] = padded_mask[:, :, z] & ~padded_mask[:, :, z-1]
        
        overhanging_indices = np.argwhere(overhang_mask == True)
        overhanging_indices = overhanging_indices[overhanging_indices[:,2].argsort()]

        distance_mask = np.zeros_like(padded_mask, dtype=np.int32)
        for idx in overhanging_indices:
            x, y, z = idx
            distance_mask[x, y, z] = 1
            for i in range(z+1, z_dim):
                distance_mask[x, y, i] = distance_mask[x, y, i-1] + 1

        return distance_mask    

    def process_section(self, section):

        offsets = np.array([(0, 0, -1), (-1, 0, 0), (1, 0, 0), 
                            (0, -1, 0), (0, 1, 0), (1, 1, 0), 
                            (-1, -1, 0), (-1, 1, 0), (1, -1, 0)], dtype=np.int32)

        
        planar_mask = np.pad(section, ((1, 1), (1, 1), (1, 1)), mode='constant')
        sorted_mask = np.zeros_like(planar_mask)
        num_planar_layers = np.max(section)
        voxels_to_place = np.count_nonzero(section)
       
        placed = 0
        remaining = voxels_to_place
        
        threads_per_block = 256
        #print('Threads per block:', threads_per_block)
        
        # Allocate GPU memory
        d_offsets = cuda.to_device(offsets)
        d_sorted = cuda.to_device(sorted_mask)
        d_planar = cuda.to_device(planar_mask)
        
        # Assuming a maximum possible size for indices
        max_indices_size = np.prod(planar_mask.shape)
        d_indices = cuda.device_array((max_indices_size, 3), dtype=np.int32)

        k = 5  # stop iteration after k*num_planar_layers (mostly to stop the infinite loop when a model fails to solve)
        n = 1
        while placed < voxels_to_place and n < num_planar_layers * k:
            
            if n == 1:
                sorted_mask[:,:,:] = (planar_mask == 1)
            else:
                indices = np.argwhere(planar_mask == n).astype(np.int32)        # Indices with layer number n
                d_indices = cuda.to_device(indices)

                # Update GPU memory for sorted_mask and indices
                d_sorted.copy_to_device(sorted_mask)
                d_planar.copy_to_device(planar_mask)
                d_indices.copy_to_device(indices)
                
                # Determine block and grid sizes
                # blocks_per_grid = (indices.shape[0] + threads_per_block - 1) // threads_per_block
                blocks_per_grid = (len(indices) + threads_per_block - 1) // threads_per_block
                
                process_and_update_indices_kernel[blocks_per_grid, threads_per_block](d_indices, d_sorted, d_planar, d_offsets, n)
                    
                # Copy the final result
                d_sorted.copy_to_host(sorted_mask)
                d_planar.copy_to_host(planar_mask)
            
            placed = np.count_nonzero(sorted_mask)
            remaining = voxels_to_place - placed
            print('', end='\r')
            print(f'  Layer: {n}    Placed: {placed:,}   Remaining: {remaining:,}', end='             \r')
            n += 1
            
        print(f'\nFinished processing section, generated {n-1} voxel layers')
        return sorted_mask[1:-1, 1:-1, 1:-1]

    def nonplanar(self, max_thickness_mm=None, save=False):
        print('Generating nonplanar voxel layers')

        if os.path.isfile(self.nonplanar_mask_path):
            nonplanar_mask = np.load(self.nonplanar_mask_path)
            print('Using cached nonplanar mask')
        else:
            planar_mask = self.planar_mask()
            nonplanar_mask = np.zeros_like(planar_mask)

            if max_thickness_mm is None or max_thickness_mm is False:
                sections = [planar_mask]
            else:
                max_t = int(max_thickness_mm/self.pitch)+1  # Max thickness in mm converted to voxel units (number of voxels)
                sections = self.section_by_overhang(planar_mask, max_t)
            
            last_layer_number = 0
            for i, section in enumerate(sections):
                print(f'Processing section {i+1}')

                processed_section = self.process_section(section)

                if last_layer_number > 0:
                    processed_section[processed_section > 0] += last_layer_number
                
                last_layer_number = processed_section.max()


                nonplanar_mask += processed_section
                # print(np.max(nonplanar_mask), len(np.unique(nonplanar_mask))-1)

                # renderables = self.create_renderables(nonplanar_mask, self.dense, section=(0, 0.5, 0))
                # show(renderables)
        if save:
                print('Saving nonplanar mask')
                np.save(self.nonplanar_mask_path, nonplanar_mask)
        # for i in range(1, last_layer_number+1):
        #     print(f'layer {i}', np.unique(np.argwhere(nonplanar_mask == i)[:, 2]))
        return nonplanar_mask
    
    def section_by_overhang(self, planar_mask, max_t):
        # max_nonplanar_thickness = int(max_thickness_mm/self.pitch)  # Max thickness in mm converted to voxel units

        overhang_mask = compute_overhang_mask(planar_mask)

        overhang_z_indices = np.sort(np.unique(np.argwhere(overhang_mask)[:, 2]))
        if not np.any(overhang_z_indices == 0):
            overhang_z_indices = np.insert(overhang_z_indices, 0, 0)    # Prepend 0 if not in overhang_z_indices
        section_index_pairs = np.array([[overhang_z_indices[i-1], overhang_z_indices[i]] for i in range(1, len(overhang_z_indices))])
        
        section_index_pairs = []

        last_overhang_z = 0
        z0, z1 = 0, 0
        for z in range(planar_mask.shape[2]):
            if np.any(overhang_mask[:, :, z]):
                last_overhang_z = z
            if z - last_overhang_z <= max_t:
                z1 = z
            else:
                section_index_pairs.append([z0, z1 + 1])
                z0 = z
                z1 = z
                last_overhang_z = z
        if section_index_pairs[-1][1] < planar_mask.shape[2]:
            section_index_pairs.append([section_index_pairs[-1][1], planar_mask.shape[2]])
        
        sections = []
        for index_pair in section_index_pairs:
            z0, z1 = index_pair
            if z0 >= planar_mask.shape[2]:
                continue
            if z1 >= planar_mask.shape[2]:
                z1 = planar_mask.shape[2] - 1
            new_section = np.zeros_like(planar_mask)
            new_section[:, :, z0:z1] = planar_mask[:, :, z0:z1]
            if np.any(new_section):
                # Renumber layer numbers in section so that the first layer is layer 1
                n_min = new_section[new_section > 0].min()  # Minimum non-zero layer number
                if n_min > 1:
                    new_section[new_section > 0] -= (n_min - 1)
                sections.append(new_section)
                print(f'Min: {new_section[new_section > 0].min()}, Max: {new_section.max()}, Unique: {len(np.unique(new_section))-1}')
        
        return sections
    
    def dense_filled_to_surface(self, dense):
        dense = np.pad(dense, ((1,1), (1,1), (1,1)), mode='constant')
        surface = np.zeros_like(dense)

        d_dense = cuda.to_device(dense)
        d_surface = cuda.to_device(surface)

        threads_per_block = (8, 8, 8)
        blocks_per_grid_x = int(np.ceil(dense.shape[0] / threads_per_block[0]))
        blocks_per_grid_y = int(np.ceil(dense.shape[1] / threads_per_block[1]))
        blocks_per_grid_z = int(np.ceil(dense.shape[2] / threads_per_block[2]))
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

        dense_surface_kernel[blocks_per_grid, threads_per_block](d_dense, d_surface)

        d_surface.copy_to_host(surface)
        print('Dense:', np.count_nonzero(dense))
        print('Surface:', np.count_nonzero(surface))

        return surface[1:-1, 1:-1, 1:-1]

    def create_grid_mesh(self):
        xmax, ymax, zmax = self.shape
        
        # Generate corner vertices
        vertices = np.array([[j-0.5, i-0.5, zmax-1] for i in range(xmax+1) for j in range(ymax+1)])
        
        triangles = []

        for i in range(xmax):
            for j in range(ymax):
                # Calculate corner indices for each voxel
                bottom_left     = i * (xmax + 1) + j
                bottom_right    = bottom_left + 1
                top_left        = bottom_left + (xmax + 1)
                top_right       = top_left + 1

                # Add two triangles per voxel
                triangles.append([bottom_left, top_right, top_left])
                triangles.append([bottom_left, bottom_right, top_right])
        triangles = np.array(triangles)
        if False:
            vertices = vertices - np.array([self.pitch, self.pitch, 0])
            grid_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            grid_mesh.rezero()
            grid_mesh.export('gridmesh.stl', file_type='stl_ascii')

        return vertices, triangles

    def generate_blanket_mesh(self, layer):  

        vertices, triangles = self.create_grid_mesh()
        filled_indices = np.argwhere(layer != 0)
        if filled_indices.size == 0:
            return
        voxel_corner_vertices = []
        for index in filled_indices:
            i, j, k = index
            voxel_corner_vertices.append([i-1, j, k+0.5])
            voxel_corner_vertices.append([i, j, k+0.5])
            voxel_corner_vertices.append([i-1, j-1, k+0.5])
            voxel_corner_vertices.append([i, j-1, k+0.5])
        voxel_corner_vertices = np.array(voxel_corner_vertices)

        # Move all vertices above filled voxels to the closest voxel
        
        for i, vertex in enumerate(vertices):
            x, y, z = vertex.astype(int)
            new_z = -1
            potential_mask = (voxel_corner_vertices[:, 0] == x) & (voxel_corner_vertices[:, 1] == y)
            potential_points = voxel_corner_vertices[potential_mask]
            for point in potential_points:
                new_z = max(new_z, point[2])    # Update value if it is above current new_z
            vertices[i, 2] = new_z
        vertices = set_negative_z_to_closest_positive(vertices)

        blanket_mesh = trimesh.Trimesh(vertices, triangles)
        
        # Transform back to same to coordinate system as the input mesh
        offset = (self.pitch * self.shape[0]) / 2

        blanket_mesh.apply_scale(self.pitch)
        blanket_mesh.apply_translation([-offset, -offset, -(offset-self.mesh.centroid[2]+self.pitch)])
    
        return blanket_mesh

def set_negative_z_to_closest_positive(vertices):
    negative_mask = (vertices[:, 2] == -1)
    negative_indices = np.argwhere(negative_mask).flatten()  # Flatten to get a 1D array
    positive_points = vertices[~negative_mask]

    for index in negative_indices:
        negative_point = vertices[index]
        distances_2D = np.sqrt(np.sum((positive_points[:, :2] - negative_point[:2].reshape(1, 2))**2, axis=1))
        closest_positive_idx = np.argmin(distances_2D)
        # Assign the z-coordinate of the closest positive point to the current negative point
        vertices[index, 2] = positive_points[closest_positive_idx, 2]
    return vertices


def compute_overhang_mask(planar_mask):
    print('Finding overhanging voxels')
    # Returns a mask of where voxels are unsupported below.
    overhang_mask = np.zeros_like(planar_mask, dtype=bool)  # True where a voxel is unsupported beneath
    
    threads_per_block = (32, 32)
    blocks_per_grid_x = int(np.ceil(planar_mask.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(planar_mask.shape[1] / threads_per_block[1]))
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    d_planar = cuda.to_device(planar_mask.astype(bool))
    d_overhang_mask = cuda.to_device(overhang_mask)
    
    for z in range(1, planar_mask.shape[2]):
        find_overhangs_kernel[blocks_per_grid, threads_per_block](d_planar, d_overhang_mask, z)
    d_overhang_mask.copy_to_host(overhang_mask)
    return overhang_mask



@cuda.jit
def dense_surface_kernel(d_dense, d_surface):
    # Get the x, y, z indices
    x, y, z = cuda.grid(3)
    dx, dy, dz = d_dense.shape
    
    # Return if indices are out of bounds
    if x >= dx or y >= dy or z >= dz:
        return
    
    # Define start and end for each dimension
    x_start, x_end = max(0, x-1), min(dx, x+2)
    y_start, y_end = max(0, y-1), min(dy, y+2)
    z_start, z_end = max(0, z-1), min(dz, z+2)
    
    # Loop over the sub-tensor (with boundary checks)
    for xi in range(x_start, x_end):
        for yi in range(y_start, y_end):
            for zi in range(z_start, z_end):
                if d_dense[x, y, z] == True:
                    if d_dense[xi, yi, zi] == False:
                        d_surface[x, y, z] = True
    

@cuda.jit
def find_overhangs_kernel(d_planar, d_overhang_mask, z):
    x, y = cuda.grid(2)
    if x > d_planar.shape[0] or y > d_planar.shape[1]:
        return
    if not d_planar[x, y, z]:
        return
    if d_planar[x, y, z-1]:
        return
    d_overhang_mask[x, y, z] = True
    return


@cuda.jit
def process_and_update_indices_kernel(d_indices, d_sorted, d_planar, d_offsets, n):
    idx = cuda.grid(1)
    
    # Check boundary
    if idx >= d_indices.shape[0]:
        return
    
    # Unpack indices
    x, y, z = d_indices[idx]  

    # Update sorted 
    for dx, dy, dz in d_offsets:
        if d_sorted[x + dx, y + dy, z + dz] == n-1:
            d_sorted[x, y, z] = n
            return
    d_planar[x, y, z] = n + 1

@cuda.jit
def process_indices_kernel(d_sorted, d_indices, d_offsets, d_surface, n):
    idx = cuda.grid(1)  # 1D grid of threads
    
    if idx >= d_indices.shape[0]:
        return
    
    x, y, z = d_indices[idx]
    
    for dx, dy, dz in d_offsets:
        if d_sorted[x + dx, y + dy, z + dz] == n - 1:
            d_surface[idx] = 1
            return

    d_surface[idx] = 0


def random_rgb():
    return [random.randint(0,1000)/1000 for i in range(3)]



if __name__ == '__main__':
    path = 'meshes\internal_external_overhang_hole.stl'
    pitch = 0.2
    max_t = 3
    
    # Initialize the VoxelSlicer
    vs = VoxelSlicer(path, pitch)

    # Show planar voxel layers
    vs.show_planar()

    # Show non-planar voxel layers
    vs.show_nonplanar(max_t, section=(0, 0, 0), layer_numbers=None, show_only_listed=True, contours=True, save=True, force_render=True,show_stl=False)
    
    # Export non-planar voxel layers
    # vs.export_layers()
