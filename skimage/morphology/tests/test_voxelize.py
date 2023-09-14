import numpy as np

from skimage.segmentation import flood_fill
from skimage import measure
from skimage.morphology import voxelize_mesh

class TestVoxelizeMesh:

    def generate_sphere(self, r):
        z, y, x = np.ogrid[-r: r + 1, -r: r + 1, -r: r + 1]
        mask = x**2 + y**2 + z**2 <= r**2
        return mask.astype(np.uint8)
    
    def test_voxelize_accuracy(self):
        mask = self.generate_sphere(20)
        verts, faces, _, _ = measure.marching_cubes(mask, 0.5)
        
        voxelized_img = voxelize_mesh(verts, faces, mask.shape)
        filled_voxelized_img = flood_fill(voxelized_img, (20, 20, 20), True)
        
        diff = np.sum(np.abs(mask.astype(int) - filled_voxelized_img.astype(int)))
        total = np.prod(mask.shape)
        
        # Assert that the difference ratio is minimal
        assert diff / total < 0.07, f"Voxelization does not match the original image closely enough. Difference: {diff} of {total} is {diff/total}"
