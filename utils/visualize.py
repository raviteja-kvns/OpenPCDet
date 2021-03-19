"""Utilities for Open3D (0.10.0)"""
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def visualize_point_cloud(points, colors=None, normals=None,
                          show_frame=False, frame_size=1.0, frame_origin=(0, 0, 0)):
    """Visualize a point cloud."""
    pc = np2pcd(points, colors, normals)
    geometries = [pc]
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries)

def create_abb(center, size, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    center = np.asarray(center)
    size = np.asarray(size)
    abb = o3d.geometry.AxisAlignedBoundingBox(center - size * 0.5, center + size * 0.5)
    abb.color = color
    return abb


def create_obb(center, rotation, size, color=(0, 1, 0)):
    """Create an oriented bounding box."""
    obb = o3d.geometry.OrientedBoundingBox(center, rotation, size)
    obb.color = color
    return obb

def visualize_point_cloud_with_boxes(points, colors=None,
                                     boxes_center=None, boxes_rotation=None, boxes_size=None, box_color=(0, 1, 0),
                                     show_frame=False, frame_size=1.0, frame_origin=(0, 0, 0)):
    """Visualize the point cloud with oriented boxes."""
    pc = np2pcd(points, colors)
    geometries = [pc]
    if boxes_center is not None:
        assert len(boxes_center) == len(boxes_size)
        for box_id, box in enumerate(boxes_center):
            if boxes_rotation is not None:
                rot = boxes_rotation[box_id]
            else:
                rot = np.eye(3)
            geometries.append(create_obb(box, rot, boxes_size[box_id], box_color))
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    # Read data
    ind = 5
    data = np.load(str(ind) + '.npy')
    data_label = np.load(str(ind) + '_label.npy')
    data_bbox = np.loadtxt(str(ind) + '.txt', dtype='object')

    # Interpret data
    colors = np.zeros((data_label.shape[0], 3))
    colors[data_label.reshape(-1) >= 0, 0] = 1

    data_bbox = (data_bbox[:, 8:14]).astype(float)
    visualize_point_cloud_with_boxes(points=data[:, 0:3], 
                                    colors=colors,
                                    boxes_center=data_bbox[:, 3:6],
                                    boxes_rotation=None,
                                    boxes_size=data_bbox[:, [2, 1, 0]])