import open3d as o3d
import numpy as np
import copy
import os

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp], width=960, height=540)


def preprocess_point_cloud(pcd, voxel_size):

    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=6),
        fast_normal_computation=False)
    radius_feature = voxel_size * 5

    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=20))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 3

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4, [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    return result


def refine_registration(source, target, voxel_size, result_ransac):
    distance_threshold = voxel_size
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return result

if __name__ == "__main__":
    voxel_size = 0.3  # means 5cm for the dataset
    source = o3d.io.read_point_cloud("./model_with_normal.ply")
    source_temp = copy.deepcopy(source)
    source_down = source_temp.voxel_down_sample(voxel_size)
    source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(source_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=20))
    path_pcd = './seg'
    filelist = os.listdir(path_pcd)
    filelist.sort()
    for files in filelist:
        file = os.path.join(path_pcd, files)
        filename, fileType = os.path.splitext(files)
        if fileType == '.pcd':
            print(file)
            target = o3d.io.read_point_cloud(file)

            target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

            result_ransac = execute_global_registration(source_down, target_down,
                                                        source_fpfh, target_fpfh,
                                                        voxel_size)
            draw_registration_result(source, target, result_ransac.transformation)
            result_icp = refine_registration(source, target, voxel_size, result_ransac)

            file_pos = os.path.join(path_pcd, filename[0:9]+'_pos_pre.txt')
            np.savetxt(file_pos,result_icp.transformation)
            # draw_registration_result(source, target, result_icp.transformation)
            # break
