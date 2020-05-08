####  PYTHON MODULES
import itk
import sys
import time
import numpy as np
import cma
import os
import matplotlib.pyplot as plt

####  MY MODULES
sys.path.append('../modules/')
import ProjectorsModule_multi as pj



def space_partition(start_point, search_range, space_num=50, dim=6):
    """K-d tree partition algorithm.

    """

    # Initialise normalized space and voxel center coordinate
    space = [1.] * dim
    center = [[0.] * dim for _ in range(space_num)]
    space_with_center = [[0.] * dim for _ in range(space_num)]

    # Binary partition of whole space
    count = 0
    while(2 ** (count + 1) < space_num):
        max_dim_index = space.index(max(space))
        space[max_dim_index] = space[max_dim_index] / 2
        for i in range(2 ** count - 1, -1, -1):
            far_index = 2 * i + 1
            near_index = 2 * i
            center[far_index][:] = center[i][:]
            center[near_index][:] = center[i][:]
            center[far_index][max_dim_index] += space[max_dim_index] / 2
            center[near_index][max_dim_index] -= space[max_dim_index] / 2
            space_with_center[far_index][:] = space[:]
            space_with_center[near_index][:] = space[:]
        count += 1

    # Continue partitioning up to space number
    count_final = 2 ** count
    max_dim_index = space.index(max(space))
    space[max_dim_index] = space[max_dim_index] / 2
    for i in range(2 ** count - 1, -1, -1):
        far_index = 2 * i - (2 * (2 ** count - 1) - (space_num - 1))
        near_index = 2 * i - (2 * (2 ** count - 1) - (space_num - 1)) - 1
        center[far_index][:] = center[i][:]
        center[near_index][:] = center[i][:]
        center[far_index][max_dim_index] += space[max_dim_index] / 2
        center[near_index][max_dim_index] -= space[max_dim_index] / 2
        space_with_center[far_index][:] = space[:]
        space_with_center[near_index][:] = space[:]
        count_final += 1
        if(count_final >= space_num):
            break
    center_np = np.array(center)
    space_with_center_np = np.array(space_with_center)
    real_center = center_np + np.tile(start_point, (np.size(center_np, 0), 1))
    real_space_with_center = space_with_center_np

    return real_center, real_space_with_center


def cost_function(x):
    """Calculate costs in each iteration (CMA-ES).

    """

    global countcount, max_result, search_range, x_ray_image_ap, x_ray_image_lat
    countcount += 1
    print(countcount)
    paras = np.multiply(x, search_range)
    # drr1 = projector_ap.update([np.deg2rad(paras[0]), np.deg2rad(paras[1]), np.deg2rad(paras[2]), paras[3], paras[4], 728 + paras[5]])
    # result1 = projector_ap.GO_metric(x_ray_image_ap)
    drr2 = projector_lat.update([np.deg2rad(paras[2]), np.deg2rad(paras[1]), np.deg2rad(-paras[0]), paras[5], paras[4], 699 - paras[3]])
    result2 = projector_lat.GO_metric(x_ray_image_lat)

    result = result2
    if countcount % 500 == 0:
        # plt.subplot(121)
        # plt.imshow(drr1, cmap='gray')
        plt.subplot(122)
        plt.imshow(drr2, cmap='gray')
        plt.show()
        max_result = result
    print("x: [np.deg2rad(%f), np.deg2rad(%f), np.deg2rad(%f), %f, %f, %f]"%tuple(paras))
    # print("result1:", result1)
    print("result2:", result2)
    print("result:", result)
    return result


def local_optimize(name, start, lb, ub):
    """Optimize locally with each strat (CMA-ES).

    """

    print('Run task %s (%s)...' % (name, os.getpid()))
    global countcount
    countcount = 0
    es = cma.CMAEvolutionStrategy(start, 0.2, {'bounds': [lb, ub], 'tolfun': 0.001, 'maxiter': 10})
    es.optimize(cost_function)
    res = es.result
    print("result:", res)
    return res.fbest, res.xbest


if __name__ == '__main__':
    # Define projector for generation of DRR from 3D model (Digitally Reconstructed Radiographs)
    projector_info_ap = {'Name': 'SiddonGpu',
                        'Direction': 'ap',
                        'movingImageFileName': '/home/leko/Desktop/CT1.mha',
                        'Dimension': 3,
                        'PixelType': itk.F,
                        'threadsPerBlock_x': 16,
                        'threadsPerBlock_y': 16,
                        'threadsPerBlock_z': 1,
                        'DRRsize_x': 500,
                        'DRRsize_y': 2000,
                        'focal_lenght': 2000,
                        'DRR_ppx': -10,  # Physical length(mm)
                        'DRR_ppy': 10,  # Physical length(mm)
                        'DRRspacing_x': 0.143,
                        'DRRspacing_y': 0.143,
                        'Crop_flag': 1,
                        'Crop_start_point': [180, 250, 0],  # Order XYZ
                        'Crop_size': [150, 150, 159]  # Order XYZ
                        }

    projector_info_lat = {'Name': 'SiddonGpu',
                        'Direction': 'lat',
                        'movingImageFileName': '/home/leko/Desktop/CT1.mha',
                        'Dimension': 3,
                        'PixelType': itk.F,
                        'threadsPerBlock_x': 16,
                        'threadsPerBlock_y': 16,
                        'threadsPerBlock_z': 1,
                        'DRRsize_x': 400,
                        'DRRsize_y': 1600,
                        'focal_lenght': 2000,
                        'DRR_ppx': 70,  # Physical length(mm)
                        'DRR_ppy': 0,  # Physical length(mm)
                        'DRRspacing_x': 0.143,
                        'DRRspacing_y': 0.143,
                        'Crop_flag': 1,
                        'Crop_start_point': [180, 250, 0],  # Order XYZ
                        'Crop_size': [150, 150, 159]  # Order XYZ
                        }
    Verbose_full_flag = 0  # Show full image or not

    # Read X-rays
    reader_ap = itk.ImageFileReader[itk.Image[itk.F, 2]].New()
    reader_ap.SetFileName("/home/leko/Desktop/x_ray1.DCM")
    reader_ap.Update()
    x_ray_ap = reader_ap.GetOutput()
    x_ray_array_ap = itk.GetArrayFromImage(x_ray_ap)
    x_ray_array_ap = -x_ray_array_ap
    hight_ap, width_ap = x_ray_array_ap.shape
    print(x_ray_array_ap.shape)
    width_ap_offset = 10 / 0.143
    hight_ap_offset = -10 / 0.143
    x_ray_cropped_ap = x_ray_array_ap[int((hight_ap - 2000) / 2 + hight_ap_offset) : int((hight_ap + 2000) / 2 + hight_ap_offset),
                    int((width_ap - 500) / 2 + width_ap_offset) : int((width_ap + 500) / 2 + width_ap_offset)]
    x_ray_image_ap = itk.image_from_array(x_ray_cropped_ap.copy())
    x_ray_image_ap.SetSpacing([0.143, 0.143])

    reader_lat = itk.ImageFileReader[itk.Image[itk.F, 2]].New()
    reader_lat.SetFileName("/home/leko/Desktop/x_ray2.DCM")
    reader_lat.Update()
    x_ray_lat = reader_lat.GetOutput()
    x_ray_array_lat = itk.GetArrayFromImage(x_ray_lat)
    x_ray_array_lat = -x_ray_array_lat
    hight_lat, width_lat = x_ray_array_lat.shape
    width_lat_offset = -70 / 0.143
    x_ray_cropped_lat = x_ray_array_lat[int((hight_lat - 1600) / 2) : int((hight_lat + 1600) / 2),
                        int((width_lat - 400) / 2 + width_lat_offset) : int((width_lat + 400) / 2 + width_lat_offset)]
    x_ray_image_lat = itk.image_from_array(x_ray_cropped_lat.copy())
    x_ray_cropped_lat_save = (x_ray_cropped_lat - np.min(x_ray_cropped_lat)) / (np.max(x_ray_cropped_lat) - np.min(x_ray_cropped_lat)) * 255
    x_ray_image_lat.SetSpacing([0.143, 0.143])

    # Show cropped X-rays
    plt.subplot(121)
    plt.imshow(x_ray_array_ap, cmap='gray')
    plt.subplot(122)
    plt.imshow(x_ray_array_lat, cmap='gray')
    plt.show()
    plt.subplot(121)
    plt.imshow(x_ray_cropped_lat, cmap='gray')
    plt.subplot(122)
    plt.imshow(x_ray_cropped_ap, cmap='gray')
    plt.show()

    # Fully show or not
    if Verbose_full_flag == 1:
        projector_info_ap['DRRsize_x'] = 2552
        projector_info_ap['DRRsize_y'] = 2988
        projector_info_ap['DRR_ppx'] = 0  # Physical length(mm)
        projector_info_ap['DRR_ppy'] = 0  # Physical length(mm)
        projector_info_ap['Crop_flag'] = 0
        projector_info_lat['DRRsize_x'] = 2552
        projector_info_lat['DRRsize_y'] = 2984
        projector_info_lat['DRR_ppx'] = 0  # Physical length(mm)
        projector_info_lat['DRR_ppy'] = 0  # Physical length(mm)
        projector_info_lat['Crop_flag'] = 0

    # Initialise DRR projectors
    projector_ap = pj.SiddonGpu(projector_info_ap)
    projector_lat = pj.SiddonGpu(projector_info_lat)

    # Show initial DRRs
    drr1 = projector_ap.update([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0, 0, 728])
    drr2 = projector_lat.update([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0), 0, 0, 699])
    plt.subplot(121)
    plt.imshow(drr1, cmap='gray')
    plt.subplot(122)
    plt.imshow(drr2, cmap='gray')
    plt.show()

    # Show checkerboard image
    if Verbose_full_flag == 1:
        width_interval = projector_info_ap['DRRsize_x'] / 4
        hight_interval = projector_info_ap['DRRsize_y'] / 4
        checerboard_image = np.zeros((projector_info_ap['DRRsize_y'], projector_info_ap['DRRsize_x']))
        checerboard_x_ray = (x_ray_array_ap - np.min(x_ray_array_ap)) / (np.max(x_ray_array_ap) - np.min(x_ray_array_ap))
        checerboard_drr = (drr1 - np.min(drr1)) / (np.max(drr1) - np.min(drr1))
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    checerboard_image[int(i * hight_interval) : int((i + 1) * hight_interval), int(j * width_interval) : int((j + 1) * width_interval)] = \
                        checerboard_x_ray[int(i * hight_interval) : int((i + 1) * hight_interval), int(j * width_interval) : int((j + 1) * width_interval)]
                else:
                    checerboard_image[int(i * hight_interval): int((i + 1) * hight_interval),int(j * width_interval): int((j + 1) * width_interval)] = \
                        checerboard_drr[int(i * hight_interval): int((i + 1) * hight_interval),int(j * width_interval): int((j + 1) * width_interval)]
        plt.imshow(checerboard_image, cmap='gray')
        plt.show()

        width_interval = projector_info_lat['DRRsize_x'] / 4
        hight_interval = projector_info_lat['DRRsize_y'] / 4
        checerboard_image = np.zeros((projector_info_lat['DRRsize_y'], projector_info_lat['DRRsize_x']))
        checerboard_x_ray = (x_ray_array_lat - np.min(x_ray_array_lat)) / (np.max(x_ray_array_lat) - np.min(x_ray_array_lat))
        checerboard_drr = (drr2 - np.min(drr2)) / (np.max(drr2) - np.min(drr2))
        for i in range(4):
            for j in range(4):
                if (i + j) % 2 == 0:
                    checerboard_image[int(i * hight_interval): int((i + 1) * hight_interval),
                    int(j * width_interval): int((j + 1) * width_interval)] = \
                        checerboard_x_ray[int(i * hight_interval): int((i + 1) * hight_interval),
                        int(j * width_interval): int((j + 1) * width_interval)]
                else:
                    checerboard_image[int(i * hight_interval): int((i + 1) * hight_interval),
                    int(j * width_interval): int((j + 1) * width_interval)] = \
                        checerboard_drr[int(i * hight_interval): int((i + 1) * hight_interval),
                        int(j * width_interval): int((j + 1) * width_interval)]
        plt.imshow(checerboard_image, cmap='gray')
        plt.show()
        sys.exit()

    # Initialize optimizer
    tic = time.time()
    countcount = 0
    max_result = 0
    space_num = 50
    start_point = [0, 0, 0, 10, 15, 10]  # [alpha(deg), beta(deg), gamma(deg), X(mm), Y(mm), Z(mm)]
    search_range = [20, 20, 20, 200, 400, 200]
    dimension = len(start_point)
    l_bounds = [start_point[i] - search_range[i] / 2 for i in range(0, len(start_point))]
    u_bounds = [start_point[i] + search_range[i] / 2 for i in range(0, len(start_point))]
    start_point_norm = [start_point[i] / search_range[i] for i in range(0, len(start_point))]
    l_bounds_norm = [l_bounds[i] / search_range[i] for i in range(0, len(l_bounds))]
    u_bounds_norm = [u_bounds[i] / search_range[i] for i in range(0, len(u_bounds))]
    center_to_optimize, space_with_center_to_optimize = space_partition(start_point_norm, search_range, space_num=space_num, dim=dimension)

    # Global multi-starts optimization
    multistrats_result = 5
    multistrats_x = 0
    multistarts_lb = []
    multistarts_ub = []
    for i in range(space_num):
        current_result, current_x = local_optimize(i, [center_to_optimize[i][j] for j in range(dimension)],
                                        [center_to_optimize[i][j] - space_with_center_to_optimize[i][j] / 2 for j in range(dimension)],
                                        [center_to_optimize[i][j] + space_with_center_to_optimize[i][j] / 2 for j in range(dimension)])
        if(current_result < multistrats_result):
            multistrats_result = current_result
            multistrats_x = current_x
            multistarts_lb = [center_to_optimize[i][j] - space_with_center_to_optimize[i][j] / 2 for j in range(dimension)]
            multistarts_ub = [center_to_optimize[i][j] + space_with_center_to_optimize[i][j] / 2 for j in range(dimension)]
    print("final result, final x:", multistrats_result, multistrats_x)

    # Local restart optimization
    es_final = cma.CMAEvolutionStrategy(multistrats_x, 0.1, {'bounds': [multistarts_lb, multistarts_ub], 'tolfun': 0.0001})
    es_final.optimize(cost_function)
    res = es_final.result
    print("result:", res)

    # Show results
    delta_paras = np.multiply(res.xbest[0:6], search_range[0:6])
    drr_final_ap = projector_lat.update([np.deg2rad(delta_paras[2]), np.deg2rad(delta_paras[1]), -np.deg2rad(delta_paras[0]), delta_paras[5], delta_paras[4], 699 - delta_paras[3]])
    plt.subplot(121)
    plt.imshow(x_ray_cropped_lat, cmap='gray')
    plt.subplot(122)
    plt.imshow(drr_final_ap, cmap='gray')
    plt.show()
    toc = time.time()
    print("total time:", toc - tic)
