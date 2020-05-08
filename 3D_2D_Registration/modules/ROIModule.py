####  PYTHON MODULES
import itk
import numpy as np
import matplotlib.pyplot as plt



def CT_ROI_elliptical(CT_image, fiducial_array, senior_axis=50, minor_axis=50):
    """Generate elliptical ROI mask of spine in CT.

    """

    # Initialization
    size_x, size_y, size_z = CT_image.GetBufferedRegion().GetSize()
    spacing_x, spacing_y, spacing_z = CT_image.GetSpacing()
    elliptical_center_list = []

    # Traveling along z direction of CT to find elliptical center points
    for index_z in range(size_z):
        if index_z < fiducial_array[0][2]:
            elliptical_center_list.append((fiducial_array[0][0], fiducial_array[0][1], index_z))
        elif index_z >= fiducial_array[-1][2]:
            elliptical_center_list.append((fiducial_array[-1][0], fiducial_array[-1][1], index_z))
        else:
            # Two-points paradigm of 3D linear function:
            # (x - x1) / (x2 - x1) = (y - y1) / (y2 - y1) = (z - z1) / (z2 - z1)
            # Traveling all fiducial points
            fall_in_a_intervel_flag = 0
            for fiducial_index in range(fiducial_array.shape[0] - 1):
                if fiducial_array[fiducial_index][2] <= index_z and index_z < fiducial_array[fiducial_index + 1][2]:
                    fall_in_a_intervel_flag += 1

                    index_x_lower = fiducial_array[fiducial_index][0]
                    index_x_upper = fiducial_array[fiducial_index + 1][0]
                    index_y_lower = fiducial_array[fiducial_index][1]
                    index_y_upper = fiducial_array[fiducial_index + 1][1]
                    index_z_lower = fiducial_array[fiducial_index][2]
                    index_z_upper = fiducial_array[fiducial_index + 1][2]

                    index_x = index_x_lower + \
                              (index_x_upper - index_x_lower) * \
                              (index_z - index_z_lower) / (index_z_upper - index_z_lower)
                    index_y = index_y_lower + \
                              (index_y_upper - index_y_lower) * \
                              (index_z - index_z_lower) / (index_z_upper - index_z_lower)
                    elliptical_center_list.append((index_x, index_y, index_z))
            if fall_in_a_intervel_flag > 2:
                raise RuntimeError()

    # Generating elliptical binary mask
    elliptical_mask = np.zeros((size_z, size_y, size_x))
    for z in range(size_z):
        for y in range(size_y):
            for x in range(size_x):
                # print(z)
                elliptical_center_point = elliptical_center_list[z]
                x_centered = (x - elliptical_center_point[0])
                y_centered = (y - elliptical_center_point[1])
                ellipitcal_function_value = ((x_centered * spacing_x) / (minor_axis / 2)) ** 2 + \
                                            ((y_centered * spacing_y) / (senior_axis / 2)) ** 2 - 1
                if ellipitcal_function_value < 0:
                    elliptical_mask[z][y][x] = 1

    # Applying mask to CT image
    CT_array = itk.GetArrayFromImage(CT_image)
    CT_array = CT_array.copy()
    CT_masked_array = np.multiply(CT_array, elliptical_mask)
    CT_masked_array_mean_1 = np.mean(CT_masked_array, 1)
    CT_masked_array_mean_1 = np.squeeze(CT_masked_array_mean_1)
    plt.imshow(CT_masked_array_mean_1, cmap='gray')
    plt.show()
    CT_masked_array_mean_2 = np.mean(CT_masked_array, 2)
    CT_masked_array_mean_2 = np.squeeze(CT_masked_array_mean_2)
    plt.imshow(CT_masked_array_mean_2, cmap='gray')
    plt.show()
    CT_masked_array_mean_3 = np.mean(CT_masked_array, 0)
    CT_masked_array_mean_3 = np.squeeze(CT_masked_array_mean_3)
    plt.imshow(CT_masked_array_mean_3, cmap='gray')
    plt.show()
    np.save("elliptical_mask_for_grad_rec.npy", elliptical_mask)


# Test elliptical mask
if __name__ == '__main__':
    reader = itk.ImageFileReader[itk.Image[itk.F, 3]].New()
    reader.SetFileName("/home/leko/Desktop/CT1.mha")
    reader.Update()
    CT_image = reader.GetOutput()

    # Fiducial points in CT volume of example1 
    fiducial_points = np.array([[245, 313, 11],
                                [252, 319, 25],
                                [252, 319, 39],
                                [255, 317, 52],
                                [257, 315, 64],
                                [259, 316, 76],
                                [261, 313, 88],
                                [261, 312, 100],
                                [261, 309, 111],
                                [263, 306, 121],
                                [265, 301, 131],
                                [265, 297, 142],
                                [265, 288, 150]])

    CT_ROI_elliptical(CT_image, fiducial_points)
