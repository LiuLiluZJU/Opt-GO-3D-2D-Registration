####  PYTHON MODULES
import sys
import numpy as np
import time
import torch
import itk
import matplotlib.pyplot as plt

####  MY MODULES
import ReadWriteImageModule as rw
import RigidMotionModule as rm
sys.path.append('../wrapped_module/')
from SiddonGpuPy import pySiddonGpu     # Python wrapped C library for GPU accelerated DRR generation



class SiddonGpu(object):
    """GPU accelearated DRR generation from volumetric image (CT or MRI scan).

    """

    def __init__(self, projector_info):
        """Reads the moving image and creates a siddon projector
                   based on the camera parameters provided in projector_info (dict)
        """

        # ITK: Instantiate types
        Dimension = projector_info['Dimension']
        PixelType = projector_info['PixelType']
        Direction = projector_info['Direction']
        self.Dimension = Dimension
        self.Direction = Direction
        self.ImageType = itk.Image[PixelType, Dimension]
        self.ImageType2D = itk.Image[PixelType, 2]
        self.RegionType = itk.ImageRegion[Dimension]
        PhyImageType = itk.Image[itk.Vector[itk.F, Dimension], Dimension]  # image of physical coordinates

        # Read moving image (CT or MRI scan)
        movingImageFileName = projector_info['movingImageFileName']
        movImage, movImageInfo = rw.ImageReader(movingImageFileName, self.ImageType)
        self.movDirection = movImage.GetDirection()
        print(self.movDirection.GetInverse())

        # Calculate side planes
        X0 = movImageInfo['Volume_center'][0] - movImageInfo['Spacing'][0] * movImageInfo['Size'][0] * 0.5
        Y0 = movImageInfo['Volume_center'][1] - movImageInfo['Spacing'][1] * movImageInfo['Size'][1] / 2.0
        Z0 = movImageInfo['Volume_center'][2] - movImageInfo['Spacing'][2] * movImageInfo['Size'][2] / 2.0
        print(movImageInfo['Spacing'][0] * movImageInfo['Size'][0])
        print(movImageInfo['Spacing'][1] * movImageInfo['Size'][1])
        print(movImageInfo['Spacing'][2] * movImageInfo['Size'][2])

        # Crop spine in moving image (box like ROI)
        if projector_info['Crop_flag'] == 1:
            Crop_start_point = projector_info['Crop_start_point']
            Crop_size = projector_info['Crop_size']
            X0_cropped = X0 + movImageInfo['Spacing'][0] * Crop_start_point[0]
            Y0_cropped = Y0 + movImageInfo['Spacing'][1] * Crop_start_point[1]
            Z0_cropped = Z0 + movImageInfo['Spacing'][2] * Crop_start_point[2]

            X0 = X0_cropped
            Y0 = Y0_cropped
            Z0 = Z0_cropped

            movImgArray_to_crop = itk.GetArrayFromImage(movImage)
            movImgArray_cropped = movImgArray_to_crop[Crop_start_point[2] : Crop_start_point[2] + Crop_size[2],
                                  Crop_start_point[1] : Crop_start_point[1] + Crop_size[1],
                                  Crop_start_point[0] : Crop_start_point[0] + Crop_size[0]]
            movImgArray_1d = np.ravel(movImgArray_cropped.copy(), order='C').astype(np.float32)
            movImageInfo['Size'] = projector_info['Crop_size']
        # Get 1d array for moving image
        else:
            movImgArray_1d = np.ravel(itk.GetArrayFromImage(movImage), order='C').astype(
                np.float32)  # ravel does not generate a copy of the array (it is faster than flatten)

        # Set parameters for GPU library SiddonGpuPy
        NumThreadsPerBlock = np.array([projector_info['threadsPerBlock_x'], projector_info['threadsPerBlock_y'],
                                       projector_info['threadsPerBlock_z']]).astype(np.int32)
        DRRsize_forGpu = np.array([projector_info['DRRsize_x'], projector_info['DRRsize_y'], 1]).astype(np.int32)
        MovSize_forGpu = np.array([movImageInfo['Size'][0], movImageInfo['Size'][1], movImageInfo['Size'][2]]).astype(
            np.int32)
        MovSpacing_forGpu = np.array(
            [movImageInfo['Spacing'][0], movImageInfo['Spacing'][1], movImageInfo['Spacing'][2]]).astype(np.float32)

        # Define source point at its initial position (at the origin = moving image center)
        self.source = [0] * Dimension
        self.source[0] = movImageInfo['Volume_center'][0]
        self.source[1] = movImageInfo['Volume_center'][1]
        self.source[2] = movImageInfo['Volume_center'][2] - projector_info['focal_lenght'] / 2.

        # Define volume center
        self.center = [0] * Dimension
        self.center[0] = movImageInfo['Volume_center'][0]
        self.center[1] = movImageInfo['Volume_center'][1]
        self.center[2] = movImageInfo['Volume_center'][2]

        # Set DRR image at initial position (at +(focal length)/2 along the z direction)
        DRR = self.ImageType.New()
        self.DRRregion = self.RegionType()

        DRRstart = itk.Index[Dimension]()
        DRRstart.Fill(0)

        self.DRRsize = [0] * Dimension
        self.DRRsize[0] = projector_info['DRRsize_x']
        self.DRRsize[1] = projector_info['DRRsize_y']
        self.DRRsize[2] = 1

        self.DRRregion.SetSize(self.DRRsize)
        self.DRRregion.SetIndex(DRRstart)

        self.DRRspacing = itk.Point[itk.F, Dimension]()
        self.DRRspacing[0] = projector_info['DRRspacing_x']
        self.DRRspacing[1] = projector_info['DRRspacing_y']
        self.DRRspacing[2] = 1.

        self.DRRorigin = itk.Point[itk.F, Dimension]()
        self.DRRorigin[0] = movImageInfo['Volume_center'][0] - projector_info['DRR_ppx'] - self.DRRspacing[0] * (
                    self.DRRsize[0] - 1.) / 2.
        self.DRRorigin[1] = movImageInfo['Volume_center'][1] - projector_info['DRR_ppy'] - self.DRRspacing[1] * (
                    self.DRRsize[1] - 1.) / 2.
        self.DRRorigin[2] = movImageInfo['Volume_center'][2] + projector_info['focal_lenght'] / 2.

        DRR.SetRegions(self.DRRregion)
        DRR.Allocate()
        DRR.SetSpacing(self.DRRspacing)
        DRR.SetOrigin(self.DRRorigin)
        self.movDirection.SetIdentity()
        DRR.SetDirection(self.movDirection)

        # Get array of physical coordinates for the DRR at the initial position
        PhysicalPointImagefilter = itk.PhysicalPointImageSource[PhyImageType].New()
        PhysicalPointImagefilter.SetReferenceImage(DRR)
        PhysicalPointImagefilter.SetUseReferenceImage(True)
        PhysicalPointImagefilter.Update()
        sourceDRR = PhysicalPointImagefilter.GetOutput()

        # self.sourceDRR_array_to_reshape = itk.PyBuffer[PhyImageType].GetArrayFromImage(sourceDRR)[0].copy(order = 'C') # array has to be reshaped for matrix multiplication
        self.sourceDRR_array_to_reshape = itk.GetArrayFromImage(sourceDRR)[
            0]  # array has to be reshaped for matrix multiplication
        print(self.sourceDRR_array_to_reshape.shape)

        # Generate projector object
        tGpu1 = time.time()
        self.projector = pySiddonGpu(NumThreadsPerBlock,
                                     movImgArray_1d,
                                     MovSize_forGpu,
                                     MovSpacing_forGpu,
                                     X0.astype(np.float32), Y0.astype(np.float32), Z0.astype(np.float32),
                                     DRRsize_forGpu)
        tGpu2 = time.time()
        print('\nSiddon object initialized. Time elapsed for initialization: ', tGpu2 - tGpu1, '\n')

        # Get array of physical coordinates of the transformed DRR (GPU accelerated)
        Tn = np.array([[1., 0., 0., self.center[0]],
                       [0., 1., 0., self.center[1]],
                       [0., 0., 1., self.center[2]],
                       [0., 0., 0., 1.]])
        invTn = np.linalg.inv(Tn)
        sourceDRR_array_reshaped = self.sourceDRR_array_to_reshape.reshape(
            (self.DRRsize[0] * self.DRRsize[1], self.Dimension), order='C')
        sourceDRR_array_augmented = np.dot(invTn, rm.augment_matrix_coord(sourceDRR_array_reshaped))
        invT = np.zeros((4, 4))
        self.Tn = torch.FloatTensor(Tn).cuda()
        self.sourceDRR_array_augmented = torch.FloatTensor(sourceDRR_array_augmented).cuda()
        self.invT = torch.FloatTensor(invT).cuda()
        

    def update(self, transform_parameters):
        """Updates a DRR given the transform parameters in GPU.

           Args:
               transform_parameters (list of floats): rotX, rotY,rotZ, transX, transY, transZ

        """

        # Get transform parameters
        tic = time.time()
        rotx = transform_parameters[0]
        roty = transform_parameters[1]
        rotz = transform_parameters[2]
        tx = transform_parameters[3]
        ty = transform_parameters[4]
        tz = transform_parameters[5]

        # Compute the transformation matrix and its inverse (itk always needs the inverse)
        if self.Direction == 'ap':
            Tr_ap = rm.get_rigid_motion_mat_from_euler(np.deg2rad(0), 'z', np.deg2rad(0), 'y', np.deg2rad(90), 'x', 0, 0, 0)
            Tr_delta = rm.get_rigid_motion_mat_from_euler(rotz, 'z', roty, 'y', rotx, 'x', tx, ty, tz)
            Tr = np.dot(Tr_delta, Tr_ap)
            # print("ap tr:", Tr)
        elif self.Direction == 'lat':
            Tr_lat = rm.get_rigid_motion_mat_from_euler(np.deg2rad(90), 'x', np.deg2rad(0), 'y', np.deg2rad(90), 'z', 0, 0, 0)
            Tr_delta = rm.get_rigid_motion_mat_from_euler(rotx, 'x', roty, 'y', rotz, 'z', tx, ty, tz)
            Tr = np.dot(Tr_delta, Tr_lat)
            # print("lat tr:", Tr)
        else:
            raise NotImplementedError()
        invT = np.linalg.inv(Tr).astype(np.float32)  # very important conversion to float32, otherwise the code crashes

        # Assign value to inverse matrix (GPU)
        for x in range(4):
            for y in range(4):
                self.invT[x, y] = np.float64(invT[x, y])

        # Move source point with transformation matrix, transform around volume center (subtract volume center point)
        source_transformed = np.dot(invT, np.array(
            [self.source[0] - self.center[0], self.source[1] - self.center[1], self.source[2] - self.center[2], 1.]).T)[
                             0:3]
        source_forGpu = np.array([source_transformed[0] + self.center[0], source_transformed[1] + self.center[1],
                                  source_transformed[2] + self.center[2]], dtype=np.float32)

        # Get array of physical coordinates of the transformed DRR (GPU accelerated)
        sourceDRR_array_augmented_transformed_gpu = torch.matmul(self.invT, self.sourceDRR_array_augmented)
        sourceDRR_array_transformed_gpu = torch.transpose(
            torch.matmul(self.Tn, sourceDRR_array_augmented_transformed_gpu)[0:3], 0, 1)
        sourceDRR_array_transformed = sourceDRR_array_transformed_gpu.cpu().numpy()
        sourceDRR_array_transf_to_ravel = sourceDRR_array_transformed.reshape(
            (self.DRRsize[0], self.DRRsize[1], self.Dimension), order='C')
        DRRPhy_array = np.ravel(sourceDRR_array_transf_to_ravel, order='C').astype(np.float32)

        # Update DRR
        output = self.projector.generateDRR(source_forGpu, DRRPhy_array)
        output_reshaped = np.reshape(output, (self.DRRsize[1], self.DRRsize[0]),
                                     order='C')
        toc = time.time()
        # print("tdrr time:", toc - tic)

        return output_reshaped


    def GO_metric(self, fixed_image):
        """Compute Gradient Orientation metric. Please call compute() first.

            Arg:
                fixed_iamge (type of 2D ITK iamge) : fixed image
        """

        # Get array from X-ray image
        fixed_array_to_ravel = itk.array_from_image(fixed_image)
        fixed_array = np.ravel(fixed_array_to_ravel, order='C').astype(np.float32)

        # Compute GO metric value
        tGpu1 = time.time()
        drr_grad_map, fixed_grad_map = self.projector.computeMetricMedian(fixed_array)
        drr_grad_threshold = np.median(drr_grad_map)
        fixed_grad_threshold = np.median(fixed_grad_map)
        drr_grad_map = np.array(drr_grad_map)
        fixed_grad_map = np.array(fixed_grad_map)
        GO_metric_value = self.projector.computeMetric(fixed_array, drr_grad_threshold, fixed_grad_threshold, 100000)[2]
        tGpu2 = time.time()
        # print("metric time:", tGpu2 - tGpu1)

        # Show gradient maps
        show_grad_map_flag = 0
        if show_grad_map_flag == 1:
            drr_grad_map[drr_grad_map < drr_grad_threshold] = 0
            fixed_grad_map[fixed_grad_map < fixed_grad_threshold] = 0
            k1_reshaped = np.reshape(drr_grad_map, (self.DRRsize[1] - 1, self.DRRsize[0] - 1), order='C')
            k2_reshaped = np.reshape(fixed_grad_map, (self.DRRsize[1] - 1, self.DRRsize[0] - 1), order='C')
            plt.subplot(121)
            plt.imshow(k1_reshaped, cmap='gray')
            plt.subplot(122)
            plt.imshow(k2_reshaped, cmap='gray')
            plt.show()

        return GO_metric_value

    def delete(self):

        """Deletes the projector object >>> GPU is freed <<<"""

        self.projector.delete()
