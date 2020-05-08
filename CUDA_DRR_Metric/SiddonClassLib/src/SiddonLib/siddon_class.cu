#include <cstdio>
#include "siddon_class.cuh"
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__device__ const float epsilon = 2.22045e-016; // to compare double float values

// auxiliary functions

__device__ void get_dest(int idx, float *dest_array, float *dest) {

	dest[0] = dest_array[0 + 3 * idx];
	dest[1] = dest_array[1 + 3 * idx];
	dest[2] = dest_array[2 + 3 * idx];

}


__device__ void compute_alpha_x(const float &X0,
	const float &spacing_x,
	const int &i,
	const float &source_x,
	const float &dest_x,
	float &alpha_x) {

	alpha_x = ((X0 + static_cast<float>(i)*spacing_x) - source_x) / (dest_x - source_x);

}


__device__ void compute_alpha_y(const float &Y0,
	const float &spacing_y,
	const int &j,
	const float &source_y,
	const float &dest_y,
	float &alpha_y) {

	alpha_y = ((Y0 + static_cast<float>(j)*spacing_y) - source_y) / (dest_y - source_y);
}


__device__ void compute_alpha_z(const float &Z0,
	const float &spacing_z,
	const int &k,
	const float &source_z,
	const float &dest_z,
	float &alpha_z) {

	alpha_z = ((Z0 + static_cast<float>(k)*spacing_z) - source_z) / (dest_z - source_z);
}


__device__ void compute_phi_x(const float &X0,
	const float &spacing_x,
	float &alpha,
	const float &source_x,
	const float &dest_x,
	float &phi_x) {

	phi_x = (source_x + alpha*(dest_x - source_x) - X0) / spacing_x;
}


__device__ void compute_phi_y(const float &Y0,
	const float &spacing_y,
	float &alpha,
	const float &source_y,
	const float &dest_y,
	float &phi_y) {

	phi_y = (source_y + alpha*(dest_y - source_y) - Y0) / spacing_y;
}


__device__ void compute_phi_z(const float &Z0,
	const float &spacing_z,
	float &alpha,
	const float &source_z,
	const float &dest_z,
	float &phi_z) {

	phi_z = (source_z + alpha*(dest_z - source_z) - Z0) / spacing_z;
}

__device__ void update_idx(unsigned int &i_v, unsigned int &j_v, unsigned int &k_v, const int &size_x, const int &size_y, int &arrayIdx) {

	arrayIdx = i_v + size_x * (j_v + size_y * k_v);
}


__global__ void generateDRR_kernel(float *DRRarray,
	float *source,
	float *DestArray,
	int DRRsize0,
	int DRRsize1,
	float *movImgArray,
	int *MovSize,
	float *MovSpacing,
	float X0, float Y0, float Z0) {

	// DRR image indeces
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// DRR array index
	int DRRidx = row + DRRsize0 * col;

	// moving image total size
	int totalMovSize = MovSize[0] * MovSize[1] * MovSize[2];

	// printf("Thread index %i\n", DRRidx);
	// printf("row, col, row * col: %d, %d, %d\n", DRRsize0, DRRsize1, DRRsize0 * DRRsize1);

	if (DRRidx < DRRsize0 * DRRsize1) { // checks if thread index is within the length of the DRR array

		// --- declaration of variables for Siddon ---
		float alpha_min, alpha_max;
		float alpha_x_min, alpha_x_max, alpha_y_min, alpha_y_max, alpha_z_min, alpha_z_max;
		int i_min, i_max, j_min, j_max, k_min, k_max; // indeces corresponding to first and last intersected voxels
		float alpha_current;
		float alpha_x_next;
		float alpha_y_next;
		float alpha_z_next;
		float distance;
		int arrayIdx;
		int arrayIdx_old;
		unsigned int i_v, j_v, k_v;
		float alpha_first_pixel;
		float density_value = 0.;
		
		// --- initialize DRR's density value ---
		DRRarray[DRRidx] = 0;

		// --- define destination point based on DRR array index --- 
		float dest[3];
		get_dest(DRRidx, DestArray, dest);

		// --- source-to-destination distance --- 
		distance = sqrtf((dest[0] - source[0])*(dest[0] - source[0]) +
			(dest[1] - source[1])*(dest[1] - source[1]) +
			(dest[2] - source[2])*(dest[2] - source[2]));

		float dx = MovSpacing[0] / fabsf(dest[0] - source[0]);
		float dy = MovSpacing[1] / fabsf(dest[1] - source[1]);
		float dz = MovSpacing[2] / fabsf(dest[2] - source[2]);

		// --- find alpha_min and alpha_max
		// initialize alpha_min and alpha_max
		alpha_min = 0.;
		alpha_max = 1.;

		// X
		if (fabsf(dest[0] - source[0]) > epsilon) {

			float alpha_x0 = (X0 - source[0]) / (dest[0] - source[0]);
			float alpha_xN;
			compute_alpha_x(X0, MovSpacing[0], MovSize[0], source[0], dest[0], alpha_xN);
			alpha_x_min = fminf(alpha_x0, alpha_xN);
			alpha_x_max = fmaxf(alpha_x0, alpha_xN);
			if (alpha_x_min > alpha_min) { alpha_min = alpha_x_min; };
			if (alpha_x_max < alpha_max) { alpha_max = alpha_x_max; };
			//printf("alpha_min: %f\n", alpha_min);
			//printf("alpha_max: %f\n", alpha_max);

		}

		// Y
		if (fabsf(dest[1] - source[1]) > epsilon) {

			float alpha_y0 = (Y0 - source[1]) / (dest[1] - source[1]);
			float alpha_yN;
			compute_alpha_y(Y0, MovSpacing[1], MovSize[1], source[1], dest[1], alpha_yN);
			alpha_y_min = fminf(alpha_y0, alpha_yN);
			alpha_y_max = fmaxf(alpha_y0, alpha_yN);
			if (alpha_y_min > alpha_min) { alpha_min = alpha_y_min; };
			if (alpha_y_max < alpha_max) { alpha_max = alpha_y_max; };

		}

		// Z
		if (fabsf(dest[2] - source[2]) > epsilon) {

			float alpha_z0 = (Z0 - source[2]) / (dest[2] - source[2]);
			float alpha_zN;
			compute_alpha_z(Z0, MovSpacing[2], MovSize[2], source[2], dest[2], alpha_zN);
			alpha_z_min = fminf(alpha_z0, alpha_zN);
			alpha_z_max = fmaxf(alpha_z0, alpha_zN);
			if (alpha_z_min > alpha_min) { alpha_min = alpha_z_min; };
			if (alpha_z_max < alpha_max) { alpha_max = alpha_z_max; };

		}

		// if (DRRidx == 0){
		// printf("Alpha min = %f\n", alpha_min);
		// printf("Alpha max = %f\n", alpha_max);
		// printf("dx = %f\n", dx);
		// printf("dy = %f\n", dy);
		// printf("dz = %f\n", dz);
		// printf("distance = %f\n", distance);
		// }

		// --- initialize alpha --- 
		alpha_current = alpha_min;

		if (alpha_min < alpha_max) {

			// compute i_min, i_max and initialize alpha_x_next 
			if (dest[0] - source[0] > 0.) {

				// i_min
				if (fabsf(alpha_min - alpha_x_min) < epsilon) { i_min = 1; } // in other words: if source point is out of 3D model(x plane parallel region)
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_min, source[0], dest[0], phi_x);
					i_min = ceil(phi_x);
				}

				// i_max
				if (fabsf(alpha_max - alpha_x_max) < epsilon) { i_max = MovSize[0] - 1; } // in other words: if destination point is out of 3D model(x plane parallel region)
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_max, source[0], dest[0], phi_x);
					i_max = floor(phi_x);
				}

				// initialize alpha_x_next
				compute_alpha_x(X0, MovSpacing[0], i_min, source[0], dest[0], alpha_x_next);
			}
			else {

				// i_max
				if (fabsf(alpha_min - alpha_x_min) < epsilon) { i_max = MovSize[0] - 1; }
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_min, source[0], dest[0], phi_x);
					i_max = floor(phi_x);
				}

				// i_min
				if (fabsf(alpha_max - alpha_x_max) < epsilon) { i_min = 1; }
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_max, source[0], dest[0], phi_x);
					i_min = ceil(phi_x);
				}

				// initialize alpha_x_next
				compute_alpha_x(X0, MovSpacing[0], i_max, source[0], dest[0], alpha_x_next);
			}

			// compute j_min, j_max and initialize alpha_y_next 
			if (dest[1] - source[1] > 0.) {

				// j_min
				if (fabsf(alpha_min - alpha_y_min) < epsilon) { j_min = 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_min, source[1], dest[1], phi_y);
					//printf("phi_y: %f\n", phi_y);
					j_min = ceil(phi_y);
					//printf("j_min: %f\n", j_min);
				}

				// j_max
				if (fabsf(alpha_max - alpha_y_max) < epsilon) { j_max = MovSize[1] - 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_max, source[1], dest[1], phi_y);
					j_max = floor(phi_y);
				}

				// initialize alpha_y_next
				compute_alpha_y(Y0, MovSpacing[1], j_min, source[1], dest[1], alpha_y_next);
			}
			else {

				// j_max
				if (fabsf(alpha_min - alpha_y_min) < epsilon) { j_max = MovSize[1] - 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_min, source[1], dest[1], phi_y);
					j_max = floor(phi_y);
				}

				// j_min
				if (fabsf(alpha_max - alpha_y_max) < epsilon) { j_min = 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_max, source[1], dest[1], phi_y);
					j_min = ceil(phi_y);
				}

				// initialize alpha_y_next
				compute_alpha_y(Y0, MovSpacing[1], j_max, source[1], dest[1], alpha_y_next);
			}

			// compute k_min, k_max and initialize alpha_z_next 
			if (dest[2] - source[2] > 0.) {

				// k_min
				if (fabsf(alpha_min - alpha_z_min) < epsilon) { k_min = 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_min, source[2], dest[2], phi_z);
					k_min = ceil(phi_z);
				}

				// k_max
				if (fabsf(alpha_max - alpha_z_max) < epsilon) { k_max = MovSize[2] - 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_max, source[2], dest[2], phi_z);
					k_max = floor(phi_z);
				}

				// initialize alpha_z_next
				compute_alpha_z(Z0, MovSpacing[2], k_min, source[2], dest[2], alpha_z_next);
			}
			else {

				// k_max
				if (fabsf(alpha_min - alpha_z_min) < epsilon) { k_max = MovSize[2] - 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_min, source[2], dest[2], phi_z);
					k_max = floor(phi_z);
				}

				// k_min
				if (fabsf(alpha_max - alpha_z_max) < epsilon) { k_min = 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_max, source[2], dest[2], phi_z);
					k_min = ceil(phi_z);
				}

				// initialize alpha_z_next
				compute_alpha_z(Z0, MovSpacing[2], k_max, source[2], dest[2], alpha_z_next);
			}

			// if (DRRidx == 0) {
			// 	printf("i_min, i_max, Alpha_x_next = %d %d %f\n", i_min, i_max, alpha_x_next);
			// 	printf("j_min, j_max, Alpha_y_next = %d %d %f\n", j_min, j_max, alpha_y_next);
			// 	printf("k_min, k_max, Alpha_z_next = %d %d %f\n", k_min, k_max, alpha_z_next);
			// }

			// --- initialize first intersected pixel i_v, j_v, k_v --- 
			if ((alpha_y_next < alpha_x_next) && (alpha_y_next < alpha_z_next)) {

				alpha_first_pixel = (alpha_y_next + alpha_min) / 2.;
			}
			else if (alpha_x_next < alpha_z_next) {

				alpha_first_pixel = (alpha_x_next + alpha_min) / 2.;
			}
			else {

				alpha_first_pixel = (alpha_z_next + alpha_min) / 2.;
			}


			float phi_x = 0.;
			float phi_y = 0.;
			float phi_z = 0.;
			compute_phi_x(X0, MovSpacing[0], alpha_first_pixel, source[0], dest[0], phi_x);
			i_v = floor(phi_x);
			compute_phi_y(Y0, MovSpacing[1], alpha_first_pixel, source[1], dest[1], phi_y);
			j_v = floor(phi_y);
			compute_phi_z(Z0, MovSpacing[2], alpha_first_pixel, source[2], dest[2], phi_z);
			k_v = floor(phi_z);

			// initialize array index of first intersected pixel
			arrayIdx = i_v + MovSize[0] * (j_v + MovSize[1] * k_v);
			arrayIdx_old = i_v + MovSize[0] * (j_v + MovSize[1] * k_v);
			
			//printf("pixel: %f\n", movImgArray[50 + MovSize[0] * (40 + MovSize[1] * 60)]);

			// if (DRRidx == 0) {
			// 	printf("i_v, j_v, k_v = %d %d %d\n", i_v, j_v, k_v);
			// 	printf("arrayIdx, arrayIdx_old = %d %d\n", arrayIdx, arrayIdx_old);
			// }

			// iterator indeces
			int stop = (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1);
			int iter = 0;

			//while (alpha_current < 1. && alpha_current < alpha_max) {
			while (iter < stop) {

				float l;

				// next intersection plane is y
				if ((alpha_y_next < alpha_x_next) && (alpha_y_next < alpha_z_next)) {

					//T alpha_mid = (alpha_current + alpha_y_next) / 2.;
					l = (alpha_y_next - alpha_current);

					alpha_current = alpha_y_next;

					// update
					alpha_y_next += dy;
					j_v += (dest[1] - source[1] > 0.) ? 1 : -1;

				}

				else if (alpha_x_next < alpha_z_next) {

					// next intersection plane is x
					//T alpha_mid = (alpha_current + alpha_x_next) / 2.;
					l = (alpha_x_next - alpha_current);

					alpha_current = alpha_x_next;

					// update
					alpha_x_next += dx;
					i_v += (dest[0] - source[0] > 0.) ? 1 : -1;


				}

				else {

					// next intersection plane is z
					//T alpha_mid = (alpha_current + alpha_z_next) / 2.;
					l = (alpha_z_next - alpha_current);

					alpha_current = alpha_z_next;

					// update
					alpha_z_next += dz;
					k_v += (dest[2] - source[2] > 0.) ? 1 : -1;
				}

				// update array index
				update_idx(i_v, j_v, k_v, MovSize[0], MovSize[1], arrayIdx);

				//if (arrayIdx < 0) {
				//	printf("arrayIdx negative! %i", arrayIdx);
				//}

				// if (arrayIdx > MovSize[0] * MovSize[1] * MovSize[2]){
				// 	printf("OUT OF BOUND!!!");
				// 	printf("arrayIdx:%d", arrayIdx);
				// }

				// if (DRRidx == 0){
				// 	printf("arrayIdx:%d\n", arrayIdx);
				// }

				if (0 < arrayIdx_old && arrayIdx_old < totalMovSize){
				// update density value
					if ((movImgArray[arrayIdx_old]) > 0.) { // Note: don't use fabsf(movImgArray[arrayIdx_old]) > epsilon !!!

						density_value += (movImgArray[arrayIdx_old]) * l;
						
						// if(DRRidx == 134885)
						// {
						// 	printf("l: %f\n", l);
						// 	printf("movImgArray[arrayIdx_old]: %f\n", movImgArray[arrayIdx_old]);
						// 	printf("density_value: %f\n", density_value);
						// }
						
						}
				}

				// update arrayIdx
				arrayIdx_old = arrayIdx;

				// update iter
				iter += 1;


			}

			// multiply by the distance
			density_value *= distance;
			//if(DRRidx == 134885)
			//{
			//	printf("final density value: %f\n", density_value);
			//}
			//std::cout << density_value << std::endl;

		}

		// update density value array
		// if (DRRidx < 0){
		// 	printf("OUT OF BOUND!!!");
		// }

		DRRarray[DRRidx] = density_value;
		

		// if(DRRidx == 0)
		// {
		// 	printf("finalfinal density value: %f\n", DRRarray[DRRidx]);
		// }

	}

}

__global__ void computeMetric_kernel(float *weightSum,
	int *weightNum,
	int DRRsize0,
	int DRRsize1,
	float *DRRArray,
	float FixedThreshold,
	float DRRThreshold,
	float *FixedArray) {

	// DRR and fixed image indeces
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// DRR and fixed array index
	int idx = row + DRRsize0 * col;

	// printf("Thread index %i\n", DRRidx);
	// printf("row, col: %d, %d\n", row, col);

	const int s_idx = threadIdx.y * blockDim.x + threadIdx.x;
	
	// allocate shared memory in each block
	__shared__ float s_sum[256];
	__shared__ int s_num[256];

	s_sum[s_idx] = 0;
	s_num[s_idx] = 0;

	if (DRRsize0 < idx && idx < DRRsize0 * (DRRsize1 - 1) && 
		idx % DRRsize0 != 0 && 
		idx % DRRsize0 != DRRsize0 - 1
		) { // checks if thread index is within the length of the DRR array

		// printf("index:%d\n", idx);

		float drrPixel = DRRArray[idx];
		float drrPixel_left = DRRArray[idx - 1];
		float drrPixel_right = DRRArray[idx + 1];
		float drrPixel_top = DRRArray[idx - DRRsize0];
		float drrPixel_bottom = DRRArray[idx + DRRsize0];
		float drrPixel_left_top = DRRArray[idx - DRRsize0 - 1];
		float drrPixel_right_top = DRRArray[idx - DRRsize0 + 1];
		float drrPixel_left_bottom = DRRArray[idx + DRRsize0 - 1];
		float drrPixel_right_bottom = DRRArray[idx + DRRsize0 + 1];
		float fixedPixel = FixedArray[idx];
		float fixedPixel_left = FixedArray[idx - 1];
		float fixedPixel_right = FixedArray[idx + 1];
		float fixedPixel_top = FixedArray[idx - DRRsize0];
		float fixedPixel_bottom = FixedArray[idx + DRRsize0];
		float fixedPixel_left_top = FixedArray[idx - DRRsize0 - 1];
		float fixedPixel_right_top = FixedArray[idx - DRRsize0 + 1];
		float fixedPixel_left_bottom = FixedArray[idx + DRRsize0 - 1];
		float fixedPixel_right_bottom = FixedArray[idx + DRRsize0 + 1];

		// if(idx == 599) {
		// 	printf("drrPixel:%f\n", drrPixel);
		// 	printf("drrPixel:%f\n", drrPixel_left);
		// 	printf("drrPixel:%f\n", drrPixel_right);
		// 	printf("drrPixel:%f\n", drrPixel_top);
		// 	printf("drrPixel:%f\n", drrPixel_bottom);
		// 	printf("fixedPixel:%f\n", fixedPixel);
		// 	printf("fixedPixel:%f\n", fixedPixel_left);
		// 	printf("fixedPixel:%f\n", fixedPixel_right);
		// 	printf("fixedPixel:%f\n", fixedPixel_top);
		// 	printf("fixedPixel:%f\n", fixedPixel_bottom);
		// }

		float m_dx = (drrPixel_right_top + 2 * drrPixel_right + drrPixel_right_bottom) - (drrPixel_left_top + 2 * drrPixel_left + drrPixel_left_bottom);
		float m_dy = (drrPixel_left_bottom + 2 * drrPixel_bottom + drrPixel_right_bottom) - (drrPixel_left_top + 2 * drrPixel_top + drrPixel_right_top);
		float f_dx = (fixedPixel_right_top + 2 * fixedPixel_right + fixedPixel_right_bottom) - (fixedPixel_left_top + 2 * fixedPixel_left + fixedPixel_left_bottom);
		float f_dy = (fixedPixel_left_bottom + 2 * fixedPixel_bottom + fixedPixel_right_bottom) - (fixedPixel_left_top + 2 * fixedPixel_top + fixedPixel_right_top);

		float grad_m_mod = sqrtf(m_dx * m_dx + m_dy * m_dy);
		float grad_f_mod = sqrtf(f_dx * f_dx + f_dy * f_dy);

		// printf("grad_m_mod = %f, grad_f_mod = %f\n", grad_m_mod, grad_f_mod);

		if(grad_m_mod > DRRThreshold && grad_f_mod > FixedThreshold)
		{
			// int current_num = weightNum[0];
			s_num[s_idx] = 1;
			// num++;

			float cos_theta = (f_dx * m_dx + f_dy * m_dy) / (grad_f_mod * grad_m_mod);
			if(cos_theta > 1) cos_theta = 1;
			if(cos_theta < -1) cos_theta = -1;

			float w = (2 - log(fabsf(acos(cos_theta)) + 1)) / 2;
			// float current_sum = weightSum[0];
			
			s_sum[s_idx] = w;

			// printf("%d\n", *weightNum);
			// printf("w = %f\n", w);
		}
	}
	__syncthreads();

	if (s_idx == 0) {
		float block_sum = 0;
		int block_num = 0;
		for(int j = 0; j < blockDim.x * blockDim.y; ++j) {
			block_sum += s_sum[j];
			block_num += s_num[j];
		}
		// printf("block_sum = %f, block_num = %d\n", block_sum, block_num);
		atomicAdd(weightSum, block_sum);
		atomicAdd(weightNum, block_num);
	}
}

__global__ void computeMetricMedian_kernel(float *DRRGradientMap,
	float *FixedGradientMap,
	int DRRsize0,
	int DRRsize1,
	float *DRRArray,
	float *FixedArray) {

	// DRR and fixed image indeces
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// DRR and fixed array index
	int idx = row + DRRsize0 * col;

	// printf("Thread index %i\n", DRRidx);
	// printf("row, col: %d, %d\n", row, col);

	if (DRRsize0 < idx && idx < DRRsize0 * (DRRsize1 - 1) && 
		idx % DRRsize0 != 0 && 
		idx % DRRsize0 != DRRsize0 - 1
		) { // checks if thread index is within the length of the DRR array

		// printf("index:%d\n", idx);

		float drrPixel = DRRArray[idx];
		float drrPixel_left = DRRArray[idx - 1];
		float drrPixel_right = DRRArray[idx + 1];
		float drrPixel_top = DRRArray[idx - DRRsize0];
		float drrPixel_bottom = DRRArray[idx + DRRsize0];
		float drrPixel_left_top = DRRArray[idx - DRRsize0 - 1];
		float drrPixel_right_top = DRRArray[idx - DRRsize0 + 1];
		float drrPixel_left_bottom = DRRArray[idx + DRRsize0 - 1];
		float drrPixel_right_bottom = DRRArray[idx + DRRsize0 + 1];
		float fixedPixel = FixedArray[idx];
		float fixedPixel_left = FixedArray[idx - 1];
		float fixedPixel_right = FixedArray[idx + 1];
		float fixedPixel_top = FixedArray[idx - DRRsize0];
		float fixedPixel_bottom = FixedArray[idx + DRRsize0];
		float fixedPixel_left_top = FixedArray[idx - DRRsize0 - 1];
		float fixedPixel_right_top = FixedArray[idx - DRRsize0 + 1];
		float fixedPixel_left_bottom = FixedArray[idx + DRRsize0 - 1];
		float fixedPixel_right_bottom = FixedArray[idx + DRRsize0 + 1];

		// if(idx == 599) {
		// 	printf("drrPixel:%f\n", drrPixel);
		// 	printf("drrPixel:%f\n", drrPixel_left);
		// 	printf("drrPixel:%f\n", drrPixel_right);
		// 	printf("drrPixel:%f\n", drrPixel_top);
		// 	printf("drrPixel:%f\n", drrPixel_bottom);
		// 	printf("fixedPixel:%f\n", fixedPixel);
		// 	printf("fixedPixel:%f\n", fixedPixel_left);
		// 	printf("fixedPixel:%f\n", fixedPixel_right);
		// 	printf("fixedPixel:%f\n", fixedPixel_top);
		// 	printf("fixedPixel:%f\n", fixedPixel_bottom);
		// }

		float m_dx = (drrPixel_right_top + 2 * drrPixel_right + drrPixel_right_bottom) - (drrPixel_left_top + 2 * drrPixel_left + drrPixel_left_bottom);
		float m_dy = (drrPixel_left_bottom + 2 * drrPixel_bottom + drrPixel_right_bottom) - (drrPixel_left_top + 2 * drrPixel_top + drrPixel_right_top);
		float f_dx = (fixedPixel_right_top + 2 * fixedPixel_right + fixedPixel_right_bottom) - (fixedPixel_left_top + 2 * fixedPixel_left + fixedPixel_left_bottom);
		float f_dy = (fixedPixel_left_bottom + 2 * fixedPixel_bottom + fixedPixel_right_bottom) - (fixedPixel_left_top + 2 * fixedPixel_top + fixedPixel_right_top);

		float grad_m_mod = sqrtf(m_dx * m_dx + m_dy * m_dy);
		float grad_f_mod = sqrtf(f_dx * f_dx + f_dy * f_dy);

		// printf("grad_m_mod = %f, grad_f_mod = %f\n", grad_m_mod, grad_f_mod);
		
		int col_new = int(idx / DRRsize0) - 1;
		int row_new = int(idx % DRRsize0) - 1;

		int idx_new = row_new + (DRRsize0 - 1) * col_new;

		DRRGradientMap[idx_new] = grad_m_mod;
		FixedGradientMap[idx_new] = grad_f_mod;

	}

}

__global__ void backwardProp_kernel(float *movImgArray,
	float *DRRarray,
	float *source,
	float *DestArray,
	int DRRsize0,
	int DRRsize1,
	int *MovSize,
	float *MovSpacing,
	float X0, float Y0, float Z0) {

	// DRR image indeces
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	// DRR array index
	int DRRidx = row + DRRsize0 * col;

	// moving image total size
	int totalMovSize = MovSize[0] * MovSize[1] * MovSize[2];

	if (DRRidx < DRRsize0 * DRRsize1) { // checks if thread index is within the length of the DRR array

		// --- declaration of variables for Siddon ---
		float alpha_min, alpha_max;
		float alpha_x_min, alpha_x_max, alpha_y_min, alpha_y_max, alpha_z_min, alpha_z_max;
		int i_min, i_max, j_min, j_max, k_min, k_max; // indeces corresponding to first and last intersected voxels
		float alpha_current;
		float alpha_x_next;
		float alpha_y_next;
		float alpha_z_next;
		float distance;
		int arrayIdx;
		int arrayIdx_old;
		unsigned int i_v, j_v, k_v;
		float alpha_first_pixel;
		float density_value = 0.;

		// --- define destination point based on DRR array index --- 
		float dest[3];
		get_dest(DRRidx, DestArray, dest);

		// --- source-to-destination distance --- 
		distance = sqrtf((dest[0] - source[0])*(dest[0] - source[0]) +
			(dest[1] - source[1])*(dest[1] - source[1]) +
			(dest[2] - source[2])*(dest[2] - source[2]));

		float dx = MovSpacing[0] / fabsf(dest[0] - source[0]);
		float dy = MovSpacing[1] / fabsf(dest[1] - source[1]);
		float dz = MovSpacing[2] / fabsf(dest[2] - source[2]);

		// --- find alpha_min and alpha_max
		// initialize alpha_min and alpha_max
		alpha_min = 0.;
		alpha_max = 1.;

		// X
		if (fabsf(dest[0] - source[0]) > epsilon) {

			float alpha_x0 = (X0 - source[0]) / (dest[0] - source[0]);
			float alpha_xN;
			compute_alpha_x(X0, MovSpacing[0], MovSize[0], source[0], dest[0], alpha_xN);
			alpha_x_min = fminf(alpha_x0, alpha_xN);
			alpha_x_max = fmaxf(alpha_x0, alpha_xN);
			if (alpha_x_min > alpha_min) { alpha_min = alpha_x_min; };
			if (alpha_x_max < alpha_max) { alpha_max = alpha_x_max; };
			//printf("alpha_min: %f\n", alpha_min);
			//printf("alpha_max: %f\n", alpha_max);

		}

		// Y
		if (fabsf(dest[1] - source[1]) > epsilon) {

			float alpha_y0 = (Y0 - source[1]) / (dest[1] - source[1]);
			float alpha_yN;
			compute_alpha_y(Y0, MovSpacing[1], MovSize[1], source[1], dest[1], alpha_yN);
			alpha_y_min = fminf(alpha_y0, alpha_yN);
			alpha_y_max = fmaxf(alpha_y0, alpha_yN);
			if (alpha_y_min > alpha_min) { alpha_min = alpha_y_min; };
			if (alpha_y_max < alpha_max) { alpha_max = alpha_y_max; };

		}

		// Z
		if (fabsf(dest[2] - source[2]) > epsilon) {

			float alpha_z0 = (Z0 - source[2]) / (dest[2] - source[2]);
			float alpha_zN;
			compute_alpha_z(Z0, MovSpacing[2], MovSize[2], source[2], dest[2], alpha_zN);
			alpha_z_min = fminf(alpha_z0, alpha_zN);
			alpha_z_max = fmaxf(alpha_z0, alpha_zN);
			if (alpha_z_min > alpha_min) { alpha_min = alpha_z_min; };
			if (alpha_z_max < alpha_max) { alpha_max = alpha_z_max; };

		}

		// --- initialize alpha --- 
		alpha_current = alpha_min;

		if (alpha_min < alpha_max) {

			// compute i_min, i_max and initialize alpha_x_next 
			if (dest[0] - source[0] > 0.) {

				// i_min
				if (fabsf(alpha_min - alpha_x_min) < epsilon) { i_min = 1; } // in other words: if source point is out of 3D model(x plane parallel region)
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_min, source[0], dest[0], phi_x);
					i_min = ceil(phi_x);
				}

				// i_max
				if (fabsf(alpha_max - alpha_x_max) < epsilon) { i_max = MovSize[0] - 1; } // in other words: if destination point is out of 3D model(x plane parallel region)
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_max, source[0], dest[0], phi_x);
					i_max = floor(phi_x);
				}

				// initialize alpha_x_next
				compute_alpha_x(X0, MovSpacing[0], i_min, source[0], dest[0], alpha_x_next);
			}
			else {

				// i_max
				if (fabsf(alpha_min - alpha_x_min) < epsilon) { i_max = MovSize[0] - 1; }
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_min, source[0], dest[0], phi_x);
					i_max = floor(phi_x);
				}

				// i_min
				if (fabsf(alpha_max - alpha_x_max) < epsilon) { i_min = 1; }
				else {
					float phi_x;
					compute_phi_x(X0, MovSpacing[0], alpha_max, source[0], dest[0], phi_x);
					i_min = ceil(phi_x);
				}

				// initialize alpha_x_next
				compute_alpha_x(X0, MovSpacing[0], i_max, source[0], dest[0], alpha_x_next);
			}

			// compute j_min, j_max and initialize alpha_y_next 
			if (dest[1] - source[1] > 0.) {

				// j_min
				if (fabsf(alpha_min - alpha_y_min) < epsilon) { j_min = 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_min, source[1], dest[1], phi_y);
					//printf("phi_y: %f\n", phi_y);
					j_min = ceil(phi_y);
					//printf("j_min: %f\n", j_min);
				}

				// j_max
				if (fabsf(alpha_max - alpha_y_max) < epsilon) { j_max = MovSize[1] - 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_max, source[1], dest[1], phi_y);
					j_max = floor(phi_y);
				}

				// initialize alpha_y_next
				compute_alpha_y(Y0, MovSpacing[1], j_min, source[1], dest[1], alpha_y_next);
			}
			else {

				// j_max
				if (fabsf(alpha_min - alpha_y_min) < epsilon) { j_max = MovSize[1] - 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_min, source[1], dest[1], phi_y);
					j_max = floor(phi_y);
				}

				// j_min
				if (fabsf(alpha_max - alpha_y_max) < epsilon) { j_min = 1; }
				else {
					float phi_y;
					compute_phi_y(Y0, MovSpacing[1], alpha_max, source[1], dest[1], phi_y);
					j_min = ceil(phi_y);
				}

				// initialize alpha_y_next
				compute_alpha_y(Y0, MovSpacing[1], j_max, source[1], dest[1], alpha_y_next);
			}

			// compute k_min, k_max and initialize alpha_z_next 
			if (dest[2] - source[2] > 0.) {

				// k_min
				if (fabsf(alpha_min - alpha_z_min) < epsilon) { k_min = 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_min, source[2], dest[2], phi_z);
					k_min = ceil(phi_z);
				}

				// k_max
				if (fabsf(alpha_max - alpha_z_max) < epsilon) { k_max = MovSize[2] - 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_max, source[2], dest[2], phi_z);
					k_max = floor(phi_z);
				}

				// initialize alpha_z_next
				compute_alpha_z(Z0, MovSpacing[2], k_min, source[2], dest[2], alpha_z_next);
			}
			else {

				// k_max
				if (fabsf(alpha_min - alpha_z_min) < epsilon) { k_max = MovSize[2] - 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_min, source[2], dest[2], phi_z);
					k_max = floor(phi_z);
				}

				// k_min
				if (fabsf(alpha_max - alpha_z_max) < epsilon) { k_min = 1; }
				else {
					float phi_z;
					compute_phi_z(Z0, MovSpacing[2], alpha_max, source[2], dest[2], phi_z);
					k_min = ceil(phi_z);
				}

				// initialize alpha_z_next
				compute_alpha_z(Z0, MovSpacing[2], k_max, source[2], dest[2], alpha_z_next);
			}

			// --- initialize first intersected pixel i_v, j_v, k_v --- 
			if ((alpha_y_next < alpha_x_next) && (alpha_y_next < alpha_z_next)) {

				alpha_first_pixel = (alpha_y_next + alpha_min) / 2.;
			}
			else if (alpha_x_next < alpha_z_next) {

				alpha_first_pixel = (alpha_x_next + alpha_min) / 2.;
			}
			else {

				alpha_first_pixel = (alpha_z_next + alpha_min) / 2.;
			}


			float phi_x = 0.;
			float phi_y = 0.;
			float phi_z = 0.;
			compute_phi_x(X0, MovSpacing[0], alpha_first_pixel, source[0], dest[0], phi_x);
			i_v = floor(phi_x);
			compute_phi_y(Y0, MovSpacing[1], alpha_first_pixel, source[1], dest[1], phi_y);
			j_v = floor(phi_y);
			compute_phi_z(Z0, MovSpacing[2], alpha_first_pixel, source[2], dest[2], phi_z);
			k_v = floor(phi_z);

			// initialize array index of first intersected pixel
			arrayIdx = i_v + MovSize[0] * (j_v + MovSize[1] * k_v);
			arrayIdx_old = i_v + MovSize[0] * (j_v + MovSize[1] * k_v);

			// iterator indeces
			int stop = (i_max - i_min + 1) + (j_max - j_min + 1) + (k_max - k_min + 1);
			int iter = 0;

			//while (alpha_current < 1. && alpha_current < alpha_max) {
			while (iter < stop) {

				float l;

				// next intersection plane is y
				if ((alpha_y_next < alpha_x_next) && (alpha_y_next < alpha_z_next)) {

					l = (alpha_y_next - alpha_current);

					alpha_current = alpha_y_next;

					// update
					alpha_y_next += dy;
					j_v += (dest[1] - source[1] > 0.) ? 1 : -1;

				}

				else if (alpha_x_next < alpha_z_next) {

					// next intersection plane is x
					l = (alpha_x_next - alpha_current);

					alpha_current = alpha_x_next;

					// update
					alpha_x_next += dx;
					i_v += (dest[0] - source[0] > 0.) ? 1 : -1;


				}

				else {

					// next intersection plane is z
					l = (alpha_z_next - alpha_current);

					alpha_current = alpha_z_next;

					// update
					alpha_z_next += dz;
					k_v += (dest[2] - source[2] > 0.) ? 1 : -1;
				}

				// update array index
				update_idx(i_v, j_v, k_v, MovSize[0], MovSize[1], arrayIdx);

				if (0 < arrayIdx_old && arrayIdx_old < totalMovSize){
					
					// update density value
					// if(arrayIdx_old == 1000000){
					// 	printf("movImgArray[%d] = %f\n", arrayIdx_old, movImgArray[arrayIdx_old]);
					// 	printf("add = %f\n", DRRarray[DRRidx] * l * distance);
					// 	printf("DRRidx = %d\n", DRRidx);
					// 	printf("row, col: %d, %d\n", row, col);
					// 	atomicAdd(&movImgArray[arrayIdx_old], DRRarray[DRRidx] * l * distance);
					// 	printf("movImgArray[%d] = %f\n", arrayIdx_old, movImgArray[arrayIdx_old]);
					// }
					
					atomicAdd(&movImgArray[arrayIdx_old], DRRarray[DRRidx] * l * distance);
					

				}

				// update arrayIdx
				arrayIdx_old = arrayIdx;

				// update iter
				iter += 1;

			}

		}

	}

}

/**
*
* Deafult constructor
*
**/
SiddonGpu::SiddonGpu() { }

/**
*
* Overloaded constructor loads the CT scan (together with size and spacing) onto GPU memory
*
**/
SiddonGpu::SiddonGpu(int *NumThreadsPerBlock,
	float *movImgArray,
	int *MovSize,
	float *MovSpacing,
	float X0, float Y0, float Z0,
	int *DRRSize) {

	// ---- Allocate variable members ---- 
	m_NumThreadsPerBlock[0] = NumThreadsPerBlock[0];
	m_NumThreadsPerBlock[1] = NumThreadsPerBlock[1];
	m_NumThreadsPerBlock[2] = NumThreadsPerBlock[2];

	//m_MovSize[0] = MovSize[0];
	//m_MovSize[1] = MovSize[1];
	//m_MovSize[2] = MovSize[2];

	m_X0 = X0;
	m_Y0 = Y0;
	m_Z0 = Z0;

	m_DRRsize[0] = DRRSize[0];
	m_DRRsize[1] = DRRSize[1];
	m_DRRsize[2] = DRRSize[2];

	m_DRRsize0 = DRRSize[0];
	m_DRRsize1 = DRRSize[1];

	m_movImgMemSize = MovSize[0] * MovSize[1] * MovSize[2] * sizeof(float);
	m_DestMemSize = (DRRSize[0] * DRRSize[1] * DRRSize[2] * 3) * sizeof(float);
	m_DrrMemSize = (DRRSize[0] * DRRSize[1] * DRRSize[2]) * sizeof(float); // memory for each output drr

	// allocate space for device copies
	cudaMalloc((void**)&m_d_movImgArray, m_movImgMemSize);
	cudaMalloc((void**)&m_d_MovSize, 3 * sizeof(int));
	cudaMalloc((void**)&m_d_MovSpacing, 3 * sizeof(float));

	// Copy arrays related to the moving image onto device array
	cudaMemcpy(m_d_movImgArray, movImgArray, m_movImgMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(m_d_MovSize, MovSize, 3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(m_d_MovSpacing, MovSpacing, 3 * sizeof(float), cudaMemcpyHostToDevice);

	//std::cout << "Siddon object Initialization: GPU memory prepared \n" << std::endl;
	//printf("ctor %p\n", this); // in constructors

}

/**
*
* Destructor clears everything left from the GPU memory
*
**/
SiddonGpu::~SiddonGpu() {

	cudaFree(m_d_movImgArray);
	cudaFree(m_d_MovSize);
	cudaFree(m_d_MovSpacing);
	//cudaFree(d_drr_array);
	//std::cout << "Siddon object destruction: GPU memory cleared \n" << std::endl;
	//printf("dtor %p\n", this); // in destructor

}

/**
*-The function generate DRR must be called with the following variables :
*
* @param source : array of(transformed) source physical coordinates
* @param DestArray : C - ordered 1D array of physical coordinates relative to the(transformed) output DRR image.
* @param drrArray : output, 1D array for output values of projected CT densities
*
**/
void SiddonGpu::generateDRR(float *source,
							float *DestArray,
							float *drrArray) {

	cudaError_t ierrAsync;
	cudaError_t ierrSync;

	// declare pointer to device memory for output DRR array
	float *d_DestArray;
	float *d_source;
	// float *d_drr_array;

	// allocate space on device
	cudaMalloc((void**)&d_drr_array, m_DrrMemSize);
	cudaMalloc((void**)&d_source, 3 * sizeof(float));
	cudaMalloc((void**)&d_DestArray, m_DestMemSize);

	// Copy source and destination to device
	cudaMemcpy(d_DestArray, DestArray, m_DestMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_source, source, 3 * sizeof(float), cudaMemcpyHostToDevice);

	// printf("cudaMalloc1OK!\n");

	//std::cout << "DRR generation: GPU memory prepared \n" << std::endl;

	// determine number of required blocks
	dim3 threads_per_block(m_NumThreadsPerBlock[0], m_NumThreadsPerBlock[1], 1);
	dim3 number_of_blocks((m_DRRsize[0] / threads_per_block.x) + 1, (m_DRRsize[1] / threads_per_block.y) + 1, 1);

	//// Query GPU device
	//cudaDeviceProp prop;
	//cudaGetDeviceProperties(&prop, 0);
	//std::cout << "Max threads per block " << prop.maxThreadsPerBlock << std::endl;
	//cudaGetDeviceProperties(&prop, 0);
	//if (threads_per_block.x * threads_per_block.y * threads_per_block.z > prop.maxThreadsPerBlock) {
	//	printf("Too many threads per block ... exiting\n");
	//	goto cleanup;
	//}
	//if (threads_per_block.x > prop.maxThreadsDim[0]) {
	//	printf("Too many threads in x-direction ... exiting\n");
	//	goto cleanup;
	//}
	//if (threads_per_block.y > prop.maxThreadsDim[1]) {
	//	printf("Too many threads in y-direction ... exiting\n");
	//	goto cleanup;
	//}
	//if (threads_per_block.z > prop.maxThreadsDim[2]) {
	//	printf("Too many threads in z-direction ... exiting\n");
	//	goto cleanup;
	//}

	// launch kernel
	generateDRR_kernel << <number_of_blocks, threads_per_block >> >(d_drr_array,
															 d_source,
															 d_DestArray,
															 m_DRRsize0,
															 m_DRRsize1,
															 m_d_movImgArray,
															 m_d_MovSize,
															 m_d_MovSpacing,
															 m_X0, m_Y0, m_Z0);


	// Check for errors in Kernel launch
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { 
		printf("Cuda Sync error: %s\n", cudaGetErrorString(ierrSync));
		//goto cleanup; 
	}
	if (ierrAsync != cudaSuccess) { 
		printf("Cuda Async error: %s\n", cudaGetErrorString(ierrAsync)); 
		//goto cleanup;
	}

	// printf("kernel1finished!\n");

	// Copy result to host array
	cudaMemcpy(drrArray, d_drr_array, m_DrrMemSize, cudaMemcpyDeviceToHost);

	// printf("drrArray[599]:%f\n", drrArray[599]);

	// Clean up device DRR array
cleanup:
	cudaFree(d_source);
	cudaFree(d_DestArray);
	//std::cout << "DRR generation: GPU memory cleared \n" << std::endl;

	return;

}

void SiddonGpu::computeMetric(float *fixedArray,
	float drrThreshold,
	float fixedThreshold,
	int lowNum,
	float *weightSum,
	int *weightNum,
	float *metricValue) {
	
	// printf("start initialising!\n");

	cudaError_t ierrAsync;
	cudaError_t ierrSync;

	// declare pointer to device memory for metric
	float *d_fixed_array;
	float *d_weight_sum;
	int *d_weight_num;
	float d_drr_threshold = drrThreshold;
	float d_fixed_threshold = fixedThreshold;

	// printf("initialised!\n");

	// allocate space on device
	// cudaMalloc((void**)&d_metric_value, sizeof(float));
	cudaMalloc((void**)&d_fixed_array, m_DrrMemSize);
	cudaMalloc((void**)&d_weight_sum, sizeof(float));
	cudaMalloc((void**)&d_weight_num, sizeof(int));

	// printf("cudaMalloc2OK!\n");

	// Copy source and destination to device
	cudaMemcpy(d_fixed_array, fixedArray, m_DrrMemSize, cudaMemcpyHostToDevice);
	cudaMemset(d_weight_sum, 0, sizeof(float));
	cudaMemset(d_weight_num, 0, sizeof(int));

	// cudaMemcpy(d_test_drr, d_drr_array, m_DrrMemSize, cudaMemcpyDeviceToHost);

	// printf("cudacopy2OK!\n");
	// printf("fixedArray[599]:%f\n", fixedArray[599]);

	// determine number of required blocks
	dim3 threads_per_block(m_NumThreadsPerBlock[0], m_NumThreadsPerBlock[1], 1);
	dim3 number_of_blocks((m_DRRsize[0] / threads_per_block.x) + 1, (m_DRRsize[1] / threads_per_block.y) + 1, 1);

	// launch kernel
	computeMetric_kernel << <number_of_blocks, threads_per_block >> >(d_weight_sum, 
																	d_weight_num, 
																	m_DRRsize0,
																	m_DRRsize1,
																	d_drr_array, 
																	d_fixed_threshold,
																	d_drr_threshold,
																	d_fixed_array);

	// Check for errors in Kernel launch
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { 
		printf("Cuda Sync error2: %s\n", cudaGetErrorString(ierrSync));
		//goto cleanup; 
	}
	if (ierrAsync != cudaSuccess) { 
		printf("Cuda Async error2: %s\n", cudaGetErrorString(ierrAsync)); 
		//goto cleanup;
	}

	// Copy result to host array
	cudaMemcpy(weightSum, d_weight_sum, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(weightNum, d_weight_num, sizeof(int), cudaMemcpyDeviceToHost);

	// Compute metric value
	if(*weightNum > lowNum) {
		*metricValue = (1 - *weightSum / *weightNum);
	}
	else {
		*metricValue = (1 - *weightSum / lowNum);
	}
	printf("weightnum:%d\n", *weightNum);

	// Clean up device DRR array
cleanup:
	cudaFree(d_weight_sum);
	cudaFree(d_weight_num);
	cudaFree(d_fixed_array);
	cudaFree(d_drr_array);

	return;
}

void SiddonGpu::backwardProp(float *source,
								float *DestArray,
								float *inputGradArray, 
								float *outputGradArray){
	cudaError_t ierrAsync;
	cudaError_t ierrSync;

	// declare pointer to device memory for output DRR array
	float *d_DestArray;
	float *d_source;
	float *d_inputGradArray;

	// allocate space on device
	cudaMalloc((void**)&d_inputGradArray, m_DrrMemSize);
	cudaMalloc((void**)&d_source, 3 * sizeof(float));
	cudaMalloc((void**)&d_DestArray, m_DestMemSize);

	// Copy source and destination to device
	cudaMemcpy(d_inputGradArray, inputGradArray, m_DrrMemSize, cudaMemcpyHostToDevice); // Copy drr gradient array to device
	cudaMemcpy(d_DestArray, DestArray, m_DestMemSize, cudaMemcpyHostToDevice); // Copy drr destination array to device
	cudaMemcpy(d_source, source, 3 * sizeof(float), cudaMemcpyHostToDevice); // Copy source point array to device

	// printf("cudaMalloc1OK!\n");

	// determine number of required blocks
	dim3 threads_per_block(m_NumThreadsPerBlock[0], m_NumThreadsPerBlock[1], 1);
	dim3 number_of_blocks((m_DRRsize[0] / threads_per_block.x) + 1, (m_DRRsize[1] / threads_per_block.y) + 1, 1);

	// launch kernel
	backwardProp_kernel << <number_of_blocks, threads_per_block >> >(m_d_movImgArray,
															 d_inputGradArray,
															 d_source,
															 d_DestArray,
															 m_DRRsize0,
															 m_DRRsize1,
															 m_d_MovSize,
															 m_d_MovSpacing,
															 m_X0, m_Y0, m_Z0);


	// Check for errors in Kernel launch
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { 
		printf("Cuda Sync error3: %s\n", cudaGetErrorString(ierrSync));
		//goto cleanup; 
	}
	if (ierrAsync != cudaSuccess) { 
		printf("Cuda Async error3: %s\n", cudaGetErrorString(ierrAsync)); 
		//goto cleanup;
	}

	// printf("kernel1finished!\n");

	// Copy result to host array
	cudaMemcpy(outputGradArray, m_d_movImgArray, m_movImgMemSize, cudaMemcpyDeviceToHost);

	// printf("drrArray[599]:%f\n", drrArray[599]);

	// Clean up device DRR array
cleanup:
	cudaFree(d_source);
	cudaFree(d_DestArray);
	cudaFree(d_inputGradArray);
	//std::cout << "DRR generation: GPU memory cleared \n" << std::endl;

	return;
}

void SiddonGpu::computeMetricMedian(float *fixedArray,
	float *drrGradientMap,
	float *fixedGradientMap) {
	
	// printf("start initialising!\n");

	cudaError_t ierrAsync;
	cudaError_t ierrSync;

	// declare pointer to device memory for metric
	float *d_fixed_array;
	float *d_drrGradientMap;
	float *d_fixedGradientMap;

	// printf("initialised!\n");

	// allocate space on device
	// cudaMalloc((void**)&d_metric_value, sizeof(float));
	cudaMalloc((void**)&d_fixed_array, m_DrrMemSize);
	cudaMalloc((void**)&d_drrGradientMap, ((m_DRRsize[0] - 1) * (m_DRRsize[1] - 1) * m_DRRsize[2]) * sizeof(float));
	cudaMalloc((void**)&d_fixedGradientMap, ((m_DRRsize[0] - 1) * (m_DRRsize[1] - 1) * m_DRRsize[2]) * sizeof(float));

	// printf("cudaMalloc2OK!\n");

	// Copy source and destination to device
	cudaMemcpy(d_fixed_array, fixedArray, m_DrrMemSize, cudaMemcpyHostToDevice);

	// cudaMemcpy(d_test_drr, d_drr_array, m_DrrMemSize, cudaMemcpyDeviceToHost);

	// printf("cudacopy2OK!\n");
	// printf("fixedArray[599]:%f\n", fixedArray[599]);

	// determine number of required blocks
	dim3 threads_per_block(m_NumThreadsPerBlock[0], m_NumThreadsPerBlock[1], 1);
	dim3 number_of_blocks((m_DRRsize[0] / threads_per_block.x) + 1, (m_DRRsize[1] / threads_per_block.y) + 1, 1);

	// launch kernel
	computeMetricMedian_kernel << <number_of_blocks, threads_per_block >> >(d_drrGradientMap,
																	d_fixedGradientMap,
																	m_DRRsize0,
																	m_DRRsize1,
																	d_drr_array,
																	d_fixed_array);

	// Check for errors in Kernel launch
	ierrSync = cudaGetLastError();
	ierrAsync = cudaDeviceSynchronize(); // Wait for the GPU to finish
	if (ierrSync != cudaSuccess) { 
		printf("Cuda Sync error4: %s\n", cudaGetErrorString(ierrSync));
		//goto cleanup; 
	}
	if (ierrAsync != cudaSuccess) { 
		printf("Cuda Async error4: %s\n", cudaGetErrorString(ierrAsync)); 
		//goto cleanup;
	}

	// Copy result to host array
	cudaMemcpy(drrGradientMap, d_drrGradientMap, ((m_DRRsize[0] - 1) * (m_DRRsize[1] - 1) * m_DRRsize[2]) * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(fixedGradientMap, d_fixedGradientMap, ((m_DRRsize[0] - 1) * (m_DRRsize[1] - 1) * m_DRRsize[2]) * sizeof(float), cudaMemcpyDeviceToHost);

	// Clean up device DRR array
cleanup:
	cudaFree(d_fixed_array);
	cudaFree(d_drrGradientMap);
	cudaFree(d_fixedGradientMap);

	return;
}
