#ifndef PIPE_CUH
#define PIPE_CUH
#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <stdio.h>
#include <ctime>

#define CNT 4
using namespace std;
#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
int imgsize = 1920 * 1080;
int wsize = 64 * 5 * 5 + 32 * 64 * 3 * 3 + 4 * 32 * 3 * 3 + 64 + 32 + 4;
__constant__ float ww[5];
__global__ void inity(unsigned char * input, float * output)
{
	int wp = blockIdx.x * blockDim.x + threadIdx.x;
	int hp = blockIdx.y * blockDim.y + threadIdx.y;

	int pti = 1920 * hp + wp;
	output[pti] = ((float)input[pti]) / 255;
}
__global__ void inituv(unsigned char * input, float * outp) {
	int wp = blockIdx.x * blockDim.x + threadIdx.x;
	int hp = blockIdx.y * blockDim.y + threadIdx.y;

	int pti = (1920 / 2) * hp + wp;
	int pto = (1920 / 2 + 2 * 8) * (hp + 2) + wp + 8;
	outp[pto] = ((float)(unsigned int)input[pti]);
	outp[pto + 976 * 544] = ((float)(unsigned int)input[pti + 960 * 540]);
}

__global__ void bicubic(float * input, unsigned char * output)
{
	//float ww[5] = { 1.0, 0.5625, 0, -0.0625,0 };//parameters for bicubic interpolation
	__shared__ float data[8][48];//load data (4x8)
	__shared__ float res[8][32];//result data(8x16)
	int x = threadIdx.x;
	int y = threadIdx.y;
	int a, b, c, i, j, outp;
	//load data into shared memory
	for (i = 0; i < 4; i++)
	{
		a = (64 * i + y * blockDim.x + x) / 32;
		b = (64 * i + y * blockDim.x + x) % 32;
		c = (blockIdx.y * blockDim.y + a) * (1920 / 2 + 2 * 8) + (blockIdx.x * blockDim.x + b);
		data[a][b] = input[c];
	}
	__syncthreads();
	//init result data
	res[2 * y][2 * x] = data[y + 2][x + 8];
	res[2 * y + 1][2 * x] = 0;
	res[2 * y][2 * x + 1] = 0;
	res[2 * y + 1][2 * x + 1] = 0;
	int idx, idy;
	//bicubic
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
		{
			idy = 3 - 2 * i;
			idx = 2 - 2 * j;
			idy = idy > 0 ? idy : -idy;
			idx = idx > 0 ? idx : -idx;
			idy = idy > 4 ? 4 : idy;
			idx = idx > 4 ? 4 : idx;
			res[2 * y + 1][2 * x] += data[y + 1 + i][x + 7 + j] * ww[idy] * ww[idx];
		}
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
		{
			idy = 2 - 2 * i;
			idx = 3 - 2 * j;
			idy = idy > 0 ? idy : -idy;
			idx = idx > 0 ? idx : -idx;
			idy = idy > 4 ? 4 : idy;
			idx = idx > 4 ? 4 : idx;
			res[2 * y][2 * x + 1] += data[y + 1 + i][x + 7 + j] * ww[idy] * ww[idx];
		}
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
		{
			idy = 3 - 2 * i;
			idx = 3 - 2 * j;
			idy = idy > 0 ? idy : -idy;
			idx = idx > 0 ? idx : -idx;
			idy = idy > 4 ? 4 : idy;
			idx = idx > 4 ? 4 : idx;
			res[2 * y + 1][2 * x + 1] += data[y + 1 + i][x + 7 + j] * ww[idy] * ww[idx];
		}
	}

	__syncthreads();
	//write result to global memory
	for (i = 0; i < 4; i++)
	{
		a = blockDim.x * y + x + i * 64;
		b = a / 32;
		c = a % 32;
		outp = (2 * blockIdx.y * blockDim.y + b) * 1920 + 2 * blockIdx.x * blockDim.x + c;
		output[outp] = (unsigned char)(int)res[b][c];
		__syncthreads();
	}

	for (i = 0; i < 4; i++)
	{
		a = (64 * i + y * blockDim.x + x) / 32;
		b = (64 * i + y * blockDim.x + x) % 32;
		c = (blockIdx.y * blockDim.y + a) * (1920 / 2 + 2 * 8) + (blockIdx.x * blockDim.x + b);
		data[a][b] = input[c+976*544];
	}
	__syncthreads();
	//init result data
	res[2 * y][2 * x] = data[y + 2][x + 8];
	res[2 * y + 1][2 * x] = 0;
	res[2 * y][2 * x + 1] = 0;
	res[2 * y + 1][2 * x + 1] = 0;
	//bicubic
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
		{
			idy = 3 - 2 * i;
			idx = 2 - 2 * j;
			idy = idy > 0 ? idy : -idy;
			idx = idx > 0 ? idx : -idx;
			idy = idy > 4 ? 4 : idy;
			idx = idx > 4 ? 4 : idx;
			res[2 * y + 1][2 * x] += data[y + 1 + i][x + 7 + j] * ww[idy] * ww[idx];
		}
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
		{
			idy = 2 - 2 * i;
			idx = 3 - 2 * j;
			idy = idy > 0 ? idy : -idy;
			idx = idx > 0 ? idx : -idx;
			idy = idy > 4 ? 4 : idy;
			idx = idx > 4 ? 4 : idx;
			res[2 * y][2 * x + 1] += data[y + 1 + i][x + 7 + j] * ww[idy] * ww[idx];
		}
	}
	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++)
		{
			idy = 3 - 2 * i;
			idx = 3 - 2 * j;
			idy = idy > 0 ? idy : -idy;
			idx = idx > 0 ? idx : -idx;
			idy = idy > 4 ? 4 : idy;
			idx = idx > 4 ? 4 : idx;
			res[2 * y + 1][2 * x + 1] += data[y + 1 + i][x + 7 + j] * ww[idy] * ww[idx];
		}
	}

	__syncthreads();
	//write result to global memory
	for (i = 0; i < 4; i++)
	{
		a = blockDim.x * y + x + i * 64;
		b = a / 32;
		c = a % 32;
		outp = (2 * blockIdx.y * blockDim.y + b) * 1920 + 2 * blockIdx.x * blockDim.x + c;
		output[outp+1920*1080] = (unsigned char)(int)res[b][c];
		__syncthreads();
	}

}
__global__ void pix_shuffle(float * in, unsigned char * out)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	for (int i = 0; i < 2; i++)
	{
		for (int j = 0; j < 2; j++)
		{
			out[(2 * y + i) * 3840 + 2 * x + j] = (unsigned char)(in[1920 * 1080 * (2 * i + j) + y * 1920 + x] * 255);
		}
	}
	/*
	__shared__ float data[16][64];
	int x = threadIdx.x;
	int y = threadIdx.y;
	int xx = x / 2;
	int yy = y / 2;
	int ind = 2 * (y % 2) + x % 2;
	int p = (blockIdx.y * blockDim.y / 2 + yy) * 1920 + (blockIdx.x * blockDim.x / 2 + xx);
	data[y][x] = in[ind * 1920 * 1080 + p];
	__syncthreads();
	p = (blockIdx.y * blockDim.y + y) * 3840 + (blockIdx.x * blockDim.x + x);
	out[p] = (unsigned char)(unsigned int)(data[y][x] * 255);
	*/
}
dim3 Block1(16, 4);
dim3 Grid1(60, 135);
dim3 UV_Block(64, 4);
dim3 UV_Grid(15, 135);
dim3 Block(16, 8);
dim3 Grid(120, 135);
dim3 Blocksf(32, 8);
dim3 Gridsf(60, 135);
class my_sr
{
private:
	cudnnHandle_t cudnn;
	cudnnTensorDescriptor_t input_descriptor, output_descriptor, output_descriptor2, output_descriptor3;
	cudnnFilterDescriptor_t kernel_descriptor, kernel_descriptor2, kernel_descriptor3;
	cudnnTensorDescriptor_t bias_descriptor, bias_descriptor2, bias_descriptor3;
	cudnnActivationDescriptor_t   relu_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor;
	cudnnConvolutionDescriptor_t convolution_descriptor2;
	cudnnConvolutionDescriptor_t convolution_descriptor3;
	cudnnConvolutionFwdAlgo_t convolution_algorithm, convolution_algorithm2, convolution_algorithm3;
	size_t workspace_bytes = 0;
	size_t workspace_bytes2 = 0;
	size_t workspace_bytes3 = 0;
	const float alpha = 1, beta = 0;

public:
	cudaStream_t my_stream;
	float * d_w1;
	float * d_w2;
	float * d_w3;
	float * d_b1;
	float * d_b2;
	float * d_b3;
	unsigned char * in_YUV;
	unsigned char * out_YUV;
	unsigned char * d_in_YUV;
	float * d_in1;
	float * d_out1;
	float * d_out2;
	float * d_out3;
	float * d_tmpuv;
	unsigned char * d_out_YUV;
	void * d_workspace;
	void * d_workspace2;
	void * d_workspace3;

public:
	void init()
	{
		checkCUDNN(cudnnCreate(&cudnn));
		cudaStreamCreate(&my_stream);
		cudnnSetStream(cudnn, my_stream);
		checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
		checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
		checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor2));
		checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor3));

		checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
		checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor2));
		checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor3));

		checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
		checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor2));
		checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor3));

		checkCUDNN(cudnnCreateActivationDescriptor(&relu_descriptor));

		checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor2));
		checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor3));

		cudaMalloc((void**)&d_w1, sizeof(float) * 64 * 5 * 5);
		cudaMalloc((void**)&d_w2, sizeof(float) * 32 * 64 * 3 * 3);
		cudaMalloc((void**)&d_w3, sizeof(float) * 4 * 32 * 3 * 3);
		cudaMalloc((void**)&d_b1, sizeof(float) * 64);
		cudaMalloc((void**)&d_b2, sizeof(float) * 32);
		cudaMalloc((void**)&d_b3, sizeof(float) * 4);

		cudaMallocHost((void**)&in_YUV, imgsize * 3 / 2);
		cudaMallocHost((void**)&out_YUV, imgsize * 6);
		cudaMalloc((void**)&d_in_YUV, imgsize * 3 / 2);
		cudaMalloc((void**)&d_tmpuv, sizeof(float) * 976 * 544 * 2);
		cudaMemset(d_tmpuv, 0, sizeof(float) * 976 * 544 * 2);
		cudaMalloc((void**)&d_in1, sizeof(float) * imgsize);
		cudaMalloc((void**)&d_out1, sizeof(float) * imgsize * 64);
		cudaMalloc((void**)&d_out2, sizeof(float) * imgsize * 32);
		cudaMalloc((void**)&d_out3, sizeof(float) * imgsize * 4);
		cudaMalloc((void**)&d_out_YUV, imgsize * 6);

		initw("/vapour/weight.sr");

		checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/1,
			/*image_height=*/1080,
			/*image_width=*/1920));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/64,
			/*image_height=*/1080,
			/*image_width=*/1920));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor2,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/32,
			/*image_height=*/1080,
			/*image_width=*/1920));
		checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor3,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/4,
			/*image_height=*/1080,
			/*image_width=*/1920));

		checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/64,
			/*image_height=*/1,
			/*image_width=*/1));
		checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor2,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/32,
			/*image_height=*/1,
			/*image_width=*/1));
		checkCUDNN(cudnnSetTensor4dDescriptor(bias_descriptor3,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*batch_size=*/1,
			/*channels=*/4,
			/*image_height=*/1,
			/*image_width=*/1));

		checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/64,
			/*in_channels=*/1,
			/*kernel_height=*/5,
			/*kernel_width=*/5));
		checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor2,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/32,
			/*in_channels=*/64,
			/*kernel_height=*/3,
			/*kernel_width=*/3));
		checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor3,
			/*dataType=*/CUDNN_DATA_FLOAT,
			/*format=*/CUDNN_TENSOR_NCHW,
			/*out_channels=*/4,
			/*in_channels=*/32,
			/*kernel_height=*/3,
			/*kernel_width=*/3));


		checkCUDNN(cudnnSetActivationDescriptor(relu_descriptor, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 1000.0));

		checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
			/*pad_height=*/2,
			/*pad_width=*/2,
			/*vertical_stride=*/1,
			/*horizontal_stride=*/1,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION,
			/*computeType=*/CUDNN_DATA_FLOAT));
		checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor2,
			/*pad_height=*/1,
			/*pad_width=*/1,
			/*vertical_stride=*/1,
			/*horizontal_stride=*/1,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION,
			/*computeType=*/CUDNN_DATA_FLOAT));
		checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor3,
			/*pad_height=*/1,
			/*pad_width=*/1,
			/*vertical_stride=*/1,
			/*horizontal_stride=*/1,
			/*dilation_height=*/1,
			/*dilation_width=*/1,
			/*mode=*/CUDNN_CROSS_CORRELATION,
			/*computeType=*/CUDNN_DATA_FLOAT));



		checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
				input_descriptor,
				kernel_descriptor,
				convolution_descriptor,
				output_descriptor,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				/*memoryLimitInBytes=*/0,
				&convolution_algorithm));
		checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
				output_descriptor,
				kernel_descriptor2,
				convolution_descriptor2,
				output_descriptor2,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				/*memoryLimitInBytes=*/0,
				&convolution_algorithm2));
		checkCUDNN(
			cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
				output_descriptor2,
				kernel_descriptor3,
				convolution_descriptor3,
				output_descriptor3,
				CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
				/*memoryLimitInBytes=*/0,
				&convolution_algorithm3));

		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			input_descriptor,
			kernel_descriptor,
			convolution_descriptor,
			output_descriptor,
			convolution_algorithm,
			&workspace_bytes));
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			output_descriptor,
			kernel_descriptor2,
			convolution_descriptor2,
			output_descriptor2,
			convolution_algorithm2,
			&workspace_bytes2));
		checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
			output_descriptor2,
			kernel_descriptor3,
			convolution_descriptor3,
			output_descriptor3,
			convolution_algorithm3,
			&workspace_bytes3));

		cudaMalloc((void **)&d_workspace, workspace_bytes);
		cudaMalloc((void **)&d_workspace2, workspace_bytes2);
		cudaMalloc((void **)&d_workspace3, workspace_bytes3);

	}
	void initw(char * wn)
	{
		ifstream fin(wn, ios::binary);
		unsigned char * tmp = new unsigned char[wsize * sizeof(float)];
		fin.read((char*)tmp, wsize * sizeof(float));
		fin.close();
		int p = 0;
		cudaMemcpyAsync(&d_w1[0], &tmp[p], sizeof(float) * 64 * 5 * 5, cudaMemcpyHostToDevice,my_stream);
		p += sizeof(float) * 64 * 5 * 5;
		cudaMemcpyAsync(&d_w2[0], &tmp[p], sizeof(float) * 32 * 64 * 3 * 3, cudaMemcpyHostToDevice, my_stream);
		p += sizeof(float) * 32 * 64 * 3 * 3;
		cudaMemcpyAsync(&d_w3[0], &tmp[p], sizeof(float) * 4 * 32 * 3 * 3, cudaMemcpyHostToDevice, my_stream);
		p += sizeof(float) * 4 * 32 * 3 * 3;
		cudaMemcpyAsync(&d_b1[0], &tmp[p], sizeof(float) * 64, cudaMemcpyHostToDevice, my_stream);
		p += sizeof(float) * 64;
		cudaMemcpyAsync(&d_b2[0], &tmp[p], sizeof(float) * 32, cudaMemcpyHostToDevice,my_stream);
		p += sizeof(float) * 32;
		cudaMemcpyAsync(&d_b3[0], &tmp[p], sizeof(float) * 4, cudaMemcpyHostToDevice,my_stream);
		p += sizeof(float) * 4;
		delete[] tmp;
		float ww2[5] = { 1.0, 0.5625, 0, -0.0625,0 };
		cudaMemcpyToSymbolAsync(ww, ww2, sizeof(ww2),0,cudaMemcpyHostToDevice,my_stream);
	}

	void super_resolution()
	{
		cudaMemcpyAsync(d_in_YUV, in_YUV, 1920 * 1080 * 3 / 2, cudaMemcpyHostToDevice, my_stream);
		inity << <Grid, Block, 0, my_stream >> > (d_in_YUV, d_in1);
		checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn,
			&alpha,
			input_descriptor,
			d_in1,
			kernel_descriptor,
			d_w1,
			convolution_descriptor,
			convolution_algorithm,
			d_workspace,
			workspace_bytes,
			&beta,
			output_descriptor,
			d_out1,
			bias_descriptor,
			d_b1,
			relu_descriptor,
			output_descriptor,
			d_out1));
		checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn,
			&alpha,
			output_descriptor,
			d_out1,
			kernel_descriptor2,
			d_w2,
			convolution_descriptor2,
			convolution_algorithm2,
			d_workspace2,
			workspace_bytes2,
			&beta,
			output_descriptor2,
			d_out2,
			bias_descriptor2,
			d_b2,
			relu_descriptor,
			output_descriptor2,
			d_out2));
		checkCUDNN(cudnnConvolutionBiasActivationForward(cudnn,
			&alpha,
			output_descriptor2,
			d_out2,
			kernel_descriptor3,
			d_w3,
			convolution_descriptor3,
			convolution_algorithm3,
			d_workspace3,
			workspace_bytes3,
			&beta,
			output_descriptor3,
			d_out3,
			bias_descriptor3,
			d_b3,
			relu_descriptor,
			output_descriptor3,
			d_out3));
		pix_shuffle << <Gridsf, Blocksf, 0, my_stream >> > (d_out3, d_out_YUV);
		inituv << <Grid1, Block1, 0, my_stream >> > (&d_in_YUV[1920 * 1080], d_tmpuv);
		bicubic << <Grid1, Block1, 0, my_stream >> > (d_tmpuv, &d_out_YUV[3840 * 2160]);
		//cudaMemcpyAsync(out_YUV, d_out_YUV, 3840 * 2160,cudaMemcpyDeviceToHost,my_stream);
		//cudaStreamSynchronize(my_stream);
	}
	void copy_out()
	{
		cudaMemcpyAsync(&out_YUV[0], &d_out_YUV[0], 3840 * 2160 * 3 / 2, cudaMemcpyDeviceToHost, my_stream);
	}
	void sync()
	{
		
		while (cudaStreamQuery(my_stream) != cudaSuccess)
		{
			usleep(1000);
		}
		
		//cudaStreamSynchronize(my_stream);
	}
	void free()
	{
		checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
		checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
		checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor2));
		checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor3));

		checkCUDNN(cudnnDestroyTensorDescriptor(bias_descriptor));
		checkCUDNN(cudnnDestroyTensorDescriptor(bias_descriptor2));
		checkCUDNN(cudnnDestroyTensorDescriptor(bias_descriptor3));

		checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
		checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor2));
		checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor3));

		checkCUDNN(cudnnDestroyActivationDescriptor(relu_descriptor));

		checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor2));
		checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor3));

		checkCUDNN(cudnnDestroy(cudnn));
		cudaStreamDestroy(my_stream);
		cudaFree(d_w1);
		cudaFree(d_w2);
		cudaFree(d_w3);
		cudaFree(d_b1);
		cudaFree(d_b2);
		cudaFree(d_b3);
		cudaFreeHost(in_YUV);
		cudaFreeHost(out_YUV);
		cudaFree(d_tmpuv);
		cudaFree(d_in_YUV);
		cudaFree(d_in1);
		cudaFree(d_out1);
		cudaFree(d_out2);
		cudaFree(d_out3);
		cudaFree(d_out_YUV);
		cudaFree(d_workspace);
		cudaFree(d_workspace2);
		cudaFree(d_workspace3);
	}
};
#endif
//void test()
//{
//	char * f1 = "out.yuv";
//	char * f2 = "ref.yuv";
//	ifstream fi1(f1, ios::binary);
//	ifstream fi2(f2, ios::binary);
//	char c1, c2;
//	int cnt(0);
//	int cnt2(0);
//	for (int i = 0; i < 3840 * 2160; i++)
//	{
//		fi1.read(&c1, 1);
//		fi2.read(&c2, 1);
//		if (c1 != c2)
//		{
//			cnt++;
//		}
//	}
//	for (int i = 0; i < 3840 * 2160 / 4; i++)
//	{
//		fi1.read(&c1, 1);
//		fi2.read(&c2, 1);
//		if (c1 != c2)
//		{
//			cnt2++;
//		}
//	}
//	fi1.close();
//	fi2.close();
//	cout << "Different: " << cnt <<' '<<cnt2 << endl;
//	//system("pause");
//}
//int main()
//{
//	my_sr SR;
//	//my_sr SR2;
//	//my_sr SR3;
//
//	SR.init();
//	//SR2.init(stream[1]);
//	//SR3.init(stream[2]);
//	const char * fn = "in.yuv";
//	ifstream fin(fn, ios::binary);
//	ofstream foo("out.yuv", ios::binary);
//
//	for (int fcnt = 0; fcnt < 1; fcnt++)
//	{
//		fin.read((char *)SR.in_YUV, 1920 * 1080 * 3 / 2);
//		SR.super_resolutionY();
//		SR.super_resolutionUV();
//		SR.sync();
//		foo.write((char *)SR.out_YUV, 3840 * 2160 * 3 / 2);
//	}
//	foo.close();
//	fin.close();
//	SR.free();
//	test();
//	return 0;
//}
