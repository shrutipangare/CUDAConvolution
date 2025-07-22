// GPU Convolution Implementation
// For Advanced GPU Computing Course Project

// Required headers
#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cuda_runtime.h>

// Constants and parameters
#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define BLOCK_SIZE 18
#define SHARED_I_SIZE (C * BLOCK_SIZE * BLOCK_SIZE)
#define SHARED_F_SIZE (FW * FH * C)
#define P 1


typedef struct
{
    int width;
    int height;
    int channel;
    int stride;
    double *elements;
} Matrix;

typedef struct
{
    int width;
    int height;
    int channel;
    int numKernels;
    double *elements;
} Filter;

// Implementation of a direct convolution without memory optimization
__global__ void basicConvolution(Matrix I, Filter F, Matrix O)
{
    // Identify thread positions
    int kernelIndex = blockIdx.x; // Current kernel to process
    int col = threadIdx.x;        // X position in output
    int row = blockIdx.y;         // Y position in output

    int inputGrid = I.width * I.height;                      // Size of input feature map
    int filterGrid = F.width * F.height;                     // Size of filter plane
    int kernelOffset = kernelIndex * F.channel * filterGrid; // Starting position for kernel
    int outputGrid = O.width * O.height;                     // Size of output feature map

    double sum = 0.0; // Result accumulator
    // Compute convolution across channels and spatial dimensions
    for (int c = 0; c < F.channel; c++)
    {
        for (int j = 0; j < F.height; j++)
        {
            for (int i = 0; i < F.width; i++)
            {
                sum += I.elements[c * inputGrid + (row + j) * I.width + (col + i)] *
                       F.elements[kernelOffset + c * filterGrid + (F.height - j - 1) * F.width + (F.width - i - 1)];
            }
        }
    }
    O.elements[kernelIndex * outputGrid + row * O.width + col] = sum;
}

// Optimized implementation using shared memory for data reuse
__global__ void tiledConvolutionSharedMemory(Matrix I, Filter F, Matrix O)
{
    int kernelIndex = blockIdx.z; // Which output feature map
    int thread_row = threadIdx.y; // Thread's row position in block
    int thread_col = threadIdx.x; // Thread's column position in block
    int block_row = blockIdx.y;   // Block's row position
    int block_col = blockIdx.x;   // Block's column position

    int inputGrid = I.width * I.height;      // Input feature map size
    int filterGrid = F.width * F.height;     // Filter spatial size
    int filterSize = filterGrid * F.channel; // Total filter elements for one kernel
    int isubGrid = BLOCK_SIZE * BLOCK_SIZE;  // Shared memory tile size
    int outputGrid = O.width * O.height;     // Output feature map size

    // Calculate input starting position for this block
    int y_0_in_I = block_row * (BLOCK_SIZE - 2 * P);
    int x_0_in_I = block_col * (BLOCK_SIZE - 2 * P);
    double *subTensorStart = &I.elements[y_0_in_I * I.width + x_0_in_I];

    // Load input data to shared memory
    __shared__ double sharedInput[SHARED_I_SIZE];
    for (int c = 0; c < I.channel; c++)
    {
        sharedInput[c * isubGrid + thread_row * BLOCK_SIZE + thread_col] = subTensorStart[c * inputGrid + thread_row * I.width + thread_col];
    }

    // Load filter weights to shared memory
    __shared__ double sharedFilters[SHARED_F_SIZE];
    int thread_idx_in_block = thread_row * BLOCK_SIZE + thread_col;
    if (thread_idx_in_block < filterSize)
    {
        sharedFilters[thread_idx_in_block] = F.elements[kernelIndex * filterSize + thread_idx_in_block];
    }

    __syncthreads();

    // Compute convolution from shared memory
    if (thread_row < BLOCK_SIZE - 2 * P && thread_col < BLOCK_SIZE - 2 * P)
    {
        double sum = 0.0;
        for (int c = 0; c < F.channel; c++)
        {
            for (int j = 0; j < F.height; j++)
            {
                for (int i = 0; i < F.width; i++)
                {
                    sum += sharedInput[c * isubGrid + (thread_row + j) * BLOCK_SIZE + (thread_col + i)] *
                           sharedFilters[c * filterGrid + (F.height - j - 1) * F.width + (F.width - i - 1)];
                }
            }
        }
        int row = block_row * (BLOCK_SIZE - 2 * P) + thread_row;
        int col = block_col * (BLOCK_SIZE - 2 * P) + thread_col;
        O.elements[kernelIndex * outputGrid + row * O.width + col] = sum;
    }
}

// Host matrix allocation and initialization function
Matrix hostMatrix(int width, int height, int channel)
{
    Matrix newHostMatrix;
    newHostMatrix.width = width;
    newHostMatrix.height = height;
    newHostMatrix.channel = channel;
    // Memory size calculation
    size_t size = newHostMatrix.width * newHostMatrix.height * newHostMatrix.channel * sizeof(double);
    // Allocate host memory
    newHostMatrix.elements = (double *)malloc(size);
    return newHostMatrix;
}

// Host filter allocation and initialization function
Filter hostFilter(int width, int height, int channel, int numKernels)
{
    Filter newHostFilter;
    newHostFilter.width = width;
    newHostFilter.height = height;
    newHostFilter.channel = channel;
    newHostFilter.numKernels = numKernels;
    // Memory size calculation for filter
    size_t size = newHostFilter.width * newHostFilter.height * newHostFilter.channel *
                  newHostFilter.numKernels * sizeof(double);
    newHostFilter.elements = (double *)malloc(size);
    return newHostFilter;
}

// Device memory allocation for matrix
Matrix deviceMatrix(Matrix M, bool copy)
{
    Matrix newDeviceMatrix;
    newDeviceMatrix.width = M.width;
    newDeviceMatrix.height = M.height;
    newDeviceMatrix.channel = M.channel;
    newDeviceMatrix.stride = M.width; // Row-major stride
    // GPU memory allocation
    cudaMalloc((void **)&newDeviceMatrix.elements, M.width * M.height * M.channel * sizeof(double));
    // Optional host-to-device copy
    if (copy)
    {
        cudaMemcpy(newDeviceMatrix.elements, M.elements, M.width * M.height * M.channel * sizeof(double), cudaMemcpyHostToDevice);
    }
    return newDeviceMatrix;
}

// Device memory allocation for filter
Filter deviceFilter(Filter F, bool copy)
{
    Filter newDeviceFilter;
    newDeviceFilter.width = F.width;
    newDeviceFilter.height = F.height;
    newDeviceFilter.channel = F.channel;
    newDeviceFilter.numKernels = F.numKernels;
    // GPU memory allocation
    cudaMalloc((void **)&newDeviceFilter.elements, F.width * F.height * F.channel * F.numKernels * sizeof(double));
    // Optional host-to-device copy
    if (copy)
    {
        cudaMemcpy(newDeviceFilter.elements, F.elements, F.width * F.height * F.channel * F.numKernels * sizeof(double), cudaMemcpyHostToDevice);
    }
    return newDeviceFilter;
}

// Function to add zero-padding to input matrix
Matrix padHostInput(Matrix I, int padding)
{
    Matrix I_0;
    I_0.width = I.width + 2 * padding;
    I_0.height = I.height + 2 * padding;
    I_0.channel = I.channel;
    // Padded matrix memory allocation
    size_t size = I_0.width * I_0.height * I_0.channel * sizeof(double);
    I_0.elements = (double *)malloc(size);
    // Fill padded matrix
    for (int c = 0; c < I_0.channel; c++)
    {
        for (int y = 0; y < I_0.height; y++)
        {
            for (int x = 0; x < I_0.width; x++)
            {
                // Set padding areas to zero
                if (y < padding || y >= I.height + padding || x < padding || x >= I.width + padding)
                {
                    I_0.elements[x + y * I_0.width + c * (I_0.width * I_0.height)] = 0;
                }
                else
                {
                    // Copy original data to non-padding areas
                    I_0.elements[x + y * I_0.width + c * (I_0.width * I_0.height)] =
                        I.elements[(x - padding) + (y - padding) * I.width + c * (I.width * I.height)];
                }
            }
        }
    }
    return I_0;
}

// Generate test data for input matrix
Matrix getHostInput(int width, int height, int channel)
{
    Matrix M = hostMatrix(width, height, channel);
    for (int c = 0; c < M.channel; c++)
    {
        for (int y = 0; y < M.height; y++)
        {
            for (int x = 0; x < M.width; x++)
            {
                // Data initialization pattern
                M.elements[x + y * M.width + c * (M.width * M.height)] = c * (x + y);
            }
        }
    }
    return M;
}

// Generate test data for filter weights
Filter getHostFilter(int width, int height, int channel, int numKernels)
{
    Filter F = hostFilter(width, height, channel, numKernels);
    for (int k = 0; k < F.numKernels; k++)
    {
        for (int c = 0; c < F.channel; c++)
        {
            for (int j = 0; j < F.height; j++)
            {
                for (int i = 0; i < F.width; i++)
                {
                    // Filter weights initialization pattern
                    F.elements[i + j * F.width + c * (F.width * F.height) + k * (F.width * F.height * F.channel)] =
                        (c + k) * (i + j);
                }
            }
        }
    }
    return F;
}

// Validation function to verify result correctness
double getCheckSum(Matrix M)
{
    double sum = 0.0;
    int grid = M.width * M.height; // Elements per channel
    // Accumulate all matrix elements
    for (int c = 0; c < M.channel; c++)
    {
        for (int y = 0; y < M.height; y++)
        {
            for (int x = 0; x < M.width; x++)
            {
                sum += M.elements[x + y * M.width + c * grid];
            }
        }
    }
    return sum;
}

// Implementation 1: Basic CUDA kernel approach
void convolutionC1(const Matrix input, const Filter filter, Matrix output, float &executionTime)
{
    Matrix device_input = deviceMatrix(input, true);
    Matrix device_output = deviceMatrix(output, false);
    Filter device_filter = deviceFilter(filter, true);

    dim3 dimBlock(output.width);
    dim3 dimGrid(output.channel, output.height);

    // Warm up GPU 
    basicConvolution<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);
    cudaThreadSynchronize();

    // Performance measurement setup
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    // Timed execution
    cudaEventRecord(start_time);
    basicConvolution<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);
    cudaThreadSynchronize();
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(&executionTime, start_time, end_time);

    // Copy results back to host
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

// Implementation 2: Shared memory tiled approach
void convolutionC2(const Matrix input, const Filter filter, Matrix output, float &executionTime)
{
    Matrix device_input = deviceMatrix(input, true);
    Filter device_filter = deviceFilter(filter, true);
    Matrix device_output = deviceMatrix(output, false);

    // Block configuration for tiled approach
    int block_per_row = (input.width - 2 * P) / (BLOCK_SIZE - 2 * P);
    int block_per_col = (input.height - 2 * P) / (BLOCK_SIZE - 2 * P);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(block_per_row, block_per_col, K);

    // Warm up GPU
    tiledConvolutionSharedMemory<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);
    cudaThreadSynchronize();

    // Performance measurement setup
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);

    // Timed execution
    cudaEventRecord(start_time);
    tiledConvolutionSharedMemory<<<dimGrid, dimBlock>>>(device_input, device_filter, device_output);
    cudaThreadSynchronize();
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);

    cudaEventElapsedTime(&executionTime, start_time, end_time);

    // Copy results back to host
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);

    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
}

// Implementation 3: cuDNN optimized approach
void convolutionC3(const Matrix input, const Filter filter, Matrix output, float &executionTime) {
    Matrix device_input = deviceMatrix(input, true);
    Filter device_filter = deviceFilter(filter, true);
    Matrix device_output = deviceMatrix(output, false);

    // cuDNN initialization
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Input tensor configuration
    cudnnTensorDescriptor_t inputDesc;
    cudnnCreateTensorDescriptor(&inputDesc);
    cudnnSetTensor4dDescriptor(inputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, input.channel, input.height, input.width);

    // Output tensor configuration
    cudnnTensorDescriptor_t outputDesc;
    cudnnCreateTensorDescriptor(&outputDesc);
    cudnnSetTensor4dDescriptor(outputDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, output.channel, output.height, output.width);

    // Filter configuration
    cudnnFilterDescriptor_t filterDesc;
    cudnnCreateFilterDescriptor(&filterDesc);
    cudnnSetFilter4dDescriptor(filterDesc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, filter.numKernels, filter.channel, filter.height, filter.width);

    // Convolution operation configuration
    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnSetConvolution2dDescriptor(convDesc, P, P, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_DOUBLE);

    // Algorithm selection with automatic tuning
    cudnnConvolutionFwdAlgoPerf_t perfResults;
    cudnnConvolutionFwdAlgo_t selectedAlgo;
    int returnedCount;
    cudnnFindConvolutionForwardAlgorithm(cudnn, inputDesc, filterDesc, convDesc, outputDesc, 1, &returnedCount, &perfResults);
    selectedAlgo = perfResults.algo;

    // Workspace allocation for selected algorithm
    size_t workspaceSize;
    cudnnGetConvolutionForwardWorkspaceSize(cudnn, inputDesc, filterDesc, convDesc, outputDesc, selectedAlgo, &workspaceSize);

    void *workspace;
    cudaMalloc(&workspace, workspaceSize);

    // Performance measurement setup
    double alpha = 1.0, beta = 0.0;
    cudaEvent_t start_time, end_time;
    cudaEventCreate(&start_time);
    cudaEventCreate(&end_time);
    cudaEventRecord(start_time);

    // Execute cuDNN convolution
    cudnnConvolutionForward(cudnn, &alpha, inputDesc, device_input.elements, filterDesc, device_filter.elements,
                            convDesc, selectedAlgo, workspace, workspaceSize, &beta,
                            outputDesc, device_output.elements);

    cudaThreadSynchronize();

    // Timing measurement
    cudaEventRecord(end_time);
    cudaEventSynchronize(end_time);
    cudaEventElapsedTime(&executionTime, start_time, end_time);

    // Copy results back to host
    size_t size = output.width * output.height * output.channel * sizeof(double);
    cudaMemcpy(output.elements, device_output.elements, size, cudaMemcpyDeviceToHost);

    // Resource cleanup
    cudaFree(workspace);
    cudaFree(device_input.elements);
    cudaFree(device_filter.elements);
    cudaFree(device_output.elements);
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);
}

int main() {
    float executionTimeC1, executionTimeC2, executionTimeC3 = 0.0;
    double checksumC1, checksumC2, checksumC3 = 0.0;

    Matrix inputMatrix = getHostInput(W, H, C);               // Original data
    Matrix paddedInputMatrix = padHostInput(inputMatrix, P);  // Data with padding
    Filter convolutionFilter = getHostFilter(FW, FH, C, K);   // Filter weights
    Matrix outputMatrix1 = hostMatrix(W, H, K);               // Results from method 1
    Matrix outputMatrix2 = hostMatrix(W, H, K);               // Results from method 2
    Matrix outputMatrix3 = hostMatrix(W, H, K);               // Results from method 3

    // Run basic CUDA implementation
    convolutionC1(paddedInputMatrix, convolutionFilter, outputMatrix1, executionTimeC1);
    checksumC1 = getCheckSum(outputMatrix1);

    // Run shared memory implementation
    convolutionC2(paddedInputMatrix, convolutionFilter, outputMatrix2, executionTimeC2);
    checksumC2 = getCheckSum(outputMatrix2);

    // Run cuDNN implementation
    convolutionC3(inputMatrix, convolutionFilter, outputMatrix3, executionTimeC3);
    checksumC3 = getCheckSum(outputMatrix3);

    printf("Checksum : %.0f, Time : %.3f millisec\n", checksumC1, executionTimeC1);
    printf("Checksum : %.0f, Time : %.3f millisec\n", checksumC2, executionTimeC2);
    printf("Checksum : %.0f, Time : %.3f millisec\n", checksumC3, executionTimeC3);

    free(inputMatrix.elements);
    free(paddedInputMatrix.elements);
    free(convolutionFilter.elements);
    free(outputMatrix1.elements);
    free(outputMatrix2.elements);
    free(outputMatrix3.elements);

    return 0;
}