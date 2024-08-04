#include <cuda_runtime.h>
#include <iostream>

//Width and Height are 16 respectively
#define BLOCK_SIZE 16

struct Matrix{
	int* elementPtr;
	int width;
	int stride; 
	//Pitch is stride in bytes, essentially.
	//Stride is how many elements has been allocated on that row with padding
	//while pitch is strides in bytes per row mentioned as above.
	//In this example, we will be using width = stride.
	int height;
};

__device__ void setElement(Matrix mat, int column, int row, int value){
	if(column < mat.width && row < mat.height){
		*(mat.elementPtr + row * mat.stride + column) = value;
	}
}

__device__ int getElement(Matrix mat, int column, int row){
	return *(mat.elementPtr + row * mat.stride + column);
}

__device__ Matrix returnSubMat(Matrix main, int subBlockColumn, int subBlockRow){
	Matrix sub;
	sub.width = BLOCK_SIZE;
	sub.height = BLOCK_SIZE;
	sub.stride = main.stride;
	sub.elementPtr = &main.elementPtr[ main.stride * subBlockRow * BLOCK_SIZE
									     + subBlockColumn * BLOCK_SIZE];
	/*  subBlockRow * BLOCK_SIZE determines which row we should reach.
		Multiplying above by main.stride travels elements in row-major ordering
		to reach to the first element in our subBlock's row.
		After adding subBlockColumn * BLOCK_SIZE, we are now in the first thread of our block.
		We place the pointer here.
	*/
	/*
	 	4 x 4 matrix

		|--|--|  --> Here |--| symbolizes a 2 X 2 thread block.
		|--|--|			  |--|
		|--|/*|
		|--|**|  --> Total of 4 thread blocks

		In order to make our pointer point to /
		we do the above procedure.	
	*/

	return sub;
}

//Method declerations for usage from main method
int* matrixMul(const Matrix A, const Matrix B, Matrix C);
__global__ void matMul(const Matrix, const Matrix, Matrix);

//Generate sample matrices. Use it in CPU -- Host
void generateMatrix(Matrix& mat, int value){
	for(int iCol = 0; iCol < mat.width; iCol++){
		for(int iRow = 0; iRow < mat.height; iRow++){
			*(mat.elementPtr + iRow * mat.stride + iCol) = value;
		}
	}
}

int main(){
	//Create the sample arrays
	Matrix A;
	A.width = 64;
	A.stride = 64;
	A.height = 64;

	//Allocate space for samples
	size_t size = A.width * A.height * sizeof(int);
	A.elementPtr = (int*)malloc(size);

	//Generate with random values
	int valueA = 2;
	generateMatrix(A, valueA);

	Matrix B;
	B.width = 64;
	B.stride = 64;
	B.height = 64;

	//Allocate space for samples
	size = B.width * B.height * sizeof(int);
	B.elementPtr = (int*)malloc(size);

	//Generate with random values
	int valueB = 3;
	generateMatrix(B, valueB);

	//Empty matrix result C
	Matrix C;
	C.width = 64;
	C.stride = 64;
	C.height = 64;

	//Allocate space for samples
	size = C.width * C.height * sizeof(int);
	C.elementPtr = (int*)malloc(size);

	std::cout << "Generated A , B and C in main()" << std::endl;

	std::cout << "Entering matrixMul()" << std::endl;
	int* resultPtr_C = matrixMul(A, B ,C);

	std::cout << "Exited matrixMul, kernel should have finished. Stopping the program after printing a value" << std::endl;
	std::cout << "First index of the resulting C matrix is: " << *resultPtr_C << std::endl;
	std::cout << "All indexes of the A is: " << *A.elementPtr << std::endl;
	std::cout << "All indexes of the B is: " << *B.elementPtr << std::endl;
	std::cout << "Expected value for all the indexes of resulting C was: " << valueA * valueB * 64 << std::endl;
	return 0;
}

int* matrixMul(const Matrix A, const Matrix B, Matrix C){
	//Load A to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.stride = A.stride;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(int);

	//Allocate space in device memory
	cudaMalloc(&d_A.elementPtr, size);
	//Transfer (copy) the data to device
	cudaMemcpy(d_A.elementPtr, A.elementPtr, size, cudaMemcpyHostToDevice);

	//Load B to device memory
	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	d_B.stride = B.stride;
	size = B.height * B.width * sizeof(int);

	cudaMalloc(&d_B.elementPtr, size);
	cudaMemcpy(d_B.elementPtr, B.elementPtr, size, cudaMemcpyHostToDevice);

	//Prepare a space for the resulting matrx C, allocate memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.stride = C.stride;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(int);

	cudaMalloc(&d_C.elementPtr, size);

	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

	matMul<<< dimGrid, dimBlock >>>(d_A, d_B, d_C);

	//Copy back the resulting C matrix
	cudaMemcpy(C.elementPtr, d_C.elementPtr, size, cudaMemcpyDeviceToHost);

	//CLear the device memory pointers allocation
	cudaFree(d_A.elementPtr);
	cudaFree(d_B.elementPtr);
	cudaFree(d_C.elementPtr);

	//Return the pointer pointing to the first index of the matrix --> Just for checking
	return C.elementPtr;
}

__global__ void matMul(const Matrix A, const Matrix B, Matrix C){
	int subBlockColumn = blockIdx.x;
	int subBlockRow = blockIdx.y;

	Matrix subC = returnSubMat(C, subBlockColumn, subBlockRow);

	int accMul_C = 0;

	int tidX = threadIdx.x;
	int tidY = threadIdx.y;

	for(int m = 0; m < A.width / BLOCK_SIZE ; m++){
		//Return the sub matrices for tiled multiplication
		Matrix subA = returnSubMat(A, m, subBlockRow);
		Matrix subB = returnSubMat(B, subBlockColumn, m);

		//Create a shared memory (scratcpad memory) for shared loading
		//This is a shared memory for a thread block
		//Threads within thread block will work together to get the values from DRAM
		//However, each thread within threadblock will calculate the portion of their own sum after synching.
		__shared__ int A_vals[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ int B_vals[BLOCK_SIZE][BLOCK_SIZE];

		//Load to the shared memory individual elements.
		A_vals[tidY][tidX] = getElement(subA, tidX, tidY);
		B_vals[tidY][tidX] = getElement(subB, tidX, tidY);

		__syncthreads(); //Synchronise before calculating the accMul_C

		for(int i = 0; i < BLOCK_SIZE; i++){
			accMul_C += A_vals[tidY][i] * B_vals[i][tidX];
		}

		__syncthreads(); //Synchronise before preceeding into other iteration
	}

	//Record the resulting multplication into the one thread in subC
	setElement(subC, tidX, tidY, accMul_C);
}