#include "generalCuda.cuh"
//#include <stdio.h>
//#include <math.h>
//#include <shrUtils.h>


__device__ inline int getutid()
{
int threadsPerBlock = blockDim.x * blockDim.y;
int tidWithinBlock = threadIdx.x + threadIdx.y * blockDim.x;
int gid = blockIdx.x + blockIdx.y * gridDim.x;
return gid * threadsPerBlock + tidWithinBlock;
}

__host__ __device__ double function(double x){

	double F= 0.0f;

	//tutaj wzór funkcji
	F=0.1f*(x+3)*(x+2)*(x+1)*(x)*(x-3)*(x-5)-12;
	//F=x*x;
	return F;
}

__host__ __device__ double derivative(double x, double h){
	if(h==0){
		int j =0;//exit(-1); //printf("\n Parameter h can not be equal zero");
	};
	double d = 0.0f;
	d=(function(x+h)-function(x-h))/(2*h);
	return d;
}

__host__ __device__ float functionF(float x){

	float F= 0.0f;

	//tutaj wzór funkcji
	F=0.1f*(x+3)*(x+2)*(x+1)*(x)*(x-3)*(x-5)-12;
	//F=x*x;
	return F;
}

__host__ __device__ float derivativeF(float x, float h){
	if(h==0){
		int j =0;//exit(-1); //printf("\n Parameter h can not be equal zero");
	};
	float d = 0.0f;
	d=(functionF(x+h)-functionF(x-h))/(2*h);
	return d;
}

__global__ void compKernel(float *d_minima,float end,float start,float incr,float h,int noOfThreads){
	int x1 = blockIdx.x*blockDim.x + threadIdx.x;
	int y1 = blockIdx.y* blockDim.y + threadIdx.y;
	int thIndex = getutid();//y1*gridDim.x +x1;
	float d =0.0f;
	float dnext = 0.0f;
	float dprev = 0.0f;
	float x = (float)start+thIndex*incr;
	
	d_minima[thIndex] = start-10.0f;
	if(x<end){
		//d_debug[thIndex]=x;
		d=derivativeF(x,h);
		dnext=derivativeF(x+incr,h);
		dprev=derivativeF(x-incr,h);

		if(thIndex>=noOfThreads ||thIndex<0) return;
		if(dprev<=0.0f && dnext>=0.0f){// ((d<threshold) && ( d>-threshold)) ){
			//if(dprev<=0.0f && dnext>=0.0f){
				d_minima[thIndex]=x;
		}else{
			d_minima[thIndex] = start-10.0f;
		}
	}
	return;


}

extern "C"  void callkernel(float* h_tableOfMinima, float end,float start, int noOfSamples,float incr, float h){
	
	float *d_minima_x = NULL;
	float *d_derivatives = NULL;
	float *d_debug = NULL;
	float *h_debug = NULL;
	int noOfThreads = (int)((end-start+incr)/incr) +1;

	noOfThreads = 16*16*32*32;
	 cudaMalloc ( (void**) &d_minima_x,  noOfThreads * sizeof(float) );
	 //cudaMalloc ( (void**) &d_debug,  noOfThreads * sizeof(float) );
	// h_debug = (float *) malloc(noOfThreads*sizeof(float));

	cudaError err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf( "Cuda error: %s: %s.\n", "error", 
                                  cudaGetErrorString( err) );
    } 
	//cutilSafeCall( cudaMalloc( (void**) &d_minima_x,  (noOfThreads * sizeof(float)) ));

	dim3 dimBlock(16,16);
	uint sqrtResult=(uint)sqrtl((long)noOfThreads);
	dim3 dimGrid(32,32);
	 err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf( "Cuda error: %s: %s.\n", "error", 
                                  cudaGetErrorString( err) );
    }  

	uint x = dimGrid.x;

	compKernel<<<dimGrid,dimBlock>>>(d_minima_x,end,start,(end-start)/((float)noOfThreads+1.0f),h,noOfThreads/*,d_debug*/);
	//compKernel<<<dimGrid,dimBlock>>>(d_minima_x,,-3.2f,9.5f/(float)noOfThreads,0.001f,noOfThreads,d_debug);
	cutilSafeCall( cudaThreadSynchronize());
	 err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        printf( "Cuda error: %s: %s.\n", "error", 
                                  cudaGetErrorString( err) );
    }  

	cutilSafeCall(cudaMemcpy(h_tableOfMinima,d_minima_x,noOfThreads*sizeof(float), cudaMemcpyDeviceToHost));
	//cutilSafeCall(cudaMemcpy(h_debug,d_debug,noOfThreads*sizeof(float), cudaMemcpyDeviceToHost));

	/*for(int i =0;i<noOfThreads;i++){
		 if(h_debug[i]!=0.0f)printf ("\n i = %d  val = %f ",i,h_debug[i]);
	}*/

	cudaFree(d_minima_x);
}