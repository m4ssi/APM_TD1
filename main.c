#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

pthread_mutex_t m_stdout = PTHREAD_MUTEX_INITIALIZER;

typedef struct
{
	int blockDim_x,
		blockIdx_x,
		dimGrid_x,
		threadIdx_x,
		N;
	double	* a,
			* b,
			* c;
} arg_t;

void kernel ( double * a, double * b, double * c, int N, int blockDim_x, int blockIdx_x, int threadIdx_x)
{
	int i = blockIdx_x * blockDim_x + threadIdx_x;
	if ( i < N)
	{
		c[i] = a[i] + b[i];
	}
}

void * routine ( void * arg)
{
	arg_t * a = ( arg_t *) arg;
	kernel ( a->a, a->b, a->c, a->N, a->blockDim_x, a->blockIdx_x, a->threadIdx_x);
	return NULL;
}

int main ( int argc, char ** argv)
{
	int N = 1000;
	int sz_in_bytes = N*sizeof(double);
	
	double * h_a, * h_b, * h_c;
	double * d_a, * d_b, * d_c;
	
	h_a = (double *) malloc ( sz_in_bytes);
	h_b = (double *) malloc ( sz_in_bytes);
	h_c = (double *) malloc ( sz_in_bytes);
	
	// Initiate vallues on h_a and h_b
	for (int i = 0; i < N; i++)
	{
		h_a[i] = 1./(1.+i);
		h_b[i] = (i-1.)/(i+1.);
	}

	// 3-arrays allocation on "device"
	d_a = (double *) malloc ( sz_in_bytes);
	d_b = (double *) malloc ( sz_in_bytes);
	d_c = (double *) malloc ( sz_in_bytes);
	
	// copy on device values pointed on host by h_a and h_b
	// (the new values are pointed by d_a and d_b on device)
	memcpy ( d_a, h_a, sz_in_bytes);
	memcpy ( d_b, h_b, sz_in_bytes);
	
	// dim3 dimBlock(64, 1, 1);
	int dimBlock_x = 64;
	
	// dim3 dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);
	int dimGrid_x = (N + dimBlock_x - 1)/dimBlock_x;

	// Preparing variables to pass as CUDA parameters for each thread
	arg_t ** a;
	pthread_t ** tid;
	
	a = malloc ( dimGrid_x * sizeof ( arg_t *));
	tid = malloc ( dimGrid_x * sizeof ( pthread_t *));
	for ( int i = 0; i < dimGrid_x; i++)
	{
		a[i] = malloc ( dimBlock_x * sizeof ( arg_t));
		tid[i] = malloc ( dimBlock_x * sizeof ( pthread_t));
	}
	// Init variables to emulate cudaMemcpy, gridDim, blockDim, blockIdx and threadIdx
	for ( int i = 0; i < dimGrid_x; i++)
	{
		for ( int j = 0; j < dimBlock_x; j++)
		{
			a[i][j].a = d_a;
			a[i][j].b = d_b;
			a[i][j].c = d_c;
			a[i][j].N = N;
			a[i][j].blockDim_x = dimBlock_x;
			a[i][j].blockIdx_x = i;
			a[i][j].dimGrid_x = dimGrid_x;
			a[i][j].threadIdx_x = j;
		}
	}
	// Creating threads
	// kernel<<<dimGrid , dimBlock>>>(d_a, d_b, d_c, N)
	for ( int i = 0; i < dimGrid_x; i++)
	{
		for ( int j = 0; j < dimBlock_x; j++)
		{
			pthread_create ( &(tid[i][j]), NULL, routine, &(a[i][j]));
		}
	}

	for ( int i = 0; i < dimGrid_x; i++)
	{
		for ( int j = 0; j < dimBlock_x; j++)
		{
			pthread_join ( tid[i][j], NULL);
		}
	}

	// Result is pointed by d_c on device
	// Copy this result on host (result pointed by h_c on host)
	memcpy ( h_c, d_c, sz_in_bytes);

	// freeing on device
	free ( d_a), free ( d_b), free ( d_c);
	
	free ( h_a), free ( h_b), free ( h_c);
	return 0;
}
