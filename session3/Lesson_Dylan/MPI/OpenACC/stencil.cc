/* stencil.cc
 * by: G. Dylan Dickerson (gdicker@ucar.edu)
 */

#include <openacc.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>
#include <openacc.h>
#include "pch.h"

#define MAX(X,Y) (((X) > (Y)) ? (X) : (Y))

void mapGPUToMPIRanks(int rank){
	// Get device count
	const int num_dev = acc_get_num_devices(acc_device_nvidia);
	const int dev_id = rank % num_dev;
	acc_set_device_num(dev_id, acc_device_nvidia);
}

LJ_return LaplaceJacobi_naiveACC(float *M, const int ny, const int nx){
	/*
	 * Use an iterative Jacobi solver to find the steady-state of
	 * the differential equation of the Laplace equation in 2 dimensions.
	 * M models the initial state of the system and is used to return the
	 * result in-place. M has a border of b entries that aren't updated
	 * by the Jacobi solver. For the iterative Jacobi method, the unknowns
	 * are a flattened version of the interior points. See another source
	 * for more information.
	 */
	int itr = 0;
	float maxdiff = 0.0f;
	float *M_new;
	LJ_return ret;

	// Allocate the second version of the M matrix used for the computation
	M_new = (float*)malloc(ny*nx*sizeof(float));

	#pragma acc data copy(M[0:ny*nx]) create(M_new[0:ny*nx])
	{
	do{
		maxdiff = 0.0f;
		itr++;
		#pragma acc parallel copy(maxdiff)
		{
		// Update M_new with M
		#pragma acc loop collapse(2)
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
							M[(i+1)*nx+j]+M[i*nx+j-1]);
			}
		}

		// Check for convergence while copying values into M
		#pragma acc loop collapse(2) reduction(max:maxdiff)
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
				M[i*nx+j] = M_new[i*nx+j];
			}
		}
		} // acc end parallel
	} while(itr < JACOBI_MAX_ITR && maxdiff > JACOBI_TOLERANCE);
	} // acc end data
	
	// Free malloc'd memory
	free(M_new);
	// Fill in the return value
	ret.itr = itr;
	ret.error = maxdiff;
	return ret;
}

LJ_return LaplaceJacobi_MPIACC(float *M, const int ny, const int nx,
			  const int rank, const int *coord, const int *neighbors){
/*
 * Performs the same calculations as naiveCPU, but also does a halo exchange
 * at the end of each iteration to update the ghost areas and uses OpenACC pragmas
 */
	int matsz = ny*nx,
	    buffsz_x = nx-2,
	    buffsz_y = ny-2;
	int itr = 0;
	float maxdiff = 0.0f, // The error for this process
	      g_maxdiff=0.0f; // The max error over all processes
	// Arrays used by this function
	// M_new is the version of M that is updated in the body of the loop before
	// being copied back into M at the end of an iteration
	float *M_new;
	LJ_return ret;

	// MPI Specific Variables
	// Arrays used to send the ghost area values to each neighbor
	float *send_top, *send_right, *send_bot, *send_left;
	// Arrays used to receive the ghost area values from each neighbor
	float *recv_top, *recv_right, *recv_bot, *recv_left;
	// Holds the statuses returned by MPI_Waitall related to a Irecv/Isend pair
	MPI_Status status[2];
	// Groups Irecv/Isend calls together from the sender's perspective and are
	// used by MPI_Waitall before putting received values into M_new
	// (e.g. requestR are the requests for receiving and sending to its right neighbor)
	MPI_Request requestT[2], requestR[2], requestB[2], requestL[2];
	// The (optional) tags for the MPI Isend/Irecv.
	// Tags are relative to the sender (e.g. a process sending data to its 
	// left neighbor uses tag_l in the Isend and the neighbor will use tag_l in its Irecv)
	int tag_t = DIR_TOP, tag_b=DIR_BOTTOM, tag_r=DIR_RIGHT, tag_l=DIR_LEFT;

	// Allocate local arrays
	M_new = (float*)malloc(matsz*sizeof(float));
	send_top = (float*)malloc(buffsz_x*sizeof(float));
	send_right = (float*)malloc(buffsz_y*sizeof(float));
	send_bot = (float*)malloc(buffsz_x*sizeof(float));
	send_left = (float*)malloc(buffsz_y*sizeof(float));
	recv_top = (float*)malloc(buffsz_x*sizeof(float));
	recv_right = (float*)malloc(buffsz_y*sizeof(float));
	recv_bot = (float*)malloc(buffsz_x*sizeof(float));
	recv_left = (float*)malloc(buffsz_y*sizeof(float));

	// Make M_new a copy of M, this helps for the last loop inside the do-while
	std::copy(M, M+(ny*nx), M_new);

#pragma acc enter data copyin(M[0:ny*nx], M_new[0:ny*nx])
#pragma acc enter data create(send_top[0:(nx-2)], recv_top[0:nx-2])
#pragma acc enter data create(send_right[0:(ny-2)], recv_right[0:ny-2])
#pragma acc enter data create(send_bot[0:(nx-2)], recv_bot[0:nx-2])
#pragma acc enter data create(send_left[0:(ny-2)], recv_left[0:ny-2])
	do {
		maxdiff = 0.0f;
		itr++;

		// Update M_new with M
#pragma acc parallel loop collapse(2) present(M[0:matsz], M_new[0:matsz])
		for(int i=1; i<ny-1; i++){
			for(int j=1; j<nx-1; j++){
				M_new[i*nx+j] = 0.25f *(M[(i-1)*nx+j]+M[i*nx+j+1]+ \
							M[(i+1)*nx+j]+M[i*nx+j-1]);
			}
		}
		//printf("Rank:%d finished jacobi update in M_new, starting halo exchange\n",rank); fflush(stdout);
		
		// Perform halo exchange
		if(HasNeighbor(neighbors, DIR_TOP)){
			//printf("Rank:%d Start top exchange\n", rank); fflush(stdout);
			
			// Copy the values from the top row of the interior
#pragma acc parallel loop present(send_top[0:buffsz_x], M_new[0:matsz])
			for(int j=1; j<nx-1; j++){
				send_top[j-1] = M_new[1*nx+j];
			}
			
			//printf("Rank:%d filled top send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_top\n",rank);fflush(stdout);
			//PrintMatrix(send_top, 1, nx-2); fflush(stdout);
#pragma acc host_data use_device(recv_top, send_top)
{			
			MPI_Irecv(recv_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_b, MPI_COMM_WORLD, requestT);
			MPI_Isend(send_top, nx-2, MPI_FLOAT, neighbors[DIR_TOP], tag_t, MPI_COMM_WORLD, requestT+1);
}			
			//printf("Rank:%d End top exchange\n", rank);fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_BOTTOM)){
			//printf("Rank:%d Start bottom exchange\n", rank);fflush(stdout);

			// Copy the values from the bottom row of the interior
#pragma acc parallel loop  present(send_bot[0:buffsz_x], M_new[0:matsz])
			for(int j=1; j<nx-1; j++){
				send_bot[j-1] = M_new[(ny-2)*nx+j];
			}

			//printf("Rank:%d filled bottom send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_bot\n",rank);fflush(stdout);
			//PrintMatrix(send_bot, 1, nx-2); fflush(stdout);

#pragma acc host_data use_device(recv_top, send_top)
{			
			MPI_Irecv(recv_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_t, MPI_COMM_WORLD, requestB);
			MPI_Isend(send_bot, nx-2, MPI_FLOAT, neighbors[DIR_BOTTOM], tag_b, MPI_COMM_WORLD, requestB+1);
}			
			//printf("Rank:%d End bottom exchange\n", rank);fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_RIGHT)){
			//printf("Rank:%d Start right exchange\n", rank);fflush(stdout);

			// Copy the values from the right column of the interior
#pragma acc parallel loop  present(send_right[0:buffsz_y], M_new[0:matsz])
			for(int i=1; i<ny-1; i++){
				send_right[i-1] = M_new[i*nx+(nx-2)];
			}

			//printf("Rank:%d filled right send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_right\n",rank);fflush(stdout);
			//PrintMatrix(send_right, 1, nx-2); fflush(stdout);

#pragma acc host_data use_device(recv_top, send_top)
{			
			MPI_Irecv(recv_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_l, MPI_COMM_WORLD, requestR);
			MPI_Isend(send_right, nx-2, MPI_FLOAT, neighbors[DIR_RIGHT], tag_r, MPI_COMM_WORLD, requestR+1);
}
			//printf("Rank:%d End right exchange\n", rank);fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_LEFT)){
			//printf("Rank:%d Start left exchange\n", rank);fflush(stdout);

			// Copy the values from the left column of the interior
#pragma acc parallel loop present(send_left[0:buffsz_y], M_new[0:matsz])  
			for(int i=1; i<ny-1; i++){
				send_left[i-1] = M_new[i*nx+1];
			}

			//printf("Rank:%d filled left send buffer\n", rank); fflush(stdout);
			//printf("\tRank:%d send_left\n",rank);fflush(stdout);
			//PrintMatrix(send_left, 1, nx-2); fflush(stdout);

#pragma acc host_data use_device(recv_top, send_top)
{			
			MPI_Irecv(recv_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_r, MPI_COMM_WORLD, requestL);
			MPI_Isend(send_left, nx-2, MPI_FLOAT, neighbors[DIR_LEFT], tag_l, MPI_COMM_WORLD, requestL+1);
}			//printf("Rank:%d End left exchange\n", rank);fflush(stdout);
			//printf("Rank:%d End left exchange\n", rank);fflush(stdout);
		}

		// Wait for the values and fill in the correct areas of M_new
		if(HasNeighbor(neighbors, DIR_TOP)){ // Fill the values in the top row
			MPI_Waitall(2, requestT, status);

			//printf("Rank:%d using recv_top to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_top\n", rank);fflush(stdout);
			//PrintMatrix(recv_top, 1, nx-2); fflush(stdout);

#pragma acc parallel loop present(recv_top[0:buffsz_x], M_new[0:matsz])  
			for(int j=1; j<nx-1; j++){
				M_new[j] = recv_top[j-1];
			}

			//printf("Rank:%d filled M_new with recv_top\n", rank); fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_BOTTOM)){
			MPI_Waitall(2, requestB, status);

			//printf("Rank:%d using recv_bot to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_bot\n", rank);fflush(stdout);
			//PrintMatrix(recv_bot, 1, nx-2); fflush(stdout);

#pragma acc parallel loop present(recv_bot[0:buffsz_x], M_new[0:matsz]) 
			for(int j=1; j<nx-1; j++){ // Fill the values in the bottom row
				M_new[(ny-1)*nx+j] = recv_bot[j-1];	
			}

			//printf("Rank:%d filled M_new with recv_bot\n", rank); fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_RIGHT)){
			MPI_Waitall(2, requestR, status);

			//printf("Rank:%d using recv_right to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_right\n", rank);fflush(stdout);
			//PrintMatrix(recv_right, 1, nx-2); fflush(stdout);

#pragma acc parallel loop present(recv_right[0:buffsz_y], M_new[0:matsz])  
			for(int i=1; i<ny-1; i++){ // Fill the values in the rightmost column
				M_new[i*nx+(nx-1)] = recv_right[i-1];
			}

			//printf("Rank:%d filled M_new with recv_right\n", rank); fflush(stdout);
		}
		if(HasNeighbor(neighbors, DIR_LEFT)){
			MPI_Waitall(2, requestL, status);

			//printf("Rank:%d using recv_left to fill M_new\n", rank); fflush(stdout);
			//printf("Rank:%d recv_left\n", rank);fflush(stdout);
			//PrintMatrix(recv_left, 1, nx-2); fflush(stdout);

#pragma acc parallel loop present(recv_left[0:buffsz_y], M_new[0:matsz])  
			for(int i=1; i<ny-1; i++){ // Fill the values in the leftmost column
				M_new[i*nx] = recv_left[i-1];
			}

			//printf("Rank:%d filled M_new with recv_left\n", rank); fflush(stdout);
		}

		//printf("Rank:%d End halo exchange\n", rank); fflush(stdout);
		// End the halo exchange section

		// Check for convergence while copying values into M
#pragma acc parallel loop collapse(2) reduction(max:maxdiff) \
 present(M[0:matsz],M_new[0:matsz]) copy(maxdiff)
		for(int i=0; i<ny; i++){
			for(int j=0; j<nx; j++){
				maxdiff = MAX(fabs(M_new[i*nx+j] - M[i*nx+j]), maxdiff);
				M[i*nx+j] = M_new[i*nx+j];
			}
		}
		// Find the global max difference. Have each process exit when the global error is low enough
		MPI_Allreduce(&maxdiff, &g_maxdiff, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
		
		//printf("Rank:%d Completed transfer and iteration %d\n",rank, itr); fflush(stdout);
	} while(itr < JACOBI_MAX_ITR && g_maxdiff > JACOBI_TOLERANCE);

#pragma acc exit data copyout(M[0:matsz]) delete(M_new[0:matsz])
#pragma acc exit data delete(send_top[0:buffsz_x], send_right[0:buffsz_y])
#pragma acc exit data delete(send_bot[0:buffsz_x], send_left[0:buffsz_y])
#pragma acc exit data delete(recv_top[0:buffsz_x], recv_right[0:buffsz_y])
#pragma acc exit data delete(recv_bot[0:buffsz_x], recv_left[0:buffsz_y])

	//printf("Rank:%d MPI-ACC Jacobi exiting on itr=%d of max_itr=%d with error=%f vs threshold=%f\n", rank, itr, JACOBI_MAX_ITR, maxdiff, JACOBI_TOLERANCE);

	// Free malloc'ed memory
	free(M_new);
	free(send_top);
	free(send_right);
	free(send_bot);
	free(send_left);
	free(recv_top);
	free(recv_right);
	free(recv_bot);
	free(recv_left);	

	// Fill in the return value
	ret.itr = itr;
	ret.error = maxdiff;
	return ret;
}