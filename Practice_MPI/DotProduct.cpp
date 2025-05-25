#include <iostream>
#include <mpi.h>
#include <omp.h>

int main(int argc, char ** argv){

    MPI_Init(&argc, &argv);

    int world_size;
    int world_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    const int N = 1024;
    float * A = new float[N];
    float * B = new float[N];

    // Only root initializes data
    if(world_rank == 0){
        for (int i = 0; i<N; i++){
            A[i] = 2.0f;
            B[i] = 1.0f;
        }
    }

    // Broadcast vectors to all processes
    MPI_Bcast(A, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(world_rank == 0){
        int nT = world_size - 1;
        int Workload = N / nT;
        int remWorkload = N % nT;
        int istart;
        int iend = 0;

        MPI_Request *request = new MPI_Request[nT];
        float *partial_sums = new float[nT];
        
        for (int i=0; i<nT; i++) {
            istart = iend;
            iend = i<remWorkload ? istart+Workload+1 : istart+Workload;
            MPI_Send(&istart, 1, MPI_INT, i+1, 1, MPI_COMM_WORLD);
            MPI_Send(&iend, 1, MPI_INT, i+1, 2, MPI_COMM_WORLD);
            MPI_Irecv(&partial_sums[i], 1, MPI_FLOAT, i+1, 3, MPI_COMM_WORLD, &request[i]);
        }
        
        MPI_Waitall(nT, request, MPI_STATUSES_IGNORE);

        float total_dot_product = 0.0f;
        for(int i = 0; i < nT; i++) {
            total_dot_product += partial_sums[i];
        }
        std::cout << "Dot product result: " << total_dot_product << std::endl;
        
        delete[] request;
        delete[] partial_sums;
    }
    else {
        int istart = 0;
        int iend = 0;

        MPI_Recv(&istart, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&iend, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        float local_sum = 0.0f;

        #pragma omp parallel for
        for(int i = istart; i < iend; i++){
            local_sum += A[i] * B[i];
        }

        MPI_Send(&local_sum, 1, MPI_FLOAT, 0, 3, MPI_COMM_WORLD);
    }
    
    delete[] A;
    delete[] B;
    MPI_Finalize();
    return 0;
}