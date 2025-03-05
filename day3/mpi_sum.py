from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Compute the sum over all ranks
sum_result = comm.reduce(rank, op=MPI.SUM, root=0)

# Print the result from rank 0
if rank == 0:
    print("Sum over all ranks: %d" %sum_result)