#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace chrono;

vector<vector<int>> readMatrix(const string& filename, int& size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << "\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    file >> size;
    vector<vector<int>> matrix(size, vector<int>(size));

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }

    file.close();
    return matrix;
}

void writeMatrix(const string& filename, const vector<vector<int>>& matrix, int size, long long microseconds, long long operations, int num_processes) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error creating file: " << filename << "\n";
        exit(1);
    }

    file << size << "\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file << matrix[i][j];
            if (j < size - 1) file << " ";
        }
        file << "\n";
    }

    file << "Время выполнения: " << microseconds << " microseconds\n";
    file << "Объем задачи: " << operations << " operations\n";
    file << "Количество процессов: " << num_processes << "\n";
    file.close();
}

vector<vector<int>> multiplyMatricesMPI(const vector<vector<int>>& A,
    const vector<vector<int>>& B,
    int size, int rank, int num_processes) {
    vector<vector<int>> C(size, vector<int>(size, 0));

    int rows_per_process = size / num_processes;
    int start_row = rank * rows_per_process;
    int end_row = (rank == num_processes - 1) ? size : start_row + rows_per_process;


    for (int i = start_row; i < end_row; i++) {
        for (int k = 0; k < size; k++) {
            int aik = A[i][k];
            for (int j = 0; j < size; j++) {
                C[i][j] += aik * B[k][j];
            }
        }
    }

    if (rank == 0) {
        for (int p = 1; p < num_processes; p++) {
            int p_start_row = p * rows_per_process;
            int p_end_row = (p == num_processes - 1) ? size : p_start_row + rows_per_process;
            int p_rows = p_end_row - p_start_row;

            vector<vector<int>> temp(p_rows, vector<int>(size));
            for (int i = 0; i < p_rows; i++) {
                MPI_Recv(temp[i].data(), size, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }

            for (int i = 0; i < p_rows; i++) {
                C[p_start_row + i] = temp[i];
            }
        }
    }
    else {
        int local_rows = end_row - start_row;
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(C[start_row + i].data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    return C;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

    string fileA = "data/matrix_a.txt";
    string fileB = "data/matrix_b.txt";
    string fileC = "data/result_mpi.txt";

    int size = 0;
    vector<vector<int>> A, B;

    if (rank == 0) {
        int sizeA, sizeB;
        A = readMatrix(fileA, sizeA);
        B = readMatrix(fileB, sizeB);

        if (sizeA != sizeB) {
            cerr << "Error: matrix dimensions do not match" << "\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size = sizeA;

        cout << "Matrix size: " << size << "x" << size << "\n";
        cout << "Number of processes: " << num_processes << "\n";
    }

    MPI_Bcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        B.resize(size, vector<int>(size));
    }
    for (int i = 0; i < size; i++) {
        MPI_Bcast(B[i].data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank != 0) {
        A.resize(size, vector<int>(size));
    }

    int rows_per_process = size / num_processes;
    vector<int> send_counts(num_processes, rows_per_process * size);
    send_counts[num_processes - 1] = (size - (num_processes - 1) * rows_per_process) * size;

    vector<int> displs(num_processes, 0);
    for (int i = 1; i < num_processes; i++) {
        displs[i] = displs[i - 1] + send_counts[i - 1];
    }

    vector<int> flat_A;
    if (rank == 0) {
        flat_A.resize(size * size);
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                flat_A[i * size + j] = A[i][j];
            }
        }
    }

    vector<int> local_flat_A(rows_per_process * size);
    MPI_Scatterv(flat_A.data(), send_counts.data(), displs.data(), MPI_INT,
        local_flat_A.data(), rows_per_process * size, MPI_INT,
        0, MPI_COMM_WORLD);

    vector<vector<int>> local_A(rows_per_process, vector<int>(size));
    for (int i = 0; i < rows_per_process; i++) {
        for (int j = 0; j < size; j++) {
            local_A[i][j] = local_flat_A[i * size + j];
        }
    }

    long long operations = 2LL * size * size * size;
    auto start_time = MPI_Wtime();

    vector<vector<int>> C(size, vector<int>(size, 0));

    int start_row = rank * rows_per_process;
    int end_row = (rank == num_processes - 1) ? size : start_row + rows_per_process;
    int local_rows = end_row - start_row;

    for (int i = 0; i < local_rows; i++) {
        int global_i = start_row + i;
        for (int k = 0; k < size; k++) {
            int aik = local_A[i][k];
            for (int j = 0; j < size; j++) {
                C[global_i][j] += aik * B[k][j];
            }
        }
    }

    if (rank == 0) {
        for (int p = 1; p < num_processes; p++) {
            int p_start_row = p * rows_per_process;
            int p_end_row = (p == num_processes - 1) ? size : p_start_row + rows_per_process;
            int p_rows = p_end_row - p_start_row;

            for (int i = 0; i < p_rows; i++) {
                MPI_Recv(C[p_start_row + i].data(), size, MPI_INT, p, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    }
    else {
        for (int i = 0; i < local_rows; i++) {
            MPI_Send(C[start_row + i].data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    auto end_time = MPI_Wtime();
    double elapsed_seconds = end_time - start_time;
    long long microseconds = static_cast<long long>(elapsed_seconds * 1e6);

    if (rank == 0) {
        cout << "Operations: " << operations << "\n";
        cout << "Time: " << microseconds << " mcs (" << elapsed_seconds << " s)\n";
        writeMatrix(fileC, C, size, microseconds, operations, num_processes);
        cout << "Result written to " << fileC << "\n";
    }

    MPI_Finalize();
    return 0;
}