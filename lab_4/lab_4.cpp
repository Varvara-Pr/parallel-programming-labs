#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

vector<vector<int>> readMatrix(const string& filename, int& size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << "\n";
        exit(1);
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

void writeMatrix(const string& filename, const vector<vector<int>>& matrix, int size, 
                 long long microseconds, long long operations, int blockX, int blockY, int gridX, int gridY) {
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

    file << "Время выполнения: " << microseconds << " microseconds (" << microseconds / 1e6 << " s)\n";
    file << "Объем задачи: " << operations << " operations\n";
    file << "Размер блока: " << blockX << "x" << blockY << "\n";
    file << "Сетка блоков: " << gridX << "x" << gridY << "\n";
    file.close();
}

__global__ void multiplyMatricesCUDA(const int* A, const int* B, int* C, int size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < size && col < size) {
        int sum = 0;
        for (int k = 0; k < size; k++) {
            sum += A[row * size + k] * B[k * size + col];
        }
        C[row * size + col] = sum;
    }
}

int main() {
    string fileA = "C:/Users/Varya/Desktop/lab_4/data/matrix_a.txt";
    string fileB = "C:/Users/Varya/Desktop/lab_4/data/matrix_b.txt";
    string fileC = "C:/Users/Varya/Desktop/lab_4/data/result_cuda.txt";
    
    int size;
    vector<vector<int>> A = readMatrix(fileA, size);
    vector<vector<int>> B = readMatrix(fileB, size);
    
    long long operations = 2LL * size * size * size;
    
    vector<int> flatA(size * size);
    vector<int> flatB(size * size);
    vector<int> flatC(size * size);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            flatA[i * size + j] = A[i][j];
            flatB[i * size + j] = B[i][j];
        }
    }
    
    int *d_A, *d_B, *d_C;
    size_t bytes = size * size * sizeof(int);
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, flatA.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, flatB.data(), bytes, cudaMemcpyHostToDevice);
    
    struct BlockConfig {
        int x;
        int y;
    };
    
    BlockConfig configs[] = {
        {8, 8},
        {8, 16},
        {16, 16},
        {16, 32},
        {32, 32}
    };
    
    int numConfigs = 5;
    
    cout << "Matrix size: " << size << "x" << size << "\n";
    cout << "Operations: " << operations << "\n\n";
    cout << "Results:\n";
    cout << "--------------------------------------------------------\n";
    
    for (int c = 0; c < numConfigs; c++) {
        int blockX = configs[c].x;
        int blockY = configs[c].y;
        
        if (blockX > size || blockY > size) continue;
        
        dim3 threadsPerBlock(blockX, blockY);
        dim3 blocksPerGrid((size + blockX - 1) / blockX, (size + blockY - 1) / blockY);
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaDeviceSynchronize();
        
        const int repeats = 10;
        float totalMilliseconds = 0;
        
        for (int r = 0; r < repeats; r++) {
            cudaEventRecord(start);
            multiplyMatricesCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            totalMilliseconds += milliseconds;
        }
        
        float avgMilliseconds = totalMilliseconds / repeats;
        long long microseconds = (long long)(avgMilliseconds * 1000);
        
        if (microseconds < 1) microseconds = 1;
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        cout << "Block " << blockX << "x" << blockY 
             << ", Grid " << blocksPerGrid.x << "x" << blocksPerGrid.y
             << ", Time: " << microseconds << " mcs (" << microseconds / 1e6 << " s)\n";
        
        if (blockX == 16 && blockY == 16) {
            cudaMemcpy(flatC.data(), d_C, bytes, cudaMemcpyDeviceToHost);
            
            vector<vector<int>> C(size, vector<int>(size));
            for (int i = 0; i < size; i++) {
                for (int j = 0; j < size; j++) {
                    C[i][j] = flatC[i * size + j];
                }
            }
            writeMatrix(fileC, C, size, microseconds, operations, blockX, blockY, blocksPerGrid.x, blocksPerGrid.y);
        }
    }
    
    cout << "--------------------------------------------------------\n";
    cout << "Result saved: " << fileC << "\n";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}