import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


def matrix_multiplication(A, B, BlockSize):
    C = np.zeros((len(A), len(B[0])), dtype=int)
    for indexBlA in range(0, len(A[0]), BlockSize):
        for indexBlB in range(0, len(B[0]), BlockSize):
            for i in range(len(A)):
                for j in range(indexBlB, indexBlB + BlockSize):
                    C[i][j] += np.dot(A[i][indexBlA:indexBlA + BlockSize], B[indexBlA:indexBlA + BlockSize, j])
    return C


def main():
    for matrix_size in [50, 100, 200, 300]:
        A = np.random.randint(5, size=(matrix_size, matrix_size))
        B = np.random.randint(5, size=(matrix_size, matrix_size))
        data = []
        for blocksize in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300]:
            if matrix_size % blocksize != 0:
                continue
            print(blocksize)
            start_time = time()
            C = matrix_multiplication(A, B, blocksize)
            end_time = time()
            execution_time = end_time - start_time
            data.append({
                'Block size': blocksize,
                'Execution Time': execution_time
            })
        print()
        label = "Matrix Size " + str(matrix_size)
        print(label)
        print(pd.DataFrame(data))
        print("-----------------------")
        data_frame = pd.DataFrame(data)
        plt.plot(data_frame["Block size"], data_frame["Execution Time"], label=label, marker='o')
    plt.xlabel('Block Size')
    plt.ylabel('Execution Time')
    plt.legend(loc='best')
    plt.title('impact of blocking on the performance of matrix multiplication')
    plt.show()


main()



