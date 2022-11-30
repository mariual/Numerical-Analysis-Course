# Maryam Alipour | 9612037

import math
import scipy
import numpy as np
from scipy.linalg import lu, inv


class Algebra:
    def __init__(self):
        self.matrix = Algebra.get_matrix()
        self.vector = Algebra.get_vector()
        

    @staticmethod
    def get_matrix():
        n = int(input("Enter the number of matrix rows: "))
        matrix = list()
        for i in range(n): 
            tmp = []
            for j in range(n):
                tmp.append(int(input("Enter the element: ")))
            matrix.append(tmp)
        print("Your Matrix:")
        for i in range(n):
            for j in range(n):
                print(matrix[i][j], end=" ")
            print()
        return matrix


    @staticmethod
    def get_vector():
        n = int(input("Enter the number of vector elements: "))
        vector = list()
        for i in range(n):
            vector.append(int(input("Enter the element: ")))
        return vector


    def is_positive(self):
        result = np.all(np.linalg.eigvals(self.matrix) > 0)
        return result


    def cholesky_decomposition(self):
        n = len(self.matrix)
        lower = [[0 for x in range(n + 1)]
                for y in range(n + 1)]
        for i in range(n):
            for j in range(i + 1):
                sum1 = 0
                if j == i:
                    for k in range(j):
                        sum1 += pow(lower[j][k], 2)
                    lower[j][j] = int(math.sqrt(self.matrix[j][j] - sum1))
                else:
                    for k in range(j):
                        sum1 += (lower[i][k] * lower[j][k])
                    if lower[j][j] > 0:
                        lower[i][j] = int((self.matrix[i][j] - sum1) / lower[j][j])
        print('Lower Triangular\t\tTranspose')
        for i in range(n):
            for j in range(n):
                print(lower[i][j], end="\t")
            print("", end="\t")
            for j in range(n):
                print(lower[j][i], end="\t")
            print("")
        x = np.linalg.solve(np.array(self.matrix), np.array(self.vector))
        print('\nX = ' + str(x))

    
    def gaussian_elimination(self):
        mat = self.matrix
        n = len(mat)
        lower = [[0 for x in range(n)]
                for y in range(n)]
        upper = [[0 for x in range(n)]
                for y in range(n)]
    
        for i in range(n):
            for k in range(i, n):
                sum = 0
                for j in range(i):
                    sum += (lower[i][j] * upper[j][k])
                upper[i][k] = mat[i][k] - sum

            for k in range(i, n):
                if (i == k):
                    lower[i][i] = 1  
                else:
                    sum = 0
                    for j in range(i):
                        sum += (lower[k][j] * upper[j][i])

                    lower[k][i] = int((mat[k][i] - sum) /
                                    upper[i][i])

        print("Lower Triangular\t\tUpper Triangular")
        for i in range(n):
            for j in range(n):
                print(lower[i][j], end="\t")
            print("", end="\t")
            for j in range(n):
                print(upper[i][j], end="\t")
            print("")
        x = np.linalg.solve(np.array(self.matrix), np.array(self.vector))
        print('\nX = ' + str(x))


if __name__ == '__main__':
    obj = Algebra()

    print('___________________________________________________')

    if obj.is_positive():
        print('\nA) Matrix is definite positive.')
        print('\nB, C) Cholesky Decomposition and solve AX=b')
        obj.cholesky_decomposition()
    else:
        print('\nA) Matrix is not definite positive.')
        print('\nB, C) Gaussian Elimination and solve AX=b')
        obj.gaussian_elimination()
