#ifndef MATRIXMULTIPLICATION_H
#define MATRIXMULTIPLICATION_H

#include <memory>
#include "./MatrixContext.h"

using namespace std;

namespace MatrixMultiplication {
    void recursive(int** matrixA, int** matrixB, int** matrixC,
                  int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd, MatrixContext &context);

    void iterative(int** matrixA, int** matrixB, int** matrixC, int matrixARows, int matrixBRows, int matrixBCols, int numThreads,
                   shared_ptr<WsServer::Connection> connection, MatrixContext &context);

    void partiallyIterative(int** matrixA, int** matrixB, int** matrixC, int rows, int cols, int numThreads,
                            shared_ptr<WsServer::Connection> connection, MatrixContext &context);

    int parDot(int** matrixA, int i, int** matrixB, int j, int length, int writeThreadNum, int numThreads, MatrixContext &context);

    int** allocateMatrix(int rows, int cols);
    void freeMatrix(int** matrix, int rows);
    void setMatrixTo0(int** matrix, int rows, int cols);
}

#endif
