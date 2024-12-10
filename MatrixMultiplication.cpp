#include "MatrixMultiplication.h"
#include <omp.h>
#include <thread>
#include <chrono>
#include <cassert>

using namespace std;

const int BASE_DELAY_TIME = 5000;

namespace MatrixMultiplication {

   void recursive(int** matrixA, int** matrixB, int** matrixC,
                  int iStart, int iEnd, int jStart, int jEnd, int kStart, int kEnd, MatrixContext &context)
    {
        if (iStart == iEnd && jStart == jEnd && kStart == kEnd) {
            while (context.is_execution_paused && !context.terminate_threads) {
                this_thread::sleep_for(chrono::milliseconds(100));
            }

            if (context.terminate_threads) {
                return;
            }

            int threadNum = omp_get_thread_num();
            json update = {
                {"threadInfo", {{"number", threadNum}}},
                {"actions", json::array({
                    {
                        {"action", "write"},
                        {"matrix", "C"},
                        {"indices", {{"i", iStart}, {"j", jStart}}},
                        {"value", matrixC[iStart][jStart]}
                    }
                })}
            };

            #pragma omp critical
            context.queueJsonSend(update);

            json readUpdate = {
                {"threadInfo", {{"number", threadNum}}},
                {"actions", json::array({
                    {
                        {"action", "read"},
                        {"matrix", "A"},
                        {"indices", {{"i", iStart}, {"j", kStart}}},
                        {"writeCell", to_string(iStart) + "-" + to_string(jStart)}
                    },
                    {
                        {"action", "read"},
                        {"matrix", "B"},
                        {"indices", {{"i", kStart}, {"j", jStart}}},
                        {"writeCell", to_string(iStart) + "-" + to_string(jStart)}
                    }
                })}
            };

            #pragma omp critical
            context.queueJsonSend(readUpdate);

            matrixC[iStart][jStart] += matrixA[iStart][kStart] * matrixB[kStart][jStart];

            json setValueUpdate = {
                {"threadInfo", {{"number", threadNum}}},
                {"actions", json::array({
                    {
                        {"action", "setValue"},
                        {"matrix", "C"},
                        {"indices", {{"i", iStart}, {"j", jStart}}},
                        {"value", matrixC[iStart][jStart]}
                    }
                })}
            };

            #pragma omp critical
            context.queueJsonSend(setValueUpdate);

            int delay = BASE_DELAY_TIME * context.velocityMultiplier;
            this_thread::sleep_for(chrono::milliseconds(delay));
            return;
        }

        int midI = (iStart + iEnd) / 2;
        int midJ = (jStart + jEnd) / 2;
        int midK = (kStart + kEnd) / 2;

        if (iStart > iEnd || jStart > jEnd || kStart > kEnd) return;

        // Create tasks for subdividing the problem
        #pragma omp task shared(context)
        recursive(matrixA, matrixB, matrixC, iStart, midI, jStart, midJ, kStart, midK, context);

        #pragma omp task shared(context)
        recursive(matrixA, matrixB, matrixC, iStart, midI, midJ + 1, jEnd, kStart, midK, context);

        #pragma omp task shared(context)
        recursive(matrixA, matrixB, matrixC, midI + 1, iEnd, jStart, midJ, kStart, midK, context);

        recursive(matrixA, matrixB, matrixC, midI + 1, iEnd, midJ + 1, jEnd, kStart, midK, context);

        #pragma omp taskwait

        #pragma omp task shared(context)
        recursive(matrixA, matrixB, matrixC, iStart, midI, jStart, midJ, midK + 1, kEnd, context);

        #pragma omp task shared(context)
        recursive(matrixA, matrixB, matrixC, iStart, midI, midJ + 1, jEnd, midK + 1, kEnd, context);

        #pragma omp task shared(context)
        recursive(matrixA, matrixB, matrixC, midI + 1, iEnd, jStart, midJ, midK + 1, kEnd, context);

        recursive(matrixA, matrixB, matrixC, midI + 1, iEnd, midJ + 1, jEnd, midK + 1, kEnd, context);

        #pragma omp taskwait
    }

    void iterative(int** matrixA, int** matrixB, int** matrixC, int matrixARows, int matrixBRows, int matrixBCols, int numThreads,
                   shared_ptr<WsServer::Connection> connection, MatrixContext &context) {
        #pragma omp parallel for collapse(3) num_threads(numThreads)
        for (int i = 0; i < matrixARows; i++) {
            for (int j = 0; j < matrixBCols; j++) {
                for (int k = 0; k < matrixBRows; k++) {
                    while (context.is_execution_paused && !context.terminate_threads) {
                        this_thread::sleep_for(chrono::milliseconds(100));
                    }

                    if (context.terminate_threads) {
                        #pragma omp cancel for
                        continue;
                    }

                    int threadNum = omp_get_thread_num();
                    json update = {
                        {"threadInfo", {{"number", threadNum}}},
                        {"actions", json::array({
                            {
                                {"action", "write"},
                                {"matrix", "C"},
                                {"indices", {{"i", i}, {"j", j}}},
                                {"value", matrixC[i][j]}
                            }
                        })}
                    };

                    #pragma omp critical
                    context.queueJsonSend(update);

                    json readUpdate = {
                        {"threadInfo", {{"number", threadNum}}},
                        {"actions", json::array({
                            {
                                {"action", "read"},
                                {"matrix", "A"},
                                {"indices", {{"i", i}, {"j", k}}},
                                {"writeCell", to_string(i)+"-"+to_string(j)}
                            },
                            {
                                {"action", "read"},
                                {"matrix", "B"},
                                {"indices", {{"i", k}, {"j", j}}},
                                {"writeCell", to_string(i)+"-"+to_string(j)}
                            }
                        })}
                    };

                    #pragma omp critical
                    context.queueJsonSend(readUpdate);

                    #pragma omp critical
                    matrixC[i][j] += matrixA[i][k] * matrixB[k][j];

                    json setValueUpdate = {
                        {"threadInfo", {{"number", threadNum}}},
                        {"actions", json::array({
                            {
                                {"action", "setValue"},
                                {"matrix", "C"},
                                {"indices", {{"i", i}, {"j", j}}},
                                {"value", matrixC[i][j]}
                            }
                        })}
                    };

                    #pragma omp critical
                    context.queueJsonSend(setValueUpdate);

                    int delay = BASE_DELAY_TIME * context.velocityMultiplier;
                    this_thread::sleep_for(chrono::milliseconds(delay));
                }
            }
        }
        context.processing = false;
        freeMatrix(matrixA, matrixARows);
        freeMatrix(matrixB, matrixBRows);
        freeMatrix(matrixC, matrixARows);

        json msg;
        msg["state"] = "STAND_BY";
        connection->send(msg.dump(), [](const SimpleWeb::error_code &ec) {
            if (ec) {
                cout << "Send failed: " << ec.message() << endl;
            }
        });
    }

    int parDot(int** matrixA, int i, int** matrixB, int j, int length, int writeThreadNum, int numThreads, MatrixContext &context) {
        int sum = 0;
        #pragma omp parallel for num_threads(numThreads) reduction(+:sum)
        for (int k = 0; k < length; k++) {
            int threadNum = omp_get_thread_num();
            json update = {
                {"threadInfo", {{"number", threadNum}}},
                {"actions", json::array({
                    {"action", "read"},
                    {"matrix", "A"},
                    {"indices", {{"i", i}, {"j", k}}}
                })}
            };

            #pragma omp critical
            context.queueJsonSend(update);

            sum += matrixA[i][k] * matrixB[k][j];
        }
        return sum;
    }

    void partiallyIterative(int** matrixA, int** matrixB, int** matrixC, int rows, int cols, int numThreads,
                        shared_ptr<WsServer::Connection> connection, MatrixContext &context) {
        #pragma omp parallel for collapse(2) num_threads(numThreads) 
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                // #pragma omp flush(context.is_execution_paused)
                if (context.is_execution_paused) cout << "Execution pause check, waiting..." << endl;
                while (context.is_execution_paused) {
                    this_thread::sleep_for(chrono::milliseconds(100));
                }

                // #pragma omp flush(context.terminate_threads)
                if (context.terminate_threads) {
                    cout << "Terminating threads early" << endl;
                    #pragma omp cancel for
                    continue;
                }

                int threadNum = omp_get_thread_num();

                json update = {
                    {"threadInfo", {{"number", threadNum}}},
                    {"actions", json::array({
                        {
                            {"action", "write"},
                            {"matrix", "C"},
                            {"indices", {{"i", i}, {"j", j}}},
                            {"value", matrixC[i][j]}
                        }
                    })},
                };

                #pragma omp critical
                {
                    context.queueJsonSend(update);
                }
                // matrixC[i][j] = parDot(matrixA, i, matrixB, j, 0, rows, threadNum);
                matrixC[i][j] = parDot(matrixA, i, matrixB, j, rows, threadNum, numThreads, context);

                json setValueUpdate = {
                    {"threadInfo", {{"number", threadNum}}},
                    {"actions", json::array({
                        {
                            {"action", "setValue"},
                            {"matrix", "C"},
                            {"indices", {{"i", i}, {"j", j}}},
                            {"value", matrixC[i][j]}
                        }
                    })},
                };

                #pragma omp critical
                {
                    context.queueJsonSend(setValueUpdate);
                }

                if (i != rows-1 || j != rows-1) {
                    // #pragma omp flush(context.velocityMultiplier)
                    int delay = 10000 * context.velocityMultiplier;
                    this_thread::sleep_for(chrono::milliseconds(delay));
                }
            }
        }

        context.processing = false;
        freeMatrix(matrixA, rows);
        freeMatrix(matrixB, rows);
        freeMatrix(matrixC, rows);

        json msg;
        msg["state"] = "STAND_BY";
        context.queueJsonSend(msg);
    }

    int** allocateMatrix(int rows, int cols) {
        int** matrix = new int*[rows];
        for (int i = 0; i < rows; i++) {
            matrix[i] = new int[cols];
        }
        return matrix;
    }

    void freeMatrix(int** matrix, int rows) {
        for (int i = 0; i < rows; i++) {
            delete[] matrix[i];
        }
        delete[] matrix;
    }

    void setMatrixTo0(int** matrix, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = 0;
            }
        }
    }
}
