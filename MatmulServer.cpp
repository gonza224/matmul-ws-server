#include "./MatrixContext.h"
#include "./MatrixMultiplication.h"
#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <functional>
#include <boost/asio.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using namespace std;
using namespace MatrixMultiplication;

std::string pointerToString(void* ptr) {
    std::ostringstream oss;
    oss << ptr;
    return oss.str();
}

using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;

int main() {
    try {
        auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("../server.log", true);

        std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
        auto logger = std::make_shared<spdlog::logger>("server_logger", sinks.begin(), sinks.end());

        spdlog::register_logger(logger);
        spdlog::set_default_logger(logger);

        spdlog::set_level(spdlog::level::info);
        spdlog::flush_on(spdlog::level::info);
    } catch (const spdlog::spdlog_ex& ex) {
        std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        return 1;
    }

    WsServer server;
    server.config.port = 3000;
    server.config.address = "0.0.0.0";
    auto& matrix_endpoint = server.endpoint["^/matmul/?$"];

    unordered_map<void*, shared_ptr<MatrixContext>> connection_contexts;
    mutex context_map_mutex;

    matrix_endpoint.on_message = [&](shared_ptr<WsServer::Connection> connection,
                                     shared_ptr<WsServer::InMessage> in_message) {
        shared_ptr<MatrixContext> context;
        void* key = static_cast<void*>(connection.get());
        {
            lock_guard<mutex> lock(context_map_mutex);
            if (connection_contexts.find(key) == connection_contexts.end()) {
                connection_contexts[key] = make_shared<MatrixContext>();
                spdlog::info("Created new MatrixContext for connection {}", pointerToString(key));
            }
            context = connection_contexts[key];
        }

        try {
            auto message = in_message->string();
            spdlog::info("Received message: {}", message);

            auto data = json::parse(message);

            if (data.contains("action")) {
                string action = data["action"];
                if (action == "STOP") {
                    connection->send("{\"error\": \"Cancelling execution.\"}");
                    context->terminateActiveThreads();
                    spdlog::info("STOP action received. Terminated active threads for connection {}", pointerToString(key));
                    return;
                } else if (action == "PAUSE") {
                    context->is_execution_paused = true;
                    connection->send("{\"error\": \"Pausing execution.\"}");
                    spdlog::info("PAUSE action received. Execution paused for connection {}", pointerToString(key));
                    return;
                } else if (action == "UNPAUSE") {
                    context->is_execution_paused = false;
                    connection->send("{\"error\": \"Unpausing execution.\"}");
                    spdlog::info("UNPAUSE action received. Execution resumed for connection {}", pointerToString(key));
                    return;
                }
            }

            if (data.contains("velocityMultiplier") && !data["velocityMultiplier"].is_null()) {
                try {
                    if (data["velocityMultiplier"].is_number()) { 
                        context->velocityMultiplier = data["velocityMultiplier"].get<double>();
                        spdlog::info("Set velocityMultiplier to {} for connection {}", context->velocityMultiplier, pointerToString(key));
                    }
                } catch (const exception& e) {
                    spdlog::error("Error parsing velocityMultiplier: {} for connection {}", e.what(), pointerToString(key));
                }
                return;
            }

            if (context->processing) {
                connection->send("{\"error\": \"Matrix multiplication in progress. Please wait.\"}");
                spdlog::warn("Matrix multiplication already in progress for connection {}", pointerToString(key));
                return;
            }

            spdlog::info("Processing matrix multiplication for connection {}", pointerToString(key));

            context->terminateActiveThreads();
            context->processing = true;
            
            int matrixARows = data["matrixARows"];
            int matrixACols = data["matrixACols"];
                        
            int matrixBRows = data["matrixBRows"];
            int matrixBCols = data["matrixBCols"];

            int numThreads = data["numThreads"];
            int algorithm = data["algorithm"];

            int** matrixA = allocateMatrix(matrixARows, matrixACols);
            int** matrixB = allocateMatrix(matrixBRows, matrixBCols);
            int** matrixC = allocateMatrix(matrixARows, matrixBCols);

            for (int i = 0; i < matrixARows; i++) {
                for (int j = 0; j < matrixACols; j++) {
                    matrixA[i][j] = data["matrix1"][i][j];
                }
            }

            for (int i = 0; i < matrixBRows; i++) {
                for (int j = 0; j < matrixBCols; j++) {
                    matrixB[i][j] = data["matrix2"][i][j];
                }
            }

            if (algorithm == 0) {
                context->active_threads.emplace_back(iterative, matrixA, matrixB, matrixC, matrixARows, matrixBRows, matrixBCols,
                                                    numThreads, connection, ref(*context));
                spdlog::info("Started iterative matrix multiplication for connection {}", pointerToString(key));
            } else if (algorithm == 1) {
                context->active_threads.emplace_back([=, &context]() {
                    spdlog::info("Starting recursive matrix multiplication for connection {}", pointerToString(key));
                    #pragma omp parallel num_threads(numThreads)
                    {
                        #pragma omp single
                        {
                            recursive(matrixA, matrixB, matrixC, 0, matrixARows - 1, 0, matrixBCols - 1, 0, matrixACols - 1, *context);
                        }
                    }
                    context->processing = false;
                    freeMatrix(matrixA);
                    freeMatrix(matrixB);
                    freeMatrix(matrixC);

                    json msg;
                    msg["state"] = "STAND_BY";
                    connection->send(msg.dump(), [](const SimpleWeb::error_code &ec) {
                        if (ec) {
                            spdlog::error("Send failed: {}", ec.message());
                        }
                    });
                    spdlog::info("Completed recursive matrix multiplication for connection {}", pointerToString(key));
                });
            }

            context->active_threads.emplace_back(
                bind(&MatrixContext::matrixWatcher, context.get(), connection)
            );
            spdlog::info("Added matrixWatcher thread for connection {}", pointerToString(key));
        } catch (exception& e) {
            connection->send("{\"error\": \"Invalid matrix data\"}");
            spdlog::error("Exception caught while processing message: {} for connection {}", e.what(), pointerToString(key));
        }
    };

    matrix_endpoint.on_open = [&](shared_ptr<WsServer::Connection> connection) {
        auto context = make_shared<MatrixContext>();
        void* key = static_cast<void*>(connection.get());
        {
            lock_guard<mutex> lock(context_map_mutex);
            connection_contexts[key] = context;
        }
        context->velocityMultiplier = 1;
        spdlog::info("Connection opened with {}", pointerToString(key));
    };

    matrix_endpoint.on_close = [&](shared_ptr<WsServer::Connection> connection, int status, const string& reason) {
        void* key = static_cast<void*>(connection.get());
        {
            lock_guard<mutex> lock(context_map_mutex);
            auto it = connection_contexts.find(key);
            if (it != connection_contexts.end()) {
                it->second->terminateActiveThreads();
                connection_contexts.erase(it);
                spdlog::info("Cleaned up MatrixContext for connection {}", pointerToString(key));
            }
        }
        spdlog::info("Connection closed with {} with status code {}", pointerToString(key), status);
    };

    matrix_endpoint.on_error = [&](shared_ptr<WsServer::Connection> connection, const SimpleWeb::error_code& ec) {
        void* key = static_cast<void*>(connection.get());
        {
            lock_guard<mutex> lock(context_map_mutex);
            auto it = connection_contexts.find(key);
            if (it != connection_contexts.end()) {
                it->second->terminateActiveThreads();
                connection_contexts.erase(it);
                spdlog::info("Cleaned up MatrixContext for connection {} due to error", pointerToString(key));
            }
        }
        spdlog::error("Error in connection {}: {}", pointerToString(key), ec.message());
    };

    spdlog::info("Starting WebSocket server on port {}", server.config.port);
    server.start();
    return 0;
}
