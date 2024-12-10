#ifndef MATRIXCONTEXT_H
#define MATRIXCONTEXT_H

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <vector>
#include <server_ws.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;
using WsServer = SimpleWeb::SocketServer<SimpleWeb::WS>;

class MatrixContext {
public:
    double velocityMultiplier = 1;
    atomic<bool> is_execution_paused{false};
    atomic<bool> terminate_threads{false};
    atomic<bool> processing{false};

    mutex matrix_mutex;
    condition_variable cv;
    queue<json> update_queue;
    mutex queue_mutex;

    vector<thread> active_threads;
    mutex thread_vector_mutex;

    void matrixWatcher(shared_ptr<WsServer::Connection> connection) {
        while (!terminate_threads) {
            unique_lock<mutex> lock(matrix_mutex);
            cv.wait(lock, [&] { return !update_queue.empty() || terminate_threads; });
            
            if (terminate_threads) return;

            while (!update_queue.empty()) {
                json update;
                {
                    lock_guard<mutex> queue_lock(queue_mutex);
                    update = update_queue.front();
                    update_queue.pop();
                }

                // cout << "Sending update: " << update.dump() << endl;
                connection->send(update.dump(), [](const SimpleWeb::error_code &ec) {
                    if (ec) {
                        cout << "Send failed: " << ec.message() << endl;
                    }
                });
            }
        }
    }

    void queueJsonSend(json json) {
        {
            lock_guard<mutex> queue_lock(queue_mutex);
            update_queue.push(json);
        }
        cv.notify_all();
    }

    void terminateActiveThreads() {
        terminate_threads = true;
        cv.notify_all();
        {
            lock_guard<mutex> lock(thread_vector_mutex);
            for (auto &t : active_threads) {
                if (t.joinable()) t.join();
            }
            active_threads.clear();
        }
        terminate_threads = false;
    }
};

#endif
