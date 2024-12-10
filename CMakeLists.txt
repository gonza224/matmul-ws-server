cmake_minimum_required(VERSION 3.14)
project(MatmulServer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable FetchContent
include(FetchContent)

# Fetch Dependencies

FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.11.0
)
FetchContent_MakeAvailable(spdlog)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
    Simple-WebSocket-Server
    GIT_REPOSITORY https://gitlab.com/eidheim/Simple-WebSocket-Server.git
    GIT_TAG master
)
FetchContent_MakeAvailable(Simple-WebSocket-Server)

# Find System Packages

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Boost (Required by Simple-WebSocket-Server)
find_package(Boost 1.74.0 REQUIRED COMPONENTS system thread)

# ------------------------
# Add Executable
# ------------------------

add_executable(MatmulServer MatmulServer.cpp MatrixMultiplication.cpp)

# Include Directories

# Include Simple-WebSocket-Server headers
target_include_directories(MatmulServer PRIVATE
    ${Simple-WebSocket-Server_SOURCE_DIR}/
)

# Include nlohmann/json headers
target_include_directories(MatmulServer PRIVATE
    ${nlohmann_json_SOURCE_DIR}/include
)

# ------------------------
# Link Libraries
# ------------------------

target_link_libraries(MatmulServer
    PRIVATE
        spdlog::spdlog
        nlohmann_json::nlohmann_json
        OpenSSL::SSL
        OpenSSL::Crypto
        OpenMP::OpenMP_CXX
        Boost::system
        Boost::thread
)
