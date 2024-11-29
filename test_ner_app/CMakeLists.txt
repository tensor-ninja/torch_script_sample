# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(bert_ner_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add these lines for debug configuration
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

# Find required packages
find_package(Torch REQUIRED)
find_package(nlohmann_json REQUIRED)

# Set include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Create library for core functionality
add_library(bert_ner_lib
    src/tokenizer.cpp
    src/ner_model.cpp
)

target_include_directories(bert_ner_lib 
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(bert_ner_lib 
    PUBLIC 
        ${TORCH_LIBRARIES} 
        nlohmann_json::nlohmann_json
)

target_compile_definitions(bert_ner_lib PUBLIC DEBUG)

# Main application
add_executable(ner_demo src/main.cpp)
target_link_libraries(ner_demo PRIVATE bert_ner_lib)

# Testing executable
enable_testing()
add_executable(tokenizer_test tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test PRIVATE bert_ner_lib)
add_test(NAME TokenizerTest COMMAND tokenizer_test)