add_library(llm_rpc_inference STATIC model.cpp kv_cache.cpp tokenizer.cpp snapshot.cpp)
target_include_directories(llm_rpc_inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(llm_rpc_inference PRIVATE -Wall -Wextra)

add_executable(llm_rpc_generate generate.cpp)
target_link_libraries(llm_rpc_generate PRIVATE llm_rpc_inference)
set_target_properties(llm_rpc_generate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

protobuf_generate_cpp(INFERENCE_PROTO_SOURCES INFERENCE_PROTO_HEADERS inference_rpc.proto)
add_library(llm_rpc_inference_protocol STATIC ${INFERENCE_PROTO_SOURCES} ${INFERENCE_PROTO_HEADERS})
target_include_directories(llm_rpc_inference_protocol PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(llm_rpc_inference_protocol PUBLIC protobuf)

add_executable(llm_rpc_inference_worker inference_worker.cpp)
add_executable(llm_rpc_inference_client inference_coordinator.cpp)
set_target_properties(llm_rpc_inference_worker llm_rpc_inference_client PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
target_link_libraries(llm_rpc_inference_worker PRIVATE llm_rpc_inference llm_rpc_inference_protocol llm_rpc_core ${LIBS})
target_link_libraries(llm_rpc_inference_client PRIVATE llm_rpc_inference_protocol llm_rpc_core ${LIBS})
