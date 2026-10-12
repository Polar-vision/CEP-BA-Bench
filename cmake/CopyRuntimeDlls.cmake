foreach(runtime_dll IN LISTS DLLS)
    get_filename_component(runtime_dll_name "${runtime_dll}" NAME)
    file(COPY_FILE "${runtime_dll}" "${OUTPUT_DIR}/${runtime_dll_name}"
        ONLY_IF_DIFFERENT)
endforeach()
