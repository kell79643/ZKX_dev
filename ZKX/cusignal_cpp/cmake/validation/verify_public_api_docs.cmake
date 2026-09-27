cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PUBLIC_API_SOURCE_ROOT OR PUBLIC_API_SOURCE_ROOT STREQUAL "")
    get_filename_component(PUBLIC_API_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
endif()

# 固定 53 个业务算子的正式五类型 GPU 声明。字段为：
# 算子 ID | 正式 device 符号（不含 _device）| 相对 cusignal_cpp 的公开头文件。
set(PUBLIC_API_PUBLIC_CONTRACTS
    "cubic|cubic|include/cusignal/operators/bsplines/bsplines_typed.h"
    "gauss_spline|gauss_spline|include/cusignal/operators/bsplines/bsplines_typed.h"
    "quadratic|quadratic|include/cusignal/operators/bsplines/bsplines_typed.h"
    "correlate|correlate:2|include/cusignal/operators/convolution/convolution_typed.h"
    "correlate2d|correlate2d|include/cusignal/operators/convolution/convolution_typed.h"
    "fm_demod|fm_demod:3|include/cusignal/operators/demod/demod_typed.h"
    "kalman_filter|kalman_predict,kalman_update|include/cusignal/operators/estimation/estimation_typed.h"
    "firwin|firwin|include/cusignal/operators/filter_design/filter_design_typed.h"
    "firwin2|firwin2|include/cusignal/operators/filter_design/filter_design_typed.h"
    "channelize_poly|channelize_poly|include/cusignal/operators/filtering/filtering_typed.h"
    "detrend|detrend|include/cusignal/operators/filtering/filtering_typed.h"
    "firfilter|firfilter|include/cusignal/operators/filtering/filtering_typed.h"
    "firfilter2|firfilter2|include/cusignal/operators/filtering/filtering_typed.h"
    "freq_shift|freq_shift|include/cusignal/operators/filtering/filtering_typed.h"
    "hilbert|hilbert|include/cusignal/operators/filtering/filtering_typed.h"
    "hilbert2|hilbert2|include/cusignal/operators/filtering/filtering_typed.h"
    "lfilter_zi|lfilter_zi|include/cusignal/operators/filtering/filtering_typed.h"
    "sosfilt|sosfilt|include/cusignal/operators/filtering/filtering_typed.h"
    "wiener|wiener|include/cusignal/operators/filtering/filtering_typed.h"
    "decimate|decimate|include/cusignal/operators/filtering/filtering_typed.h"
    "resample|resample|include/cusignal/operators/filtering/filtering_typed.h"
    "resample_poly|resample_poly|include/cusignal/operators/filtering/filtering_typed.h"
    "upfirdn|upfirdn|include/cusignal/operators/filtering/filtering_typed.h"
    "argrelextrema|argrelextrema:2|include/cusignal/operators/peak_finding/peak_finding_typed.h"
    "ambgfun|ambgfun|include/cusignal/operators/radartools/radartools_typed.h"
    "ca_cfar|ca_cfar|include/cusignal/operators/radartools/radartools_typed.h"
    "cfar_alpha|cfar_alpha|include/cusignal/operators/radartools/radartools_typed.h"
    "pulse_compression|pulse_compression|include/cusignal/operators/radartools/radartools_typed.h"
    "pulse_doppler|pulse_doppler|include/cusignal/operators/radartools/radartools_typed.h"
    "csd|csd:2|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "istft|istft:2|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "lombscargle|lombscargle|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "spectrogram|spectrogram:2|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "stft|stft:2|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "vectorstrength|vectorstrength|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "chirp|chirp:2|include/cusignal/operators/waveforms/waveforms_typed.h"
    "sawtooth|sawtooth|include/cusignal/operators/waveforms/waveforms_typed.h"
    "square|square|include/cusignal/operators/waveforms/waveforms_typed.h"
    "gausspulse|gausspulse:2|include/cusignal/operators/waveforms/waveforms_typed.h"
    "unit_impulse|unit_impulse|include/cusignal/operators/waveforms/waveforms_typed.h"
    "cwt|cwt|include/cusignal/operators/wavelets/wavelets_typed.h"
    "morlet|morlet|include/cusignal/operators/wavelets/wavelets_typed.h"
    "morlet2|morlet2|include/cusignal/operators/wavelets/wavelets_typed.h"
    "qmf|qmf|include/cusignal/operators/wavelets/wavelets_typed.h"
    "ricker|ricker|include/cusignal/operators/wavelets/wavelets_typed.h"
    "chebwin|chebwin|include/cusignal/operators/windows/windows_typed.h"
    "general_cosine|general_cosine|include/cusignal/operators/windows/windows_typed.h"
    "general_gaussian|general_gaussian|include/cusignal/operators/windows/windows_typed.h"
    "hamming|hamming|include/cusignal/operators/windows/windows_typed.h"
    "kaiser|kaiser|include/cusignal/operators/windows/windows_typed.h"
    "parzen|parzen|include/cusignal/operators/windows/windows_typed.h"
    "taylor|taylor|include/cusignal/operators/windows/windows_typed.h"
    "triang|triang|include/cusignal/operators/windows/windows_typed.h"
)

# 同一 53 算子的公开 typed CPU reference。FM demod 有一维/二维两个 overload。
set(PUBLIC_API_CPU_CONTRACTS
    "cubic|cubic_cpu|include/cusignal/operators/bsplines/bsplines_typed.h"
    "gauss_spline|gauss_spline_cpu|include/cusignal/operators/bsplines/bsplines_typed.h"
    "quadratic|quadratic_cpu|include/cusignal/operators/bsplines/bsplines_typed.h"
    "correlate|correlate_typed_cpu|include/cusignal/operators/convolution/convolution_typed.h"
    "correlate2d|correlate2d_typed_cpu|include/cusignal/operators/convolution/convolution_typed.h"
    "fm_demod|fm_demod_typed_cpu:2|include/cusignal/operators/demod/demod_typed.h"
    "kalman_filter|kalman_predict_typed_cpu,kalman_update_typed_cpu|include/cusignal/operators/estimation/estimation_typed.h"
    "firwin|firwin_typed_cpu|include/cusignal/operators/filter_design/filter_design_typed.h"
    "firwin2|firwin2_typed_cpu|include/cusignal/operators/filter_design/filter_design_typed.h"
    "channelize_poly|channelize_poly_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "detrend|detrend_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "firfilter|firfilter_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "firfilter2|firfilter2_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "freq_shift|freq_shift_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "hilbert|hilbert_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "hilbert2|hilbert2_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "lfilter_zi|lfilter_zi_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "sosfilt|sosfilt_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "wiener|wiener_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "decimate|decimate_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "resample|resample_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "resample_poly|resample_poly_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "upfirdn|upfirdn_typed_cpu|include/cusignal/operators/filtering/filtering_typed.h"
    "argrelextrema|argrelextrema_typed_cpu|include/cusignal/operators/peak_finding/peak_finding_typed.h"
    "ambgfun|ambgfun_typed_cpu|include/cusignal/operators/radartools/radartools_typed.h"
    "ca_cfar|ca_cfar_typed_cpu|include/cusignal/operators/radartools/radartools_typed.h"
    "cfar_alpha|cfar_alpha_typed_cpu|include/cusignal/operators/radartools/radartools_typed.h"
    "pulse_compression|pulse_compression_typed_cpu|include/cusignal/operators/radartools/radartools_typed.h"
    "pulse_doppler|pulse_doppler_typed_cpu|include/cusignal/operators/radartools/radartools_typed.h"
    "csd|csd_typed_cpu|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "istft|istft_typed_cpu|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "lombscargle|lombscargle_typed_cpu|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "spectrogram|spectrogram_typed_cpu|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "stft|stft_typed_cpu|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "vectorstrength|vectorstrength_typed_cpu|include/cusignal/operators/spectral_analysis/spectral_analysis_typed.h"
    "chirp|chirp_typed_cpu|include/cusignal/operators/waveforms/waveforms_typed.h"
    "sawtooth|sawtooth_typed_cpu|include/cusignal/operators/waveforms/waveforms_typed.h"
    "square|square_typed_cpu|include/cusignal/operators/waveforms/waveforms_typed.h"
    "gausspulse|gausspulse_typed_cpu|include/cusignal/operators/waveforms/waveforms_typed.h"
    "unit_impulse|unit_impulse_typed_cpu|include/cusignal/operators/waveforms/waveforms_typed.h"
    "cwt|cwt_typed_cpu|include/cusignal/operators/wavelets/wavelets_typed.h"
    "morlet|morlet_typed_cpu|include/cusignal/operators/wavelets/wavelets_typed.h"
    "morlet2|morlet2_typed_cpu|include/cusignal/operators/wavelets/wavelets_typed.h"
    "qmf|qmf_typed_cpu|include/cusignal/operators/wavelets/wavelets_typed.h"
    "ricker|ricker_typed_cpu|include/cusignal/operators/wavelets/wavelets_typed.h"
    "chebwin|chebwin_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "general_cosine|general_cosine_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "general_gaussian|general_gaussian_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "hamming|hamming_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "kaiser|kaiser_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "parzen|parzen_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "taylor|taylor_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
    "triang|triang_typed_cpu|include/cusignal/operators/windows/windows_typed.h"
)

list(LENGTH PUBLIC_API_PUBLIC_CONTRACTS PUBLIC_API_CONTRACT_COUNT)
if(NOT PUBLIC_API_CONTRACT_COUNT EQUAL 53)
    message(FATAL_ERROR "public API documentation public contract registry must contain exactly 53 operators")
endif()
list(LENGTH PUBLIC_API_CPU_CONTRACTS PUBLIC_API_CPU_CONTRACT_COUNT)
if(NOT PUBLIC_API_CPU_CONTRACT_COUNT EQUAL 53)
    message(FATAL_ERROR "public API documentation CPU contract registry must contain exactly 53 operators")
endif()

set(PUBLIC_API_SEEN_OPERATORS)
set(PUBLIC_API_GPU_DECLARATION_COUNT 0)
set(PUBLIC_API_OPERATOR_REGISTRY_PATH
    "${PUBLIC_API_SOURCE_ROOT}/include/cusignal/runtime/operator_type_dispatch.h")
if(NOT EXISTS "${PUBLIC_API_OPERATOR_REGISTRY_PATH}")
    message(FATAL_ERROR "public API documentation cannot find the authoritative 53-operator registry")
endif()
file(READ "${PUBLIC_API_OPERATOR_REGISTRY_PATH}" PUBLIC_API_OPERATOR_REGISTRY_TEXT)

foreach(PUBLIC_API_ENTRY IN LISTS PUBLIC_API_PUBLIC_CONTRACTS)
    string(REPLACE "|" ";" PUBLIC_API_FIELDS "${PUBLIC_API_ENTRY}")
    list(LENGTH PUBLIC_API_FIELDS PUBLIC_API_FIELD_COUNT)
    if(NOT PUBLIC_API_FIELD_COUNT EQUAL 3)
        message(FATAL_ERROR "Malformed public API documentation public contract row: ${PUBLIC_API_ENTRY}")
    endif()
    list(GET PUBLIC_API_FIELDS 0 PUBLIC_API_OPERATOR)
    list(GET PUBLIC_API_FIELDS 1 PUBLIC_API_SYMBOL)
    list(GET PUBLIC_API_FIELDS 2 PUBLIC_API_HEADER)

    list(FIND PUBLIC_API_SEEN_OPERATORS "${PUBLIC_API_OPERATOR}" PUBLIC_API_DUPLICATE_INDEX)
    if(NOT PUBLIC_API_DUPLICATE_INDEX EQUAL -1)
        message(FATAL_ERROR "Duplicate public API documentation operator: ${PUBLIC_API_OPERATOR}")
    endif()
    list(APPEND PUBLIC_API_SEEN_OPERATORS "${PUBLIC_API_OPERATOR}")

    string(FIND
        "${PUBLIC_API_OPERATOR_REGISTRY_TEXT}" "    ${PUBLIC_API_OPERATOR}," PUBLIC_API_REGISTRY_OFFSET)
    if(PUBLIC_API_REGISTRY_OFFSET EQUAL -1)
        message(FATAL_ERROR
            "public API documentation operator is not present in operator_type_dispatch.h: ${PUBLIC_API_OPERATOR}")
    endif()

    set(PUBLIC_API_HEADER_PATH "${PUBLIC_API_SOURCE_ROOT}/${PUBLIC_API_HEADER}")
    if(NOT EXISTS "${PUBLIC_API_HEADER_PATH}")
        message(FATAL_ERROR "public API documentation public header is missing for ${PUBLIC_API_OPERATOR}: ${PUBLIC_API_HEADER}")
    endif()
    file(READ "${PUBLIC_API_HEADER_PATH}" PUBLIC_API_HEADER_TEXT)

    string(REPLACE "," ";" PUBLIC_API_SYMBOL_SPECS "${PUBLIC_API_SYMBOL}")
    foreach(PUBLIC_API_SYMBOL_SPEC IN LISTS PUBLIC_API_SYMBOL_SPECS)
        string(REPLACE ":" ";" PUBLIC_API_SYMBOL_FIELDS "${PUBLIC_API_SYMBOL_SPEC}")
        list(LENGTH PUBLIC_API_SYMBOL_FIELDS PUBLIC_API_SYMBOL_FIELD_COUNT)
        list(GET PUBLIC_API_SYMBOL_FIELDS 0 PUBLIC_API_CURRENT_SYMBOL)
        if(PUBLIC_API_SYMBOL_FIELD_COUNT EQUAL 1)
            set(PUBLIC_API_EXPECTED_OVERLOADS 1)
        elseif(PUBLIC_API_SYMBOL_FIELD_COUNT EQUAL 2)
            list(GET PUBLIC_API_SYMBOL_FIELDS 1 PUBLIC_API_EXPECTED_OVERLOADS)
        else()
            message(FATAL_ERROR "Malformed public API documentation symbol spec: ${PUBLIC_API_SYMBOL_SPEC}")
        endif()

        # 正式入口不一定返回void（例如extrema坐标集合、雷达结果对象和
        # cuSignal规则要求的FP64结果），因此只按公开符号定位，返回类型由头文件契约决定。
        set(PUBLIC_API_DECLARATION "${PUBLIC_API_CURRENT_SYMBOL}_device(")
        set(PUBLIC_API_SEARCH_TEXT "${PUBLIC_API_HEADER_TEXT}")
        set(PUBLIC_API_SEARCH_OFFSET 0)
        foreach(PUBLIC_API_OVERLOAD_INDEX RANGE 1 ${PUBLIC_API_EXPECTED_OVERLOADS})
            math(EXPR PUBLIC_API_GPU_DECLARATION_COUNT "${PUBLIC_API_GPU_DECLARATION_COUNT} + 1")
            string(FIND "${PUBLIC_API_SEARCH_TEXT}" "${PUBLIC_API_DECLARATION}" PUBLIC_API_RELATIVE_OFFSET)
            if(PUBLIC_API_RELATIVE_OFFSET EQUAL -1)
                message(FATAL_ERROR
                    "public API documentation formal declaration ${PUBLIC_API_OVERLOAD_INDEX}/${PUBLIC_API_EXPECTED_OVERLOADS} "
                    "is missing for ${PUBLIC_API_OPERATOR}: ${PUBLIC_API_DECLARATION}")
            endif()
            math(EXPR PUBLIC_API_DECLARATION_OFFSET
                "${PUBLIC_API_SEARCH_OFFSET} + ${PUBLIC_API_RELATIVE_OFFSET}")

            string(SUBSTRING "${PUBLIC_API_HEADER_TEXT}" 0 ${PUBLIC_API_DECLARATION_OFFSET} PUBLIC_API_PREFIX)
            string(FIND "${PUBLIC_API_PREFIX}" "/**" PUBLIC_API_DOC_BEGIN REVERSE)
            string(FIND "${PUBLIC_API_PREFIX}" "*/" PUBLIC_API_DOC_END REVERSE)
            if(PUBLIC_API_DOC_BEGIN EQUAL -1 OR PUBLIC_API_DOC_END LESS PUBLIC_API_DOC_BEGIN)
                message(FATAL_ERROR
                    "public API documentation declaration has no adjacent Doxygen block: ${PUBLIC_API_OPERATOR} in ${PUBLIC_API_HEADER}")
            endif()
            math(EXPR PUBLIC_API_DOC_LENGTH "${PUBLIC_API_DOC_END} + 2 - ${PUBLIC_API_DOC_BEGIN}")
            string(SUBSTRING "${PUBLIC_API_PREFIX}" ${PUBLIC_API_DOC_BEGIN} ${PUBLIC_API_DOC_LENGTH} PUBLIC_API_DOC)

            foreach(PUBLIC_API_REQUIRED_TOKEN
                "@brief" "@param" "@details"
                "FP32" "stream" "workspace" "cuSignal")
                string(FIND "${PUBLIC_API_DOC}" "${PUBLIC_API_REQUIRED_TOKEN}" PUBLIC_API_TOKEN_OFFSET)
                if(PUBLIC_API_TOKEN_OFFSET EQUAL -1)
                    message(FATAL_ERROR
                        "public API documentation ${PUBLIC_API_OPERATOR}/${PUBLIC_API_CURRENT_SYMBOL} documentation misses "
                        "'${PUBLIC_API_REQUIRED_TOKEN}' in ${PUBLIC_API_HEADER}")
                endif()
            endforeach()

            # 只有函数模板声明需要 @tparam。具体 ABI overload（例如直接接受
            # ComplexFloat 的 fm_demod）没有模板参数，强行要求 @tparam 会产生错误文档。
            math(EXPR PUBLIC_API_AFTER_DOC_OFFSET "${PUBLIC_API_DOC_END} + 2")
            math(EXPR PUBLIC_API_AFTER_DOC_LENGTH
                "${PUBLIC_API_DECLARATION_OFFSET} - ${PUBLIC_API_AFTER_DOC_OFFSET}")
            string(SUBSTRING "${PUBLIC_API_HEADER_TEXT}" ${PUBLIC_API_AFTER_DOC_OFFSET}
                ${PUBLIC_API_AFTER_DOC_LENGTH} PUBLIC_API_DECLARATION_PREFIX)
            string(FIND "${PUBLIC_API_DECLARATION_PREFIX}" "template" PUBLIC_API_TEMPLATE_OFFSET)
            if(NOT PUBLIC_API_TEMPLATE_OFFSET EQUAL -1)
                string(FIND "${PUBLIC_API_DOC}" "@tparam" PUBLIC_API_TPARAM_OFFSET)
                if(PUBLIC_API_TPARAM_OFFSET EQUAL -1)
                    message(FATAL_ERROR
                        "public API documentation ${PUBLIC_API_OPERATOR}/${PUBLIC_API_CURRENT_SYMBOL} template documentation "
                        "misses '@tparam' in ${PUBLIC_API_HEADER}")
                endif()
            endif()

            string(FIND "${PUBLIC_API_DOC}" "@throws" PUBLIC_API_THROWS_OFFSET)
            string(FIND "${PUBLIC_API_DOC}" "异常" PUBLIC_API_EXCEPTION_OFFSET)
            if(PUBLIC_API_THROWS_OFFSET EQUAL -1 AND PUBLIC_API_EXCEPTION_OFFSET EQUAL -1)
                message(FATAL_ERROR
                    "public API documentation ${PUBLIC_API_OPERATOR}/${PUBLIC_API_CURRENT_SYMBOL} documentation does not state "
                    "error/exception behavior")
            endif()

            string(LENGTH "${PUBLIC_API_DECLARATION}" PUBLIC_API_DECLARATION_LENGTH)
            math(EXPR PUBLIC_API_ADVANCE "${PUBLIC_API_RELATIVE_OFFSET} + ${PUBLIC_API_DECLARATION_LENGTH}")
            math(EXPR PUBLIC_API_SEARCH_OFFSET "${PUBLIC_API_SEARCH_OFFSET} + ${PUBLIC_API_ADVANCE}")
            string(SUBSTRING "${PUBLIC_API_SEARCH_TEXT}" ${PUBLIC_API_ADVANCE} -1 PUBLIC_API_SEARCH_TEXT)
        endforeach()
    endforeach()
endforeach()

set(PUBLIC_API_CPU_DECLARATION_COUNT 0)
set(PUBLIC_API_SEEN_CPU_OPERATORS)
foreach(PUBLIC_API_ENTRY IN LISTS PUBLIC_API_CPU_CONTRACTS)
    string(REPLACE "|" ";" PUBLIC_API_FIELDS "${PUBLIC_API_ENTRY}")
    list(GET PUBLIC_API_FIELDS 0 PUBLIC_API_OPERATOR)
    list(GET PUBLIC_API_FIELDS 1 PUBLIC_API_SYMBOL)
    list(GET PUBLIC_API_FIELDS 2 PUBLIC_API_HEADER)

    list(FIND PUBLIC_API_SEEN_CPU_OPERATORS "${PUBLIC_API_OPERATOR}" PUBLIC_API_DUPLICATE_INDEX)
    if(NOT PUBLIC_API_DUPLICATE_INDEX EQUAL -1)
        message(FATAL_ERROR "Duplicate public API documentation CPU operator: ${PUBLIC_API_OPERATOR}")
    endif()
    list(APPEND PUBLIC_API_SEEN_CPU_OPERATORS "${PUBLIC_API_OPERATOR}")

    set(PUBLIC_API_HEADER_PATH "${PUBLIC_API_SOURCE_ROOT}/${PUBLIC_API_HEADER}")
    file(READ "${PUBLIC_API_HEADER_PATH}" PUBLIC_API_HEADER_TEXT)
    string(REPLACE "," ";" PUBLIC_API_SYMBOL_SPECS "${PUBLIC_API_SYMBOL}")
    foreach(PUBLIC_API_SYMBOL_SPEC IN LISTS PUBLIC_API_SYMBOL_SPECS)
        string(REPLACE ":" ";" PUBLIC_API_SYMBOL_FIELDS "${PUBLIC_API_SYMBOL_SPEC}")
        list(LENGTH PUBLIC_API_SYMBOL_FIELDS PUBLIC_API_SYMBOL_FIELD_COUNT)
        list(GET PUBLIC_API_SYMBOL_FIELDS 0 PUBLIC_API_CURRENT_SYMBOL)
        if(PUBLIC_API_SYMBOL_FIELD_COUNT EQUAL 1)
            set(PUBLIC_API_EXPECTED_OVERLOADS 1)
        else()
            list(GET PUBLIC_API_SYMBOL_FIELDS 1 PUBLIC_API_EXPECTED_OVERLOADS)
        endif()

        set(PUBLIC_API_DECLARATION "${PUBLIC_API_CURRENT_SYMBOL}(")
        set(PUBLIC_API_SEARCH_TEXT "${PUBLIC_API_HEADER_TEXT}")
        set(PUBLIC_API_SEARCH_OFFSET 0)
        foreach(PUBLIC_API_OVERLOAD_INDEX RANGE 1 ${PUBLIC_API_EXPECTED_OVERLOADS})
            math(EXPR PUBLIC_API_CPU_DECLARATION_COUNT "${PUBLIC_API_CPU_DECLARATION_COUNT} + 1")
            string(FIND "${PUBLIC_API_SEARCH_TEXT}" "${PUBLIC_API_DECLARATION}" PUBLIC_API_RELATIVE_OFFSET)
            if(PUBLIC_API_RELATIVE_OFFSET EQUAL -1)
                message(FATAL_ERROR
                    "public API documentation CPU declaration ${PUBLIC_API_OVERLOAD_INDEX}/${PUBLIC_API_EXPECTED_OVERLOADS} "
                    "is missing for ${PUBLIC_API_OPERATOR}: ${PUBLIC_API_DECLARATION}")
            endif()
            math(EXPR PUBLIC_API_DECLARATION_OFFSET
                "${PUBLIC_API_SEARCH_OFFSET} + ${PUBLIC_API_RELATIVE_OFFSET}")
            string(SUBSTRING "${PUBLIC_API_HEADER_TEXT}" 0 ${PUBLIC_API_DECLARATION_OFFSET} PUBLIC_API_PREFIX)
            string(FIND "${PUBLIC_API_PREFIX}" "/**" PUBLIC_API_DOC_BEGIN REVERSE)
            string(FIND "${PUBLIC_API_PREFIX}" "*/" PUBLIC_API_DOC_END REVERSE)
            if(PUBLIC_API_DOC_BEGIN EQUAL -1 OR PUBLIC_API_DOC_END LESS PUBLIC_API_DOC_BEGIN)
                message(FATAL_ERROR
                    "public API documentation CPU declaration has no adjacent Doxygen block: ${PUBLIC_API_OPERATOR}")
            endif()
            math(EXPR PUBLIC_API_DOC_LENGTH "${PUBLIC_API_DOC_END} + 2 - ${PUBLIC_API_DOC_BEGIN}")
            string(SUBSTRING "${PUBLIC_API_PREFIX}" ${PUBLIC_API_DOC_BEGIN} ${PUBLIC_API_DOC_LENGTH} PUBLIC_API_DOC)

            foreach(PUBLIC_API_REQUIRED_TOKEN
                "@brief" "@tparam" "@param" "@details"
                "FP32" "stream" "workspace" "cuSignal")
                string(FIND "${PUBLIC_API_DOC}" "${PUBLIC_API_REQUIRED_TOKEN}" PUBLIC_API_TOKEN_OFFSET)
                if(PUBLIC_API_TOKEN_OFFSET EQUAL -1)
                    message(FATAL_ERROR
                        "public API documentation CPU ${PUBLIC_API_OPERATOR}/${PUBLIC_API_CURRENT_SYMBOL} documentation misses "
                        "'${PUBLIC_API_REQUIRED_TOKEN}' in ${PUBLIC_API_HEADER}")
                endif()
            endforeach()
            string(FIND "${PUBLIC_API_DOC}" "@throws" PUBLIC_API_THROWS_OFFSET)
            string(FIND "${PUBLIC_API_DOC}" "异常" PUBLIC_API_EXCEPTION_OFFSET)
            if(PUBLIC_API_THROWS_OFFSET EQUAL -1 AND PUBLIC_API_EXCEPTION_OFFSET EQUAL -1)
                message(FATAL_ERROR
                    "public API documentation CPU ${PUBLIC_API_OPERATOR}/${PUBLIC_API_CURRENT_SYMBOL} documentation does not state "
                    "error/exception behavior")
            endif()

            string(LENGTH "${PUBLIC_API_DECLARATION}" PUBLIC_API_DECLARATION_LENGTH)
            math(EXPR PUBLIC_API_ADVANCE "${PUBLIC_API_RELATIVE_OFFSET} + ${PUBLIC_API_DECLARATION_LENGTH}")
            math(EXPR PUBLIC_API_SEARCH_OFFSET "${PUBLIC_API_SEARCH_OFFSET} + ${PUBLIC_API_ADVANCE}")
            string(SUBSTRING "${PUBLIC_API_SEARCH_TEXT}" ${PUBLIC_API_ADVANCE} -1 PUBLIC_API_SEARCH_TEXT)
        endforeach()
    endforeach()
endforeach()

if(NOT PUBLIC_API_GPU_DECLARATION_COUNT EQUAL 64 OR
   NOT PUBLIC_API_CPU_DECLARATION_COUNT EQUAL 55)
    message(FATAL_ERROR
        "public API documentation declaration totals changed: GPU=${PUBLIC_API_GPU_DECLARATION_COUNT}, "
        "CPU=${PUBLIC_API_CPU_DECLARATION_COUNT}")
endif()

math(EXPR PUBLIC_API_DOCUMENTED_COUNT
    "${PUBLIC_API_GPU_DECLARATION_COUNT} + ${PUBLIC_API_CPU_DECLARATION_COUNT}")
message(STATUS
    "[PUBLIC_API][SUMMARY] operators=${PUBLIC_API_CONTRACT_COUNT} "
    "gpu_declarations=${PUBLIC_API_GPU_DECLARATION_COUNT} "
    "cpu_declarations=${PUBLIC_API_CPU_DECLARATION_COUNT} "
    "documented=${PUBLIC_API_DOCUMENTED_COUNT} missing=0")
