#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generate a judge-facing Word document for the cusignal C++/CUDA API."""

import csv
import html
import zipfile
from datetime import date
from pathlib import Path


BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "core_required_operator_api_table.csv"
ACCURACY_PATH = Path("/workspace/ZKX/cusignal_cpp_20260414/docs/CPU_GPU_ACCURACY_COVERAGE_MATRIX.md")
OUT_PATH = BASE / "算子API接口专项说明文档.docx"

TASK1_OPERATORS = {
    "pulse_compression",
    "pulse_doppler",
    "ca_cfar",
    "cfar_alpha",
    "ambgfun",
}

TASK2_OPERATORS = {
    "chirp",
    "gausspulse",
    "sawtooth",
    "square",
    "firwin",
    "firfilter",
    "cubic",
    "quadratic",
    "gauss_spline",
    "fm_demod",
    "correlate",
    "lombscargle",
    "cwt",
    "morlet",
    "ricker",
    "kalmanfilter",
    "argrelextrema",
}


def e(text):
    return html.escape(str(text), quote=False)


def para(text="", style=None, bold=False, mono=False):
    if text == "":
        return "<w:p/>"
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    rpr = []
    if bold:
        rpr.append("<w:b/>")
    if mono:
        rpr.append('<w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:eastAsia="Microsoft YaHei"/>')
    rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
    return f'<w:p>{ppr}<w:r>{rpr_xml}<w:t xml:space="preserve">{e(text)}</w:t></w:r></w:p>'


def bullet(text):
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{e(text)}</w:t></w:r></w:p>'
    )


def code_block(lines):
    out = []
    for line in lines.strip("\n").splitlines():
        out.append(para(line, "Code", mono=True))
    return "".join(out)


def cell(text, header=False):
    shade = '<w:shd w:fill="D9EAF7"/>' if header else ""
    bold = "<w:b/>" if header else ""
    return (
        f"<w:tc><w:tcPr>{shade}</w:tcPr><w:p><w:r><w:rPr>{bold}</w:rPr>"
        f'<w:t xml:space="preserve">{e(text)}</w:t></w:r></w:p></w:tc>'
    )


def table(rows):
    xml = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/><w:tblLook w:val="04A0"/></w:tblPr>'
    ]
    for i, row in enumerate(rows):
        xml.append("<w:tr>")
        for value in row:
            xml.append(cell(value, header=(i == 0)))
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def read_rows():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def canonical_operator(name):
    name = name.replace("`", "").strip()
    if name == "KalmanFilter":
        return "kalmanfilter"
    return name.lower()


def read_all_functions():
    rows = []
    for line in ACCURACY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or "`" not in line or "Host Yes" not in line:
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 7:
            continue
        rows.append({
            "Module": parts[0],
            "Function": parts[1].replace("`", ""),
            "HostPython": parts[2],
            "Device": parts[3],
            "LayerD": parts[4],
            "FFT": parts[5],
            "Note": parts[6],
        })
    return rows


def task_group(function_name):
    key = canonical_operator(function_name)
    if key in TASK1_OPERATORS:
        return "任务一核心算子"
    if key in TASK2_OPERATORS:
        return "任务二核心算子"
    return "其他公开算子"


def task_rows(core_rows, group_name):
    wanted = TASK1_OPERATORS if group_name == "任务一核心算子" else TASK2_OPERATORS
    out = []
    for r in core_rows:
        first = r["Operator"].split("/")[0].strip()
        if canonical_operator(first) in wanted:
            out.append(r)
    return out


def operator_namespace(op):
    name = op.split("/")[0].strip()
    if name in {"pulse_compression", "pulse_doppler", "ca_cfar", "cfar_alpha", "ambgfun"}:
        return "radartools"
    if name in {"chirp", "gausspulse", "sawtooth", "square"}:
        return "waveforms"
    if name in {"firwin"}:
        return "filter_design"
    if name in {"firfilter"}:
        return "filtering"
    if name in {"cubic", "quadratic", "gauss_spline"}:
        return "bsplines"
    if name in {"fm_demod"}:
        return "demod"
    if name in {"correlate"}:
        return "convolution"
    if name in {"lombscargle"}:
        return "spectral_analysis"
    if name in {"cwt", "morlet", "ricker"}:
        return "wavelets"
    if name in {"KalmanFilter"}:
        return "estimation"
    if name in {"argrelextrema"}:
        return "peak_finding"
    return "task1_custom"


def build_doc(rows):
    all_functions = read_all_functions()
    b = []
    b.append(para("核心算子 API 接口说明文档（评委版）", "Title", True))
    b.append(para(f"生成日期：{date.today().isoformat()}"))
    b.append(para("参照 CuPy 14.0.1 文档组织方式：先说明项目和运行方式，再按模块列出 API Reference。"))

    b.append(para("1. 文档目的", "Heading1", True))
    b.append(para("本文面向评委，说明本项目如何从 Python cuSignal 迁移到 C++ CPU 与 C++/CUDA GPU 两套实现。评委阅读后应能明确：项目目录在哪里、CPU 版本和 GPU 版本分别是什么、如何编译运行、如何调用核心算子、哪些接口与 Python cuSignal 对齐。"))

    b.append(para("2. 项目结构总览", "Heading1", True))
    b.append(table([
        ["项目", "路径", "定位", "评委关注点"],
        ["Python 基准", "cusignal-23.08.00", "RAPIDS cuSignal 23.08.00 源码，作为接口和数值参考", "用于说明我们对齐的 Python API 来源"],
        ["CPU 版本", "ZKX/cusignal_cpp", "C++ CPU 迁移实现，原函数名与 *_cpu 入口运行在 CPU", "用于验证不依赖 GPU 的 host 行为"],
        ["GPU 版本", "ZKX/cusignal_cpp_20260414", "C++/CUDA GPU 迁移实现，*_device 为显式 GPU API", "用于展示 GPU 加速、device pipeline 与性能测试"],
        ["接口表与本文档", "matlab_figures/core_api_interface_table", "核心必做算子 API 对照表与评委版说明文档", "用于答辩和验收材料"],
    ]))

    b.append(para("3. 主入口头文件与目录结构说明", "Heading1", True))
    b.append(para("两个 C++ 项目都提供统一主入口头文件，评委或使用者只需包含该文件即可访问项目公开的信号处理 API。"))
    b.append(table([
        ["项目", "主入口头文件", "说明"],
        ["CPU 版本", "ZKX/cusignal_cpp/src/signal_processing.h", "包含 fft_interface、bsplines、convolution、demod、estimation、filter_design、filtering、peak_finding、radartools、spectral_analysis、waveforms、wavelets、windows 等模块头文件。"],
        ["GPU 版本", "ZKX/cusignal_cpp_20260414/src/signal_processing.h", "包含 CPU wrapper、*_cpu、*_device、FFTInterface 和 DeviceArray/DeviceComplexArray 相关公开接口；显式 GPU 调用仍使用 *_device。"],
    ]))
    b.append(para("最小包含方式："))
    b.append(code_block("""
#include "signal_processing.h"
using namespace cusignal;
"""))
    b.append(para("两个 C++ 项目采用相同的模块化目录，模块命名参考 CuPy/cupyx.scipy.signal 的 API reference 分类。"))
    b.append(table([
        ["目录", "作用", "典型 API"],
        ["src/bsplines", "B 样条和平滑函数", "cubic, quadratic, gauss_spline"],
        ["src/convolution", "相关和卷积类接口", "correlate"],
        ["src/demod", "解调接口", "fm_demod"],
        ["src/estimation", "状态估计接口", "KalmanFilter"],
        ["src/filter_design", "滤波器设计", "firwin"],
        ["src/filtering", "滤波应用与重采样", "firfilter"],
        ["src/peak_finding", "峰值/极值检测", "argrelextrema"],
        ["src/radartools", "雷达处理工具链", "pulse_compression, pulse_doppler, ca_cfar, ambgfun"],
        ["src/spectral_analysis", "谱分析", "lombscargle"],
        ["src/waveforms", "波形生成", "chirp, gausspulse, sawtooth, square"],
        ["src/wavelets", "小波接口", "cwt, morlet, ricker"],
        ["src/windows", "窗口函数", "hamming, kaiser, parzen, triang 等扩展接口"],
        ["src/fft_interface", "FFT 抽象层，可切换 Thrust/cuFFT 后端", "FFTInterface"],
        ["src/cuda_utils", "GPU 内存、拷贝、错误检查、shape 工具", "DeviceArray, DeviceComplexArray"],
        ["test/signal_processing", "Python reference 生成与结果对比脚本", "generate_reference.py, compare_results.py"],
        ["test/performance", "GPU 性能测试入口", "benchmark_cpp_gpu_performance.cpp"],
    ]))

    b.append(para("4. CPU 与 GPU 入口如何区分", "Heading1", True))
    b.append(table([
        ["入口形态", "在哪个项目使用", "含义", "返回/输出方式"],
        ["原函数名，例如 chirp、firwin、ca_cfar", "CPU 和 GPU 项目均可见", "host wrapper，语义尽量对齐 Python cuSignal；当前 GPU 项目中原函数名也复查为 CPU wrapper", "返回 std::vector、std::pair、variant 或类对象"],
        ["*_cpu，例如 chirp_cpu、cubic_cpu", "显式 CPU 实现", "用于评委或测试明确指定 CPU 路径", "返回 host 容器"],
        ["*_device，例如 chirp_device、ca_cfar_device", "GPU 项目主线", "显式 CUDA device API；输入输出在 GPU 内存中", "调用方预分配 DeviceArray/DeviceComplexArray 输出"],
        ["params + workspace，例如 pulse_doppler_device", "复杂 GPU 算子", "用于固定 shape、FFT plan 和 scratch buffer 复用", "输出写入预分配 device buffer"],
    ]))
    for item in [
        "CPU 路径用于和 Python reference 做接口与数值对齐。",
        "GPU 路径用于高性能 pipeline；不要每个算子之后都拷回 host。",
        "Python 的 None、多返回值和 callable，在 C++/CUDA 中分别用哨兵值、多个输出 buffer、字符串或布尔选项表达。",
    ]:
        b.append(bullet(item))

    b.append(para("5. 环境与编译运行", "Heading1", True))
    b.append(para("以下命令按当前工作区实际路径修正为 /workspace/ZKX/...。当前会话中已实际验证 GPU benchmark 目标可构建并可运行，因此本节把它作为当前项目可直接运行的主入口。"))
    b.append(para("环境判断："))
    b.append(code_block("nvcc --version"))
    b.append(table([
        ["环境", "用途", "项目路径", "FFT 后端", "说明"],
        ["CUDA 11.7 / 端口 9001", "日常开发、构建、测试", "/workspace/ZKX/cusignal_cpp 或 /workspace/ZKX/cusignal_cpp_20260414", "Thrust，USE_CUFFT=OFF", "当前工作区可用 nvcc 11.7；cmake 命令未出现在 PATH 中，但已有 GPU build/Makefile 可用"],
        ["CUDA 12.4 / 端口 9041", "Python cuSignal reference、cuFFT 验证", "/workspace/ZKX/cusignal_cpp_20260414", "cuFFT，USE_CUFFT=ON", "用于生成 Python 参考结果和 cuFFT 对比"],
    ]))
    b.append(para("当前工作区已验证可运行：GPU 全量公开 API benchmark。该程序会调用 CPU wrapper、*_cpu、*_device、pipeline 和其他公开算子，并输出耗时与 checksum。"))
    b.append(code_block("""
cd /workspace/ZKX/cusignal_cpp_20260414
make -C build benchmark_cpp_gpu_performance -j4
./build/benchmark_cpp_gpu_performance /tmp/PERFORMANCE_BASELINE_CPP_GPU_CHECK.md
"""))
    b.append(para("如需从零重建 GPU/CUDA 项目，可在提供 cmake 的目标环境中执行以下命令；当前会话未提供 cmake，因此当前已验证路径使用上面的已有 build/Makefile。"))
    b.append(code_block("""
cd /workspace/ZKX/cusignal_cpp_20260414
cmake -S . -B build -DUSE_CUFFT=OFF
cmake --build build --target benchmark_cpp_gpu_performance -j4
./build/benchmark_cpp_gpu_performance docs/performance/PERFORMANCE_BASELINE_CPP_GPU.md
"""))
    b.append(para("Python reference 生成与结果对比（需要 conda 环境和有效测试输出）："))
    b.append(code_block("""
source /home/zkx/miniconda3/etc/profile.d/conda.sh
conda activate /home/zkx/miniconda3/envs/cusignal_env
cd /workspace/ZKX/cusignal_cpp_20260414/test/signal_processing
python generate_reference.py
python compare_results.py
"""))

    b.append(para("6. 最小调用示例", "Heading1", True))
    b.append(para("以下示例已按当前头文件真实签名修正，并已在当前工作区完整链接为可执行文件运行通过，输出为 firwin=129, chirp_device=4。重点是：只包含 signal_processing.h；CPU 使用原函数名；GPU 使用 DeviceArray 和 *_device。"))
    b.append(para("示例源文件 /tmp/cusignal_api_usage_example.cu："))
    b.append(code_block("""
#include <iostream>
#include <vector>
#include <variant>
#include "signal_processing.h"

int main() {
using namespace cusignal;

std::vector<double> t = {0.0, 0.01, 0.02, 0.03};
auto y = chirp(t, 10.0, 1.0, 50.0, "linear", 0.0, true, "real");
auto win = firwin(129, std::vector<double>{0.2});

DeviceArray<double> t_dev = DeviceArray<double>::from_host(t);
DeviceArray<double> y_dev(t_dev.size());
chirp_device(t_dev, y_dev, 10.0, 1.0, 50.0, "linear", 0.0, true);
auto y_host = y_dev.to_host();

std::cout << "firwin=" << win.size()
          << ", chirp_device=" << y_host.size() << std::endl;
return (win.size() == 129 && y_host.size() == t.size()) ? 0 : 1;
}
"""))
    b.append(para("当前已有 build 对象下的完整链接与运行命令："))
    b.append(code_block("""
cd /workspace
nvcc -std=c++17 \
  -I/workspace/ZKX/cusignal_cpp_20260414/src \
  /tmp/cusignal_api_usage_example.cu \
  /workspace/ZKX/cusignal_cpp_20260414/build/CMakeFiles/fft_core.dir/src/fft_interface/fft_thrust.cu.o \
  $(find /workspace/ZKX/cusignal_cpp_20260414/build/CMakeFiles/signal_lib.dir -name '*.o' | sort) \
  -lcudart \
  -o /tmp/cusignal_api_usage_example
/tmp/cusignal_api_usage_example
"""))
    b.append(para("GPU pipeline 调用原则："))
    for item in [
        "输入只在 pipeline 开始时 from_host 一次。",
        "中间步骤持续使用 DeviceArray/DeviceComplexArray。",
        "需要 FFT 或 scratch 的算子复用 workspace。",
        "最终结果只在 pipeline 末尾 to_host 一次。",
    ]:
        b.append(bullet(item))

    b.append(para("7. API Reference（核心任务算子与其他算子分组）", "Heading1", True))
    b.append(para("本节覆盖 GPU 项目当前 53 个公开函数。为方便评委核对竞赛任务，先列任务一、任务二直接使用的核心算子；其余函数列为其他公开算子，表示项目已实现但不是本次任务主流程的必需接口。"))

    b.append(para("7.1 任务一核心算子", "Heading1", True))
    b.append(para("任务一雷达目标检测与多普勒分析主流程使用以下 5 个核心算子。"))
    api_rows = [["模块", "算子", "核心参数", "输出", "CPU/GPU 说明"]]
    for r in task_rows(rows, "任务一核心算子"):
        api_rows.append([
            operator_namespace(r["Operator"]),
            r["Operator"],
            r["CoreInputs"],
            r["Output"],
            r["Alignment"],
        ])
    b.append(table(api_rows))

    b.append(para("7.2 任务二核心算子", "Heading1", True))
    b.append(para("任务二多域特征提取主流程使用以下 17 个核心算子：chirp、gausspulse、sawtooth、square、firwin、firfilter、cubic、quadratic、gauss_spline、fm_demod、correlate、lombscargle、cwt、morlet、ricker、kalmanfilter（源码类名 KalmanFilter）、argrelextrema。"))
    api_rows = [["模块", "算子", "核心参数", "输出", "CPU/GPU 说明"]]
    for r in task_rows(rows, "任务二核心算子"):
        api_rows.append([
            operator_namespace(r["Operator"]),
            r["Operator"],
            r["CoreInputs"],
            r["Output"],
            r["Alignment"],
        ])
    b.append(table(api_rows))

    b.append(para("7.3 全部公开算子覆盖矩阵（53 个）", "Heading1", True))
    b.append(para("下表为 GPU 项目 53 个公开 API 的可用性覆盖表。CPU/Host 表示原函数名或 *_cpu 的 host 路径已有 reference 对比记录；GPU Device 表示对应 *_device 路径已被 benchmark 实际调用并计时。表中不再单列 FFT 路径，涉及 FFT 或特殊路径的函数在备注中说明。"))
    full_rows = [["分组", "模块", "函数", "CPU/Host", "GPU Device", "备注"]]
    for r in all_functions:
        note = r["Note"]
        if r["FFT"] in {"Yes", "Partial"}:
            fft_note = "FFT-based" if r["FFT"] == "Yes" else "部分路径 FFT-based"
            note = f"{fft_note}; {note}".strip("; ")
        full_rows.append([
            task_group(r["Function"]),
            r["Module"],
            r["Function"],
            r["HostPython"],
            r["Device"],
            note,
        ])
    b.append(table(full_rows))

    b.append(para("7.4 其他公开算子说明", "Heading1", True))
    b.append(para("其他公开算子不是任务一/任务二主流程必需项，但属于完整 cuSignal C++/CUDA 迁移能力的一部分。它们覆盖二维相关、谱分析、重采样、Hilbert 变换、窗口函数和附加小波/波形工具，可用于扩展实验、性能评测和后续应用。"))

    b.append(para("8. 评委查看建议", "Heading1", True))
    b.append(table([
        ["评审问题", "查看位置", "说明"],
        ["项目是不是有 Python 基准？", "cusignal-23.08.00", "作为接口和参考输出来源"],
        ["CPU 版本在哪里？", "ZKX/cusignal_cpp", "原函数名和 *_cpu 是 CPU 路径"],
        ["GPU 版本在哪里？", "ZKX/cusignal_cpp_20260414", "*_device 是 CUDA device API"],
        ["如何查看正确性对比？", "docs/CPU_GPU_ACCURACY_COVERAGE_MATRIX.md；test/signal_processing/compare_results.py", "查看已记录的 host/reference 对比与可用性覆盖；compare_results.py 需配合有效测试输出使用"],
        ["如何运行性能测试？", "test/performance/benchmark_cpp_gpu_performance.cpp", "写入 docs/performance/PERFORMANCE_BASELINE_CPP_GPU.md"],
        ["API 如何对齐？", "本节 API Reference；core_required_operator_api_table.csv", "按核心必做算子逐项说明"],
    ]))

    b.append(para("9. 与 CuPy 文档的对应关系", "Heading1", True))
    for item in [
        "CuPy Basic 文档先讲 ndarray 位于 CPU/GPU 哪一侧；本文对应说明 std::vector、DeviceArray 和 DeviceComplexArray。",
        "CuPy Reference 按 signal、windows 等模块列 autosummary；本文按 bsplines、waveforms、radartools 等模块列核心 API。",
        "CuPy 强调 host-device 数据传输；本文强调 from_host/to_host 只在 pipeline 边界发生。",
    ]:
        b.append(bullet(item))

    b.append(para("10. 当前验收状态", "Heading1", True))
    for item in [
        "GPU 项目 README 记录：53 个对外函数均已有 CPU 与 GPU 版本。",
        "2026-05-04 文档记录：host/wrapper vs Python reference 为 235/235 PASS。",
        "C++ Device benchmark 覆盖 53/53；性能结论以 docs/performance 下的基线和对比文档为准。",
    ]:
        b.append(bullet(item))

    b.append('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="850" w:right="720" w:bottom="850" w:left="720" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(b)}</w:body></w:document>"
    )


def styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="20"/></w:rPr><w:pPr><w:spacing w:after="80"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="27"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
  <w:style w:type="paragraph" w:styleId="Code"><w:name w:val="Code"/><w:basedOn w:val="Normal"/><w:rPr><w:rFonts w:ascii="Consolas" w:hAnsi="Consolas"/><w:sz w:val="18"/></w:rPr><w:pPr><w:spacing w:after="0"/><w:shd w:fill="F2F2F2"/></w:pPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="999999"/><w:left w:val="single" w:sz="4" w:color="999999"/><w:bottom w:val="single" w:sz="4" w:color="999999"/><w:right w:val="single" w:sz="4" w:color="999999"/><w:insideH w:val="single" w:sz="4" w:color="999999"/><w:insideV w:val="single" w:sz="4" w:color="999999"/></w:tblBorders></w:tblPr></w:style>
</w:styles>"""


def numbering_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>"""


def write_docx():
    rows = read_rows()
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/></Types>""")
        z.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>""")
        z.writestr("word/_rels/document.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/></Relationships>""")
        z.writestr("word/document.xml", build_doc(rows))
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", numbering_xml())
    print(OUT_PATH)


if __name__ == "__main__":
    write_docx()
