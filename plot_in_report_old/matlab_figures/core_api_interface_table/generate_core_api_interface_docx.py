#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import html
import zipfile
from datetime import date
from pathlib import Path


BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "core_required_operator_api_table.csv"
OUT_PATH = BASE / "core_operator_api_interface_special_report.docx"


def esc(text):
    return html.escape(str(text), quote=False)


def p(text="", style=None, bold=False):
    if not text:
        return "<w:p/>"
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    bold_xml = "<w:b/>" if bold else ""
    return (
        f"<w:p>{style_xml}<w:r><w:rPr>{bold_xml}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p>"
    )


def bullet(text):
    return (
        '<w:p><w:pPr><w:pStyle w:val="ListParagraph"/>'
        '<w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr>'
        f'<w:r><w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>'
    )


def cell(text, shade=None, bold=False):
    shading = (
        f'<w:tcPr><w:shd w:fill="{shade}"/></w:tcPr>' if shade else "<w:tcPr/>"
    )
    bold_xml = "<w:b/>" if bold else ""
    return (
        f"<w:tc>{shading}<w:p><w:r><w:rPr>{bold_xml}</w:rPr>"
        f"<w:t xml:space=\"preserve\">{esc(text)}</w:t></w:r></w:p></w:tc>"
    )


def table(rows, header=True):
    xml = [
        '<w:tbl><w:tblPr><w:tblStyle w:val="TableGrid"/>'
        '<w:tblW w:w="0" w:type="auto"/><w:tblLook w:val="04A0"/></w:tblPr>'
    ]
    for idx, row in enumerate(rows):
        xml.append("<w:tr>")
        for value in row:
            xml.append(cell(value, shade="D9EAF7" if header and idx == 0 else None, bold=header and idx == 0))
        xml.append("</w:tr>")
    xml.append("</w:tbl>")
    return "".join(xml)


def read_operator_rows():
    with CSV_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def classify_alignment(rows):
    consistent = [r for r in rows if r.get("与 Python cuSignal 对齐情况", "").startswith("一致")]
    semantic = [r for r in rows if r.get("与 Python cuSignal 对齐情况", "").startswith("语义对齐")]
    custom = [r for r in rows if "自定义" in r.get("与 Python cuSignal 对齐情况", "") or "不完全一致" in r.get("与 Python cuSignal 对齐情况", "")]
    return consistent, semantic, custom


def build_document_xml(rows):
    consistent, semantic, custom = classify_alignment(rows)
    today = date.today().isoformat()

    body = []
    body.append(p("核心算子 API 接口专项说明文档", "Title", bold=True))
    body.append(p(f"生成日期：{today}"))
    body.append(p("项目位置：Python 基准 cusignal-23.08.00；CPU 版本 ZKX/cusignal_cpp；GPU 版本 ZKX/cusignal_cpp_20260414。"))

    body.append(p("一、执行计划", "Heading1", bold=True))
    for item in [
        "梳理 matlab_figures/core_api_interface_table 中的核心必做算子接口表，明确任务1、任务2范围。",
        "对照 Python cusignal-23.08.00 的 host 语义、CPU C++ wrapper 和 GPU *_device 公开接口，归纳接口一致性口径。",
        "按接口形态将算子划分为一致、语义对齐、任务自定义三类，说明 C++/CUDA 侧的参数映射、输出方式和 workspace 约束。",
        "生成 Word 专项说明文档，作为后续答辩、验收和接口迁移的说明材料。",
    ]:
        body.append(bullet(item))

    body.append(p("二、文档范围与基准", "Heading1", bold=True))
    body.append(p("本说明聚焦任务1雷达处理链路和任务2多域特征提取链路中的核心必做算子。Python 基准来自 cusignal-23.08.00；CPU 版本以 ZKX/cusignal_cpp 的原函数名和 *_cpu 路径为准；GPU 版本以 ZKX/cusignal_cpp_20260414 中的 *_device、DeviceArray、DeviceComplexArray、params struct 和 workspace 为准。"))
    body.append(p("文档不覆盖所有扩展算子，仅覆盖 core_required_operator_api_table.csv 中列出的 23 个核心条目。"))

    body.append(p("三、总体接口设计原则", "Heading1", bold=True))
    for item in [
        "CPU host 接口优先保持与 Python cuSignal/scipy 的函数名、默认参数、参数语义和返回形态同形。",
        "GPU device 接口显式表达设备内存，不做隐式 host/device 拷贝；输入输出采用 DeviceArray<T> 或 DeviceComplexArray。",
        "GPU 输出由调用方预分配，多返回值拆分为多个 output buffer；复杂算子通过 params struct 和 workspace 复用 FFT plan 与中间缓存。",
        "Python 中的 None、动态返回和 callable 参数，在 C++/CUDA 侧使用哨兵值、重载、variant/pair、字符串或布尔参数表达等价语义。",
        "长 pipeline 推荐边界只拷贝一次，中间阶段全部使用 *_device，最终再统一回传 host。",
    ]:
        body.append(bullet(item))

    body.append(p("四、接口对齐统计", "Heading1", bold=True))
    body.append(table([
        ["统计项", "数量", "说明"],
        ["核心算子条目", str(len(rows)), "来自 core_required_operator_api_table.csv"],
        ["接口基本一致", str(len(consistent)), "host 参数语义和输出形态与 Python 基准基本同形"],
        ["语义对齐", str(len(semantic)), "C++/CUDA 因类型系统、预分配输出或 workspace 改变形态，但数学语义对齐"],
        ["任务自定义/不完全一致", str(len(custom)), "任务特定 kernel 或与 Python API 不完全同形"],
    ]))

    body.append(p("五、核心算子 API 对照总表", "Heading1", bold=True))
    body.append(table([
        ["任务", "模块/环节", "算子函数", "核心输入参数", "输出结果", "对齐情况"]
    ] + [
        [
            r.get("任务", ""),
            r.get("模块/环节", ""),
            r.get("算子函数", ""),
            r.get("核心输入参数", ""),
            r.get("输出结果", ""),
            r.get("与 Python cuSignal 对齐情况", ""),
        ]
        for r in rows
    ]))

    body.append(p("六、逐类接口说明", "Heading1", bold=True))
    body.append(p("1. 接口基本一致类", "Heading2", bold=True))
    body.append(p("此类算子适合优先使用原函数名或 *_cpu 做 Python reference 对照；切换到 GPU 时通常只需要将输入输出改为 device buffer，并由调用方预分配输出。代表算子包括 ca_cfar、cfar_alpha、sawtooth、square、cubic、quadratic、gauss_spline、fm_demod、lombscargle、morlet、ricker 等。"))
    body.append(p("2. 语义对齐类", "Heading2", bold=True))
    body.append(p("此类算子在数学含义、主要参数和输出语义上对齐 Python 基准，但 C++/CUDA 侧为了表达 None、多返回值、矩阵 shape、workspace 或类状态，采用了不同接口形态。代表算子包括 pulse_compression、pulse_doppler、ambgfun、chirp、gausspulse、firwin、firfilter、correlate、cwt、KalmanFilter、argrelextrema。"))
    body.append(p("3. 任务自定义类", "Heading2", bold=True))
    body.append(p("任务1 LFM 脉冲生成使用 generate_lfm_pulse_kernel，对照 chirp/LFM 数学语义，但函数名和参数组织不与 Python cuSignal 同形。该接口应按任务 pipeline 内部生成型 kernel 管理，而不是作为通用 cuSignal API 兼容项验收。"))

    body.append(p("七、Python 到 C++/CUDA 参数映射", "Heading1", bold=True))
    body.append(table([
        ["Python 语义", "C++/CUDA 表达", "说明"],
        ["数组输入 x/y/template", "std::vector<T> 或 DeviceArray<T>/DeviceComplexArray", "CPU 返回 host 容器；GPU 显式 device buffer"],
        ["None 默认值", "空字符串、0、-1、空数组或 params 字段", "用于表达 window=None、nfft=None、fs=None 等动态语义"],
        ["多返回值", "std::pair/variant 或多个预分配输出 buffer", "gausspulse、firfilter、谱分析类函数典型使用"],
        ["axis/shape", "axis 参数 + rows/cols 或 flat row-major buffer", "GPU 侧避免隐式推断矩阵形状"],
        ["workspace", "Hilbert/Stft/Spectrogram/Radar 等 workspace struct", "用于缓存 FFT plan 和 scratch buffer，减少重复分配"],
        ["callable comparator/wavelet", "字符串、布尔值或固定枚举语义", "argrelextrema、cwt 等函数在 device 路径采用受限表达"],
    ]))

    body.append(p("八、GPU Device API 使用约束", "Heading1", bold=True))
    for item in [
        "所有 *_device 均视为显式 GPU API，调用方负责输入输出 buffer 生命周期。",
        "不要在上层 pipeline 中混用 CPU wrapper 和 *_device 造成频繁 to_host/from_host。",
        "FFT 相关路径统一通过 FFTInterface 或对应 workspace 管理，不建议绕过公开接口直接调用后端细节。",
        "固定 shape 或重复调用场景应复用 workspace，特别是 pulse_compression、pulse_doppler、ambgfun、stft、spectrogram、csd、resample_poly 等。",
        "device 输出通常是 flat buffer；二维语义需由调用方根据 rows/cols 和文档约定恢复。",
    ]:
        body.append(bullet(item))

    body.append(p("九、验收与维护建议", "Heading1", bold=True))
    for item in [
        "新增算子时同步更新 CSV/Markdown 对照表、CPU host 说明和 GPU device 调用说明。",
        "对接口基本一致类，优先补齐 Python reference 与 CPU wrapper 的逐元素一致性测试。",
        "对语义对齐类，除数值误差外还应测试 None 哨兵值、多返回值、axis、shape、非法参数和 workspace 复用。",
        "性能验收时应区分单算子 benchmark 与 device-only pipeline benchmark，避免 host/device 拷贝掩盖 GPU 算子真实性能。",
    ]:
        body.append(bullet(item))

    body.append(p("十、来源文件", "Heading1", bold=True))
    for item in [
        "matlab_figures/core_api_interface_table/core_required_operator_api_table.csv",
        "matlab_figures/core_api_interface_table/core_required_operator_api_table.md",
        "cusignal-23.08.00/python/cusignal",
        "ZKX/cusignal_cpp/src",
        "ZKX/cusignal_cpp_20260414/src",
        "ZKX/cusignal_cpp_20260414/docs/GPU_DEVICE_API_GUIDE.md",
        "ZKX/cusignal_cpp_20260414/docs/API_COMPATIBILITY_AND_DEVICE_USAGE.md",
    ]:
        body.append(bullet(item))

    body.append('<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1134" w:right="850" w:bottom="1134" w:left="850" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(body)}</w:body></w:document>"
    )


def styles_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:rPr><w:rFonts w:ascii="Microsoft YaHei" w:eastAsia="Microsoft YaHei" w:hAnsi="Microsoft YaHei"/><w:sz w:val="21"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="28"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="ListParagraph"><w:name w:val="List Paragraph"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="720"/></w:pPr></w:style>
  <w:style w:type="table" w:styleId="TableGrid"><w:name w:val="Table Grid"/><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4" w:color="999999"/><w:left w:val="single" w:sz="4" w:color="999999"/><w:bottom w:val="single" w:sz="4" w:color="999999"/><w:right w:val="single" w:sz="4" w:color="999999"/><w:insideH w:val="single" w:sz="4" w:color="999999"/><w:insideV w:val="single" w:sz="4" w:color="999999"/></w:tblBorders></w:tblPr></w:style>
</w:styles>'''


def numbering_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:numbering xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="720" w:hanging="360"/></w:pPr></w:lvl></w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
</w:numbering>'''


def content_types_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/numbering.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"/>
</Types>'''


def root_rels_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''


def doc_rels_xml():
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/numbering" Target="numbering.xml"/>
</Relationships>'''


def main():
    rows = read_operator_rows()
    with zipfile.ZipFile(OUT_PATH, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types_xml())
        z.writestr("_rels/.rels", root_rels_xml())
        z.writestr("word/_rels/document.xml.rels", doc_rels_xml())
        z.writestr("word/document.xml", build_document_xml(rows))
        z.writestr("word/styles.xml", styles_xml())
        z.writestr("word/numbering.xml", numbering_xml())
    print(OUT_PATH)


if __name__ == "__main__":
    main()
