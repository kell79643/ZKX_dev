#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include "memory.h"
#include "sha256.h"
#include <cusignal/operators/wavelets/wavelets_typed.h>

#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef ZKX_OPERATOR_CASE_WAVELET_OPERATOR
#error "ZKX_OPERATOR_CASE_WAVELET_OPERATOR is required"
#endif

namespace {
namespace fs = std::filesystem;
constexpr const char* kOperator = ZKX_OPERATOR_CASE_WAVELET_OPERATOR;
constexpr double kPi = 3.14159265358979323846;

struct Cli { std::string dtype; int n; double width; std::size_t warmup, measured; fs::path output; };
struct Phase { std::string name; int index; zkx::common::ResourceSample heap, rss, gpu; };
struct Evidence {
    std::vector<double> cpu, gpu, cpu_ms, gpu_ms;
    std::vector<Phase> cpu_trace, gpu_trace;
    std::string manifest, typed_array_digest, input_digest, cpu_digest, gpu_digest;
    std::string output_dtype{"FP64"}, signal_preview, widths_preview;
    int width_count{0};
};

Cli parse_cli(int argc, char** argv) {
    std::map<std::string,std::string> values;
    for (int i=1;i<argc;i+=2) { if(i+1>=argc) throw std::invalid_argument("missing CLI value"); values.emplace(std::string(argv[i]).substr(2),argv[i+1]); }
    auto get=[&](const char* key){auto it=values.find(key);if(it==values.end())throw std::invalid_argument(std::string("missing --")+key);return it->second;};
    return {get("dtype"),std::stoi(get("n")),std::stod(get("width")),std::stoull(get("warmup")),std::stoull(get("measured")),get("output")};
}
zkx::common::ResourceSample gpu_sample(){return zkx::common::sample_gpu_used([]{auto info=cusignal::cuda_utils::current_memory_info();return static_cast<std::int64_t>(info.total_bytes-info.free_bytes);},"cudaMemGetInfo.total_minus_free");}
Phase phase(const char* name,int index,bool gpu){return{name,index,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU phase"}};}

template<class T>T convert(double value){return static_cast<T>(value);}template<>__half convert<__half>(double value){return __float2half_rn(static_cast<float>(value));}
template<class T>double numeric(T value){return static_cast<double>(value);}template<>double numeric<__half>(__half value){return __half2float(value);}
template<class T>std::string digest(const std::vector<T>& values){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(values.data()),values.size()*sizeof(T)));}
template<class T>std::string scalar_digest(const T& value){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(&value),sizeof(T)));}

template<class T>std::vector<T> signal_input(int n){std::vector<T> values(static_cast<std::size_t>(n));for(int i=0;i<n;++i){double value=4.0*std::sin(2.0*kPi*i/32.0)+2.0*std::cos(2.0*kPi*i/11.0)+(i==n/3?8.0:0.0);if constexpr(std::is_integral_v<T>)value=std::round(value);values[static_cast<std::size_t>(i)]=convert<T>(value);}return values;}
template<class T>std::vector<T> width_input(double base){return{convert<T>(base),convert<T>(base*2),convert<T>(base*4),convert<T>(base*6)};}

template<class T>std::string preview(const std::vector<T>& values){std::vector<std::size_t> idx;if(values.size()<=6){for(std::size_t i=0;i<values.size();++i)idx.push_back(i);}else idx={0,1,2,3,values.size()-2,values.size()-1};std::ostringstream out;out<<std::setprecision(9);for(std::size_t j=0;j<idx.size();++j){if(j)out<<';';out<<"i="<<idx[j]<<':'<<numeric(values[idx[j]]);}return out.str();}
template<class F>std::vector<double> measure(std::size_t warm,std::size_t measured,F function){for(std::size_t i=0;i<warm;++i)function();std::vector<double> samples;samples.reserve(measured);for(std::size_t i=0;i<measured;++i){auto begin=std::chrono::steady_clock::now();function();auto end=std::chrono::steady_clock::now();samples.push_back(std::chrono::duration<double,std::milli>(end-begin).count());}return samples;}

template<class T>Evidence run(const Cli& cli){
    if(cli.n<1)throw std::invalid_argument("n must be positive");
    Evidence evidence;const std::string op=kOperator;T width=convert<T>(cli.width);
    const cusignal::CwtRealWaveletCallable wavelet=[](int points,int a){return cusignal::ricker_typed_cpu<float>(points,static_cast<float>(a));};
    std::vector<T> signal,widths;if(op=="cwt"){signal=signal_input<T>(cli.n);widths=width_input<T>(cli.width);evidence.width_count=static_cast<int>(widths.size());evidence.signal_preview=preview(signal);evidence.widths_preview=preview(widths);evidence.typed_array_digest=digest(signal);evidence.manifest="signal:"+cli.dtype+":bytes="+std::to_string(signal.size()*sizeof(T))+":sha256="+digest(signal)+"|widths:"+cli.dtype+":bytes="+std::to_string(widths.size()*sizeof(T))+":sha256="+digest(widths)+"|wavelet=ricker_typed_cpu_float";}else if(op=="ricker"){evidence.width_count=1;evidence.widths_preview="i=0:"+std::to_string(numeric(width));evidence.typed_array_digest=scalar_digest(width);evidence.manifest="width:"+cli.dtype+":bytes="+std::to_string(sizeof(T))+":sha256="+evidence.typed_array_digest+"|n="+std::to_string(cli.n);}else throw std::invalid_argument("unsupported compiled operator");
    evidence.input_digest=zkx::remaining_operator_case::sha256_hex(evidence.manifest);
    std::vector<double> cpu_result,gpu_result;
    auto cpu_call=[&]{cpu_result=op=="cwt"?cusignal::cwt_typed_cpu(signal,widths,wavelet):cusignal::ricker_typed_cpu(cli.n,width);};
    auto gpu_call=[&]{if(op=="cwt"){auto workspace=cusignal::prepare_cwt_workspace(cli.n,widths,wavelet);auto device_signal=cusignal::DeviceArray<T>::from_host(signal);cusignal::DeviceArray<double> output(workspace.output_size());cusignal::cwt_device(device_signal,workspace,output);gpu_result=output.to_host();}else{cusignal::DeviceArray<double> output(static_cast<std::size_t>(cli.n));cusignal::ricker_device(cli.n,width,output);gpu_result=output.to_host();}cusignal::cuda_utils::synchronize_stream();};
    evidence.cpu_ms=measure(cli.warmup,cli.measured,cpu_call);evidence.gpu_ms=measure(cli.warmup,cli.measured,gpu_call);

    evidence.cpu_trace.push_back(phase("before",0,false));std::vector<T> cpu_signal=signal,cpu_widths=widths;evidence.cpu_trace.push_back(phase("allocate",1,false));evidence.cpu_trace.push_back(phase("h2d",2,false));cpu_result=op=="cwt"?cusignal::cwt_typed_cpu(cpu_signal,cpu_widths,wavelet):cusignal::ricker_typed_cpu(cli.n,width);evidence.cpu_trace.push_back(phase("execute_sync",3,false));evidence.cpu=cpu_result;evidence.cpu_digest=digest(cpu_result);evidence.cpu_trace.push_back(phase("d2h",4,false));cpu_signal.clear();cpu_widths.clear();cpu_result.clear();evidence.cpu_trace.push_back(phase("release_sync",5,false));evidence.cpu_trace.push_back(phase("after",6,false));

    evidence.gpu_trace.push_back(phase("before",0,true));cusignal::DeviceArray<T> device_signal;cusignal::DeviceArray<double> output;cusignal::CwtDeviceWorkspace workspace;if(op=="cwt")output=cusignal::DeviceArray<double>(static_cast<std::size_t>(cli.n)*widths.size());else output=cusignal::DeviceArray<double>(static_cast<std::size_t>(cli.n));evidence.gpu_trace.push_back(phase("allocate",1,true));if(op=="cwt"){device_signal=cusignal::DeviceArray<T>::from_host(signal);workspace=cusignal::prepare_cwt_workspace(cli.n,widths,wavelet);}evidence.gpu_trace.push_back(phase("h2d",2,true));if(op=="cwt")cusignal::cwt_device(device_signal,workspace,output);else cusignal::ricker_device(cli.n,width,output);cusignal::cuda_utils::synchronize_stream();evidence.gpu_trace.push_back(phase("execute_sync",3,true));gpu_result=output.to_host();evidence.gpu=gpu_result;evidence.gpu_digest=digest(gpu_result);evidence.gpu_trace.push_back(phase("d2h",4,true));device_signal={};output={};workspace={};gpu_result.clear();cusignal::cuda_utils::synchronize_stream();evidence.gpu_trace.push_back(phase("release_sync",5,true));evidence.gpu_trace.push_back(phase("after",6,true));return evidence;
}

void emit_samples(std::ostream& out,const std::vector<double>& values){out<<'[';for(std::size_t i=0;i<values.size();++i){if(i)out<<',';out<<std::setprecision(17)<<values[i];}out<<']';}
void emit_resource(std::ostream& out,const zkx::common::ResourceSample& sample){out<<"{\"available\":"<<(sample.available?"true":"false")<<",\"bytes\":"<<(sample.available?std::to_string(sample.bytes):"null")<<'}';}
void emit_trace(std::ostream& out,const std::vector<Phase>& trace){out<<'[';for(std::size_t i=0;i<trace.size();++i){if(i)out<<',';out<<"{\"phase\":\""<<trace[i].name<<"\",\"phase_index\":"<<trace[i].index<<",\"heap\":";emit_resource(out,trace[i].heap);out<<",\"rss\":";emit_resource(out,trace[i].rss);out<<",\"gpu\":";emit_resource(out,trace[i].gpu);out<<'}';}out<<']';}
int dispatch(const Cli& cli,Evidence& evidence){if(cli.dtype=="FP32")evidence=run<float>(cli);else if(cli.dtype=="FP16")evidence=run<__half>(cli);else if(cli.dtype=="INT32")evidence=run<std::int32_t>(cli);else if(cli.dtype=="INT16")evidence=run<std::int16_t>(cli);else if(cli.dtype=="INT8")evidence=run<std::int8_t>(cli);else return 2;return 0;}
}

int main(int argc,char** argv){try{const Cli cli=parse_cli(argc,argv);Evidence evidence;if(dispatch(cli,evidence)!=0)return 2;const auto accuracy=zkx::common::compare(evidence.cpu,evidence.gpu,1e-12);if(fs::exists(cli.output))throw std::runtime_error("output exists");std::ofstream out(cli.output);out<<"{\"operator\":\""<<kOperator<<"\",\"requested_dtype\":\""<<cli.dtype<<"\",\"output_dtype\":\""<<evidence.output_dtype<<"\",\"backend\":\"not_applicable\",\"n\":"<<cli.n<<",\"width\":"<<cli.width<<",\"width_count\":"<<evidence.width_count<<",\"output_count\":"<<evidence.cpu.size()<<",\"typed_input_manifest\":\""<<evidence.manifest<<"\",\"typed_array_digest\":\""<<evidence.typed_array_digest<<"\",\"actual_input_digest\":\""<<evidence.input_digest<<"\",\"signal_preview_heacuda_api_tail2\":\""<<evidence.signal_preview<<"\",\"widths_preview_heacuda_api_tail2\":\""<<evidence.widths_preview<<"\",\"cpu_output_digest\":\""<<evidence.cpu_digest<<"\",\"gpu_output_digest\":\""<<evidence.gpu_digest<<"\",\"mse\":"<<accuracy.mse<<",\"rmse\":"<<accuracy.rmse<<",\"relative_l2\":"<<accuracy.relative_l2<<",\"relative_linf\":"<<accuracy.relative_linf<<",\"cpu_samples_ms\":";emit_samples(out,evidence.cpu_ms);out<<",\"gpu_samples_ms\":";emit_samples(out,evidence.gpu_ms);out<<",\"cpu_trace\":";emit_trace(out,evidence.cpu_trace);out<<",\"gpu_trace\":";emit_trace(out,evidence.gpu_trace);out<<"}\n";std::cout<<"OPERATOR_CASE_WAVELET_PROBE PASS operator="<<kOperator<<" dtype="<<cli.dtype<<'\n';return 0;}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}}
