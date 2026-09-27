#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/filtering/filtering_typed.h>
#include "memory.h"
#include "sha256.h"
#include <cuda_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace fs = std::filesystem;
struct Cli { std::string dtype; std::size_t count{}, warmup{}, measured{}; double freq{}, sample_rate{}; std::uint64_t seed{}; fs::path output; };
struct Phase { std::string name; int index{}; zkx::common::ResourceSample heap, rss, gpu; };

Cli cli(int argc, char** argv) {
    std::map<std::string,std::string> values;
    for (int i=1;i<argc;i+=2) {
        if (i+1>=argc || std::string(argv[i]).rfind("--",0)) throw std::invalid_argument("args");
        values.emplace(std::string(argv[i]).substr(2),argv[i+1]);
    }
    auto get=[&](const char* key)->const std::string& { auto it=values.find(key); if(it==values.end()) throw std::invalid_argument(key); return it->second; };
    Cli c{get("dtype"),std::stoull(get("count")),std::stoull(get("warmup")),std::stoull(get("measured")),std::stod(get("freq")),std::stod(get("fs")),std::stoull(get("seed")),get("output")};
    if(values.size()!=8 || !c.count || !c.warmup || !c.measured || !std::isfinite(c.freq) || !std::isfinite(c.sample_rate) || c.sample_rate==0.0) throw std::invalid_argument("invalid");
    return c;
}
zkx::common::ResourceSample gpu_sample(){return zkx::common::sample_gpu_used([]{auto i=cusignal::cuda_utils::current_memory_info();return static_cast<std::int64_t>(i.total_bytes-i.free_bytes);},"cudaMemGetInfo.total_minus_free");}
Phase phase(const char* n,int i,bool gpu){return{n,i,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU"}};}
template<class T>T cast(float x){return static_cast<T>(x);} template<> __half cast<__half>(float x){return __float2half_rn(x);}
template<class T>std::vector<T> make(std::size_t n,std::uint64_t seed){std::vector<T> out;out.reserve(n);for(std::size_t i=0;i<n;++i){int noise=static_cast<int>((i*17+seed)%11)-5;float value=noise*.25f+static_cast<float>(i%97)*.03125f;out.push_back(cast<T>(std::is_integral_v<T>?static_cast<float>(static_cast<int>(value)):value));}return out;}
template<class T>std::string digest(const std::vector<T>& v){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)));}
std::vector<double> flatten(const std::vector<ComplexDouble>& v){std::vector<double> out;out.reserve(v.size()*2);for(const auto& z:v){out.push_back(z.re);out.push_back(z.im);}return out;}
template<class F>std::vector<double> measure(std::size_t w,std::size_t n,F&& f){for(std::size_t i=0;i<w;++i)f();std::vector<double> samples;for(std::size_t i=0;i<n;++i){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();samples.push_back(std::chrono::duration<double,std::milli>(b-a).count());}return samples;}
void samples(std::ostream& o,const std::vector<double>& v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<std::setprecision(17)<<v[i];}o<<']';}
void resource(std::ostream& o,const zkx::common::ResourceSample& s){o<<"{\"available\":"<<(s.available?"true":"false")<<",\"bytes\":"<<(s.available?std::to_string(s.bytes):"null")<<'}';}
void phases(std::ostream& o,const std::vector<Phase>& v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<"{\"phase\":\""<<v[i].name<<"\",\"phase_index\":"<<v[i].index<<",\"heap\":";resource(o,v[i].heap);o<<",\"rss\":";resource(o,v[i].rss);o<<",\"gpu\":";resource(o,v[i].gpu);o<<'}';}o<<']';}

template<class T>int run(const Cli& c){
    auto x=make<T>(c.count,c.seed);std::vector<ComplexDouble> cpu_output,gpu_output;
    auto cpu=measure(c.warmup,c.measured,[&]{cpu_output=cusignal::freq_shift_typed_cpu(x,c.freq,c.sample_rate);});
    auto gpu=measure(c.warmup,c.measured,[&]{auto dx=cusignal::DeviceArray<T>::from_host(x);cusignal::DeviceArray<ComplexDouble> out(x.size());cusignal::freq_shift_device(dx,out,c.freq,c.sample_rate);gpu_output=out.to_host();});
    std::vector<Phase> cpu_trace{phase("before",0,false),phase("allocate",1,false),phase("h2d",2,false)};auto cpu_trace_output=cusignal::freq_shift_typed_cpu(x,c.freq,c.sample_rate);cpu_trace.push_back(phase("execute_sync",3,false));cpu_trace.push_back(phase("d2h",4,false));cpu_trace_output.clear();cpu_trace_output.shrink_to_fit();cpu_trace.push_back(phase("release_sync",5,false));cpu_trace.push_back(phase("after",6,false));
    std::vector<Phase> gpu_trace;cusignal::cuda_utils::synchronize_stream();gpu_trace.push_back(phase("before",0,true));cusignal::DeviceArray<ComplexDouble> trace_output(x.size());gpu_trace.push_back(phase("allocate",1,true));auto trace_input=cusignal::DeviceArray<T>::from_host(x);gpu_trace.push_back(phase("h2d",2,true));cusignal::freq_shift_device(trace_input,trace_output,c.freq,c.sample_rate);cusignal::cuda_utils::synchronize_stream();gpu_trace.push_back(phase("execute_sync",3,true));auto trace_host=trace_output.to_host();gpu_trace.push_back(phase("d2h",4,true));trace_output={};trace_input={};trace_host.clear();trace_host.shrink_to_fit();cusignal::cuda_utils::synchronize_stream();gpu_trace.push_back(phase("release_sync",5,true));gpu_trace.push_back(phase("after",6,true));
    auto accuracy=zkx::common::compare(flatten(cpu_output),flatten(gpu_output),1e-12);if(!std::isfinite(accuracy.mse)||!std::isfinite(accuracy.rmse)||!std::isfinite(accuracy.relative_l2)||!std::isfinite(accuracy.relative_linf))throw std::runtime_error("nonfinite accuracy metric");
    std::string manifest="x:"+c.dtype+":["+std::to_string(x.size())+"]:bytes="+std::to_string(x.size()*sizeof(T))+":sha256="+digest(x);std::string input_digest=zkx::remaining_operator_case::sha256_hex(manifest);
    if(fs::exists(c.output))throw std::runtime_error("exists");std::ofstream o(c.output);o<<"{\"schema_version\":1,\"operator_name\":\"freq_shift\",\"backend\":\"not_applicable\",\"requested_dtype\":\""<<c.dtype<<"\",\"compute_dtype\":\"ComplexFP32\",\"fft_dtype\":\"not_applicable\",\"output_dtype\":\"ComplexFP64\",\"freq\":"<<std::setprecision(17)<<c.freq<<",\"fs\":"<<c.sample_rate<<",\"typed_input_count\":1,\"typed_input_manifest\":\""<<manifest<<"\",\"actual_input_digest\":\""<<input_digest<<"\",\"cpu_output_digest\":\""<<digest(cpu_output)<<"\",\"gpu_output_digest\":\""<<digest(gpu_output)<<"\",\"mse\":"<<accuracy.mse<<",\"rmse\":"<<accuracy.rmse<<",\"relative_l2\":"<<accuracy.relative_l2<<",\"relative_linf\":"<<accuracy.relative_linf<<",\"cpu_samples_ms\":";samples(o,cpu);o<<",\"gpu_samples_ms\":";samples(o,gpu);o<<",\"cpu_trace\":";phases(o,cpu_trace);o<<",\"gpu_trace\":";phases(o,gpu_trace);o<<"}\n";
    std::cout<<"REMAINING_OPERATOR_CASE_FREQ_SHIFT_PROBE PASS dtype="<<c.dtype<<" backend=not_applicable\n";return 0;
}
}
int main(int argc,char** argv){try{auto c=cli(argc,argv);if(c.dtype=="FP32")return run<float>(c);if(c.dtype=="FP16")return run<__half>(c);if(c.dtype=="INT32")return run<std::int32_t>(c);if(c.dtype=="INT16")return run<std::int16_t>(c);if(c.dtype=="INT8")return run<std::int8_t>(c);throw std::invalid_argument("dtype");}catch(const std::exception& e){std::cerr<<"REMAINING_OPERATOR_CASE_FREQ_SHIFT_PROBE FAIL error="<<e.what()<<'\n';return 2;}}
