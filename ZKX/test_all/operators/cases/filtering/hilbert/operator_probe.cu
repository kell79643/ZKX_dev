#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/filtering/filtering_typed.h>
#include "memory.h"
#include "sha256.h"
#include <cuda_fp16.h>
#include <chrono>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
namespace fs = std::filesystem;
struct Cli { std::string dtype; int rows{},cols{},axis{}; std::uint64_t seed{}; std::size_t warmup{},measured{}; fs::path output; };
struct Phase { std::string name; int index{}; zkx::common::ResourceSample heap,rss,gpu; };

Cli cli(int argc,char** argv) {
    std::map<std::string,std::string> v;
    for(int i=1;i<argc;i+=2){if(i+1>=argc||std::string(argv[i]).rfind("--",0))throw std::invalid_argument("expected --key value");v.emplace(std::string(argv[i]).substr(2),argv[i+1]);}
    auto get=[&](const char* k)->const std::string&{auto i=v.find(k);if(i==v.end())throw std::invalid_argument(std::string("missing --")+k);return i->second;};
    Cli c{get("dtype"),std::stoi(get("rows")),std::stoi(get("cols")),std::stoi(get("axis")),std::stoull(get("seed")),std::stoull(get("warmup")),std::stoull(get("measured")),get("output")};
    if(v.size()!=8||c.rows<1||c.cols<1||(c.axis!=0&&c.axis!=1)||!c.warmup||!c.measured)throw std::invalid_argument("invalid hilbert arguments");return c;
}
zkx::common::ResourceSample gpu_sample(){return zkx::common::sample_gpu_used([]{auto i=cusignal::cuda_utils::current_memory_info();return static_cast<std::int64_t>(i.total_bytes-i.free_bytes);},"cudaMemGetInfo.total_minus_free");}
Phase phase(const char* n,int i,bool gpu){return {n,i,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU path"}};}
template<class T>T cast(float x){return static_cast<T>(x);} template<> __half cast<__half>(float x){return __float2half_rn(x);}
template<class T>std::vector<T> input(std::size_t n,std::uint64_t seed){std::vector<T> v;v.reserve(n);for(std::size_t i=0;i<n;++i){int r=static_cast<int>((i*17+seed)%23)-11;float x=std::is_integral_v<T>?float(r):float(r)*0.25f;v.push_back(cast<T>(x));}return v;}
template<class T>std::string digest(const std::vector<T>& v){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)));}
template<class T>std::vector<double> measure(std::size_t w,std::size_t n,T&& f){for(std::size_t i=0;i<w;++i)f();std::vector<double> s;for(std::size_t i=0;i<n;++i){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::milli>(b-a).count());}return s;}
std::vector<std::complex<double>> cv(const std::vector<ComplexFloat>& v){std::vector<std::complex<double>> o;for(auto x:v)o.emplace_back(x.re,x.im);return o;}
std::vector<std::complex<double>> cv(const std::vector<ComplexDouble>& v){std::vector<std::complex<double>> o;for(auto x:v)o.emplace_back(x.re,x.im);return o;}
void samples(std::ostream& o,const std::vector<double>& v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<std::setprecision(17)<<v[i];}o<<']';}
void sample(std::ostream& o,const zkx::common::ResourceSample& s){o<<"{\"available\":"<<(s.available?"true":"false")<<",\"bytes\":"<<(s.available?std::to_string(s.bytes):"null")<<'}';}
void phases(std::ostream& o,const std::vector<Phase>& v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<"{\"phase\":\""<<v[i].name<<"\",\"phase_index\":"<<v[i].index<<",\"heap\":";sample(o,v[i].heap);o<<",\"rss\":";sample(o,v[i].rss);o<<",\"gpu\":";sample(o,v[i].gpu);o<<'}';}o<<']';}

template<class T>int run(const Cli& c){
    auto x=input<T>(static_cast<std::size_t>(c.rows)*c.cols,c.seed); cusignal::HilbertOptions opt;opt.shape={c.rows,c.cols};opt.axis=c.axis;opt.fft_length=c.axis?c.cols:c.rows;
    cusignal::HilbertCpuResult cr;cusignal::HilbertDeviceResult gr;std::vector<ComplexFloat> gf;std::vector<ComplexDouble> gd;
    auto cpu=measure(c.warmup,c.measured,[&]{cr=cusignal::hilbert_typed_cpu(x,opt);});
    auto gpu=measure(c.warmup,c.measured,[&]{auto dx=cusignal::DeviceArray<T>::from_host(x);cusignal::hilbert_device(dx,gr,opt);if constexpr(std::is_integral_v<T>)gd=gr.complex_fp64.to_host();else gf=gr.complex_fp32.to_host();});
    std::vector<Phase> ct{phase("before",0,false)};auto cx=x;ct.push_back(phase("allocate",1,false));ct.push_back(phase("h2d",2,false));auto tc=cusignal::hilbert_typed_cpu(cx,opt);ct.push_back(phase("execute_sync",3,false));ct.push_back(phase("d2h",4,false));tc={};cx.clear();ct.push_back(phase("release_sync",5,false));ct.push_back(phase("after",6,false));
    std::vector<Phase> gt;cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("before",0,true));std::optional<cusignal::DeviceArray<T>> dx;cusignal::HilbertDeviceResult tg;gt.push_back(phase("allocate",1,true));dx.emplace(cusignal::DeviceArray<T>::from_host(x));gt.push_back(phase("h2d",2,true));cusignal::hilbert_device(*dx,tg,opt);cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("execute_sync",3,true));if constexpr(std::is_integral_v<T>)gd=tg.complex_fp64.to_host();else gf=tg.complex_fp32.to_host();gt.push_back(phase("d2h",4,true));tg={};dx.reset();cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("release_sync",5,true));gt.push_back(phase("after",6,true));
    auto ref=std::is_integral_v<T>?cv(cr.complex_fp64):cv(cr.complex_fp32);auto actual=std::is_integral_v<T>?cv(gd):cv(gf);auto a=zkx::common::compare(ref,actual,1e-12);std::string in=digest(x),outc=std::is_integral_v<T>?digest(cr.complex_fp64):digest(cr.complex_fp32),outg=std::is_integral_v<T>?digest(gd):digest(gf);std::string od=std::is_integral_v<T>?"ComplexFP64":"ComplexFP32";
    if(fs::exists(c.output))throw std::runtime_error("output exists");std::ofstream o(c.output);if(!o)throw std::runtime_error("cannot create output");
    o<<"{\"schema_version\":1,\"operator_name\":\"hilbert\",\"backend\":\""<<ZKX_SELECTED_FFT_BACKEND_NAME<<"\",\"requested_dtype\":\""<<c.dtype<<"\",\"compute_dtype\":\"FP32\",\"fft_dtype\":\"ComplexFP32\",\"output_dtype\":\""<<od<<"\",\"typed_input_count\":1,\"typed_input_manifest\":\"x:"<<c.dtype<<":["<<x.size()<<"]:bytes="<<x.size()*sizeof(T)<<":sha256="<<in<<"\",\"actual_input_digest\":\""<<in<<"\",\"cpu_output_digest\":\""<<outc<<"\",\"gpu_output_digest\":\""<<outg<<"\",\"output_shape\":["<<cr.shape[0]<<','<<cr.shape[1]<<"],\"mse\":"<<std::setprecision(17)<<a.mse<<",\"rmse\":"<<a.rmse<<",\"relative_l2\":"<<a.relative_l2<<",\"relative_linf\":"<<a.relative_linf<<",\"cpu_samples_ms\":";samples(o,cpu);o<<",\"gpu_samples_ms\":";samples(o,gpu);o<<",\"cpu_trace\":";phases(o,ct);o<<",\"gpu_trace\":";phases(o,gt);o<<"}\n";
    std::cout<<"REMAINING_OPERATOR_CASE_HILBERT_PROBE PASS dtype="<<c.dtype<<" shape="<<c.rows<<'x'<<c.cols<<" backend="<<ZKX_SELECTED_FFT_BACKEND_NAME<<'\n';return 0;
}
}
int main(int argc,char** argv){try{auto c=cli(argc,argv);if(c.dtype=="FP32")return run<float>(c);if(c.dtype=="FP16")return run<__half>(c);if(c.dtype=="INT32")return run<std::int32_t>(c);if(c.dtype=="INT16")return run<std::int16_t>(c);if(c.dtype=="INT8")return run<std::int8_t>(c);throw std::invalid_argument("dtype");}catch(const std::exception& e){std::cerr<<"REMAINING_OPERATOR_CASE_HILBERT_PROBE FAIL error="<<e.what()<<'\n';return 2;}}
