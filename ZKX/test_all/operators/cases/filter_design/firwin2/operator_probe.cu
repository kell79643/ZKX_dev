#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/filter_design/filter_design_typed.h>
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
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace { namespace fs=std::filesystem;
struct Cli{std::string dtype,window;int numtaps{},nfreqs{};bool antisymmetric{};double fs_value{};std::size_t warmup{},measured{};fs::path output;};
struct Phase{std::string name;int index{};zkx::common::ResourceSample heap,rss,gpu;};
bool boolean(const std::string&v){if(v=="true")return true;if(v=="false")return false;throw std::invalid_argument("boolean");}
Cli cli(int argc,char**argv){std::map<std::string,std::string>v;for(int i=1;i<argc;i+=2){if(i+1>=argc||std::string(argv[i]).rfind("--",0))throw std::invalid_argument("expected --key value");v.emplace(std::string(argv[i]).substr(2),argv[i+1]);}auto g=[&](const char*k)->const std::string&{auto i=v.find(k);if(i==v.end())throw std::invalid_argument(std::string("missing --")+k);return i->second;};Cli c{g("dtype"),g("window"),std::stoi(g("numtaps")),std::stoi(g("nfreqs")),boolean(g("antisymmetric")),std::stod(g("fs")),std::stoull(g("warmup")),std::stoull(g("measured")),g("output")};if(v.size()!=9||c.numtaps<1||c.nfreqs<=c.numtaps||!c.warmup||!c.measured)throw std::invalid_argument("invalid arguments");return c;}
zkx::common::ResourceSample gpu_sample(){return zkx::common::sample_gpu_used([]{auto i=cusignal::cuda_utils::current_memory_info();return static_cast<std::int64_t>(i.total_bytes-i.free_bytes);},"cudaMemGetInfo.total_minus_free");}
Phase phase(const char*n,int i,bool gpu){return{n,i,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU path"}};}
template<class T>T cast(float x){return static_cast<T>(x);}template<>__half cast<__half>(float x){return __float2half_rn(x);}
template<class T>std::string digest(const std::vector<T>&v){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)));}
template<class F>std::vector<double> measure(std::size_t w,std::size_t n,F&&f){for(std::size_t i=0;i<w;++i)f();std::vector<double>s;for(std::size_t i=0;i<n;++i){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::milli>(b-a).count());}return s;}
void samples(std::ostream&o,const std::vector<double>&v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<std::setprecision(17)<<v[i];}o<<']';}
void sample(std::ostream&o,const zkx::common::ResourceSample&s){o<<"{\"available\":"<<(s.available?"true":"false")<<",\"bytes\":"<<(s.available?std::to_string(s.bytes):"null")<<'}';}
void phases(std::ostream&o,const std::vector<Phase>&v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<"{\"phase\":\""<<v[i].name<<"\",\"phase_index\":"<<v[i].index<<",\"heap\":";sample(o,v[i].heap);o<<",\"rss\":";sample(o,v[i].rss);o<<",\"gpu\":";sample(o,v[i].gpu);o<<'}';}o<<']';}
cusignal::Firwin2Options options(const Cli&c){cusignal::Firwin2Options o;o.nfreqs=c.nfreqs;o.antisymmetric=c.antisymmetric;o.fs=c.fs_value;if(c.window=="hamming")o.window_mode=cusignal::Firwin2WindowMode::hamming;else if(c.window=="none")o.window_mode=cusignal::Firwin2WindowMode::none;else if(c.window=="explicit"){o.window_mode=cusignal::Firwin2WindowMode::explicit_values;o.window.resize(c.numtaps);for(int i=0;i<c.numtaps;++i)o.window[i]=0.5F-0.5F*std::cos(6.283185307179586F*i/(c.numtaps-1));}else throw std::invalid_argument("window");return o;}

template<class T>int run(const Cli&c){
 std::vector<T>freq{cast<T>(0),cast<T>(4),cast<T>(8)},gain{cast<T>(0),cast<T>(1),cast<T>(0)};auto opt=options(c);std::vector<double>cr,gr;
 auto cpu=measure(c.warmup,c.measured,[&]{cr=cusignal::firwin2_typed_cpu<T>(c.numtaps,freq,gain,opt);});auto gpu=measure(c.warmup,c.measured,[&]{auto df=cusignal::DeviceArray<T>::from_host(freq);auto dg=cusignal::DeviceArray<T>::from_host(gain);cusignal::DeviceArray<double>dy(c.numtaps);cusignal::firwin2_device(c.numtaps,df,dg,dy,opt);gr=dy.to_host();});
 std::vector<Phase>ct{phase("before",0,false)};auto cf=freq,cg=gain;ct.push_back(phase("allocate",1,false));ct.push_back(phase("h2d",2,false));auto co=cusignal::firwin2_typed_cpu<T>(c.numtaps,cf,cg,opt);ct.push_back(phase("execute_sync",3,false));ct.push_back(phase("d2h",4,false));co.clear();cf.clear();cg.clear();ct.push_back(phase("release_sync",5,false));ct.push_back(phase("after",6,false));
 std::vector<Phase>gt;cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("before",0,true));gt.push_back(phase("allocate",1,true));auto df=cusignal::DeviceArray<T>::from_host(freq);auto dg=cusignal::DeviceArray<T>::from_host(gain);cusignal::DeviceArray<double>dy(c.numtaps);gt.push_back(phase("h2d",2,true));cusignal::firwin2_device(c.numtaps,df,dg,dy,opt);cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("execute_sync",3,true));auto go=dy.to_host();gt.push_back(phase("d2h",4,true));dy={};df={};dg={};cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("release_sync",5,true));gt.push_back(phase("after",6,true));
 auto a=zkx::common::compare(cr,gr,1e-12);std::string fd=digest(freq),gd=digest(gain);std::ostringstream manifest;manifest<<"freq:"<<c.dtype<<":[3]:bytes="<<freq.size()*sizeof(T)<<":sha256="<<fd<<"|gain:"<<c.dtype<<":[3]:bytes="<<gain.size()*sizeof(T)<<":sha256="<<gd;std::string input_digest=zkx::remaining_operator_case::sha256_hex(manifest.str());
 if(fs::exists(c.output))throw std::runtime_error("output exists");std::ofstream o(c.output);if(!o)throw std::runtime_error("cannot create output");o<<"{\"schema_version\":1,\"operator_name\":\"firwin2\",\"backend\":\""<<ZKX_SELECTED_FFT_BACKEND_NAME<<"\",\"requested_dtype\":\""<<c.dtype<<"\",\"compute_dtype\":\"FP32\",\"fft_dtype\":\"ComplexFP32\",\"output_dtype\":\"FP64\",\"numtaps\":"<<c.numtaps<<",\"nfreqs\":"<<c.nfreqs<<",\"window\":\""<<c.window<<"\",\"antisymmetric\":"<<(c.antisymmetric?"true":"false")<<",\"typed_input_count\":2,\"typed_input_manifest\":\""<<manifest.str()<<"\",\"actual_input_digest\":\""<<input_digest<<"\",\"cpu_output_digest\":\""<<digest(cr)<<"\",\"gpu_output_digest\":\""<<digest(gr)<<"\",\"output_shape\":["<<c.numtaps<<"],\"mse\":"<<std::setprecision(17)<<a.mse<<",\"rmse\":"<<a.rmse<<",\"relative_l2\":"<<a.relative_l2<<",\"relative_linf\":"<<a.relative_linf<<",\"cpu_samples_ms\":";samples(o,cpu);o<<",\"gpu_samples_ms\":";samples(o,gpu);o<<",\"cpu_trace\":";phases(o,ct);o<<",\"gpu_trace\":";phases(o,gt);o<<"}\n";
 std::cout<<"REMAINING_OPERATOR_CASE_FIRWIN2_PROBE PASS dtype="<<c.dtype<<" numtaps="<<c.numtaps<<" nfreqs="<<c.nfreqs<<" backend="<<ZKX_SELECTED_FFT_BACKEND_NAME<<'\n';return 0;
}}
int main(int argc,char**argv){try{auto c=cli(argc,argv);if(c.dtype=="FP32")return run<float>(c);if(c.dtype=="FP16")return run<__half>(c);if(c.dtype=="INT32")return run<std::int32_t>(c);if(c.dtype=="INT16")return run<std::int16_t>(c);if(c.dtype=="INT8")return run<std::int8_t>(c);throw std::invalid_argument("dtype");}catch(const std::exception&e){std::cerr<<"REMAINING_OPERATOR_CASE_FIRWIN2_PROBE FAIL error="<<e.what()<<'\n';return 2;}}
