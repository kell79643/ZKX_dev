#include "accuracy.h"
#include <cusignal/operators/convolution/convolution_typed.h>
#include <cusignal/runtime/cuda_utils.h>
#include "memory.h"
#include "sha256.h"
#include <cuda_fp16.h>
#include <chrono>
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
struct Cli{std::string dtype,mode,boundary;int r1{},c1{},r2{},c2{};double fill{};std::uint64_t seed{};std::size_t warmup{},measured{};fs::path output;};
struct Phase{std::string name;int index{};zkx::common::ResourceSample heap,rss,gpu;};
Cli cli(int argc,char**argv){std::map<std::string,std::string>v;for(int i=1;i<argc;i+=2){if(i+1>=argc||std::string(argv[i]).rfind("--",0))throw std::invalid_argument("expected --key value");v.emplace(std::string(argv[i]).substr(2),argv[i+1]);}auto g=[&](const char*k)->const std::string&{auto i=v.find(k);if(i==v.end())throw std::invalid_argument(std::string("missing --")+k);return i->second;};Cli c{g("dtype"),g("mode"),g("boundary"),std::stoi(g("rows1")),std::stoi(g("cols1")),std::stoi(g("rows2")),std::stoi(g("cols2")),std::stod(g("fillvalue")),std::stoull(g("seed")),std::stoull(g("warmup")),std::stoull(g("measured")),g("output")};if(v.size()!=12||c.r1<1||c.c1<1||c.r2<1||c.c2<1||!c.warmup||!c.measured)throw std::invalid_argument("invalid arguments");return c;}
zkx::common::ResourceSample gpu_sample(){return zkx::common::sample_gpu_used([]{auto i=cusignal::cuda_utils::current_memory_info();return static_cast<std::int64_t>(i.total_bytes-i.free_bytes);},"cudaMemGetInfo.total_minus_free");}
Phase phase(const char*n,int i,bool gpu){return{n,i,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU path"}};}
template<class T>T cast(float x){return static_cast<T>(x);}template<>__half cast<__half>(float x){return __float2half_rn(x);}
template<class T>std::vector<T> input(std::size_t n,std::uint64_t seed,int salt){std::vector<T>v;v.reserve(n);for(std::size_t i=0;i<n;++i){int r=static_cast<int>((i*17+seed+salt)%11)-5;float x=std::is_integral_v<T>?float(r):float(r)*0.25f;v.push_back(cast<T>(x));}return v;}
template<class T>std::string digest(const std::vector<T>&v){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)));}
template<class T>double value(T x){return static_cast<double>(x);}template<>double value(__half x){return __half2float(x);}
template<class T>std::vector<double> dv(const std::vector<T>&v){std::vector<double>r;r.reserve(v.size());for(auto x:v)r.push_back(value(x));return r;}
template<class F>std::vector<double> measure(std::size_t w,std::size_t n,F&&f){for(std::size_t i=0;i<w;++i)f();std::vector<double>s;for(std::size_t i=0;i<n;++i){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::milli>(b-a).count());}return s;}
void samples(std::ostream&o,const std::vector<double>&v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<std::setprecision(17)<<v[i];}o<<']';}
void sample(std::ostream&o,const zkx::common::ResourceSample&s){o<<"{\"available\":"<<(s.available?"true":"false")<<",\"bytes\":"<<(s.available?std::to_string(s.bytes):"null")<<'}';}
void phases(std::ostream&o,const std::vector<Phase>&v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<"{\"phase\":\""<<v[i].name<<"\",\"phase_index\":"<<v[i].index<<",\"heap\":";sample(o,v[i].heap);o<<",\"rss\":";sample(o,v[i].rss);o<<",\"gpu\":";sample(o,v[i].gpu);o<<'}';}o<<']';}

template<class T>int run(const Cli&c){
 auto x=input<T>(static_cast<std::size_t>(c.r1)*c.c1,c.seed,3),h=input<T>(static_cast<std::size_t>(c.r2)*c.c2,c.seed,7);cusignal::Correlate2DOptions opt;opt.rows1=c.r1;opt.cols1=c.c1;opt.rows2=c.r2;opt.cols2=c.c2;opt.mode=c.mode;opt.boundary=c.boundary;opt.fillvalue=c.fill;
 std::vector<T>cr,gr;auto cpu=measure(c.warmup,c.measured,[&]{cr=cusignal::correlate2d_typed_cpu(x,h,opt);});auto gpu=measure(c.warmup,c.measured,[&]{auto dx=cusignal::DeviceArray<T>::from_host(x);auto dh=cusignal::DeviceArray<T>::from_host(h);cusignal::Correlate2DWorkspace ws(opt);cusignal::DeviceArray<T>dy(ws.output_size());cusignal::correlate2d_device(dx,dh,dy,ws);gr=dy.to_host();});
 std::vector<Phase>ct{phase("before",0,false)};auto cx=x,ch=h;ct.push_back(phase("allocate",1,false));ct.push_back(phase("h2d",2,false));auto co=cusignal::correlate2d_typed_cpu(cx,ch,opt);ct.push_back(phase("execute_sync",3,false));ct.push_back(phase("d2h",4,false));co.clear();cx.clear();ch.clear();ct.push_back(phase("release_sync",5,false));ct.push_back(phase("after",6,false));
 std::vector<Phase>gt;cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("before",0,true));gt.push_back(phase("allocate",1,true));auto dx=cusignal::DeviceArray<T>::from_host(x);auto dh=cusignal::DeviceArray<T>::from_host(h);cusignal::Correlate2DWorkspace ws(opt);cusignal::DeviceArray<T>dy(ws.output_size());gt.push_back(phase("h2d",2,true));cusignal::correlate2d_device(dx,dh,dy,ws);cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("execute_sync",3,true));auto go=dy.to_host();gt.push_back(phase("d2h",4,true));dy={};dx={};dh={};cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("release_sync",5,true));gt.push_back(phase("after",6,true));
 auto a=zkx::common::compare(dv(cr),dv(gr),1e-12);std::size_t mismatch=0;for(std::size_t i=0;i<cr.size();++i)if(value(cr[i])!=value(gr[i]))++mismatch;auto shape=cusignal::correlate2d_output_shape(opt);std::string xd=digest(x),hd=digest(h);std::ostringstream manifest;manifest<<"in1:"<<c.dtype<<":["<<c.r1<<'x'<<c.c1<<"]:bytes="<<x.size()*sizeof(T)<<":sha256="<<xd<<"|in2:"<<c.dtype<<":["<<c.r2<<'x'<<c.c2<<"]:bytes="<<h.size()*sizeof(T)<<":sha256="<<hd;std::string in=zkx::remaining_operator_case::sha256_hex(manifest.str());
 if(fs::exists(c.output))throw std::runtime_error("output exists");std::ofstream o(c.output);if(!o)throw std::runtime_error("cannot create output");o<<"{\"schema_version\":1,\"operator_name\":\"correlate2d\",\"backend\":\"not_applicable\",\"requested_dtype\":\""<<c.dtype<<"\",\"compute_dtype\":\""<<(std::is_same_v<T,std::int32_t>?"INT32_modular":"FP32")<<"\",\"fft_dtype\":\"not_applicable\",\"output_dtype\":\""<<c.dtype<<"\",\"mode\":\""<<c.mode<<"\",\"boundary\":\""<<c.boundary<<"\",\"fillvalue\":"<<c.fill<<",\"typed_input_count\":2,\"typed_input_manifest\":\""<<manifest.str()<<"\",\"actual_input_digest\":\""<<in<<"\",\"cpu_output_digest\":\""<<digest(cr)<<"\",\"gpu_output_digest\":\""<<digest(gr)<<"\",\"output_shape\":["<<shape.rows<<','<<shape.cols<<"],\"mismatch_count\":"<<mismatch<<",\"exact_match\":"<<(mismatch?"false":"true")<<",\"mse\":"<<std::setprecision(17)<<a.mse<<",\"rmse\":"<<a.rmse<<",\"relative_l2\":"<<a.relative_l2<<",\"relative_linf\":"<<a.relative_linf<<",\"cpu_samples_ms\":";samples(o,cpu);o<<",\"gpu_samples_ms\":";samples(o,gpu);o<<",\"cpu_trace\":";phases(o,ct);o<<",\"gpu_trace\":";phases(o,gt);o<<"}\n";
 std::cout<<"REMAINING_OPERATOR_CASE_CORRELATE2D_PROBE PASS dtype="<<c.dtype<<" mode="<<c.mode<<" boundary="<<c.boundary<<" backend=not_applicable\n";return 0;
}}
int main(int argc,char**argv){try{auto c=cli(argc,argv);if(c.dtype=="FP32")return run<float>(c);if(c.dtype=="FP16")return run<__half>(c);if(c.dtype=="INT32")return run<std::int32_t>(c);if(c.dtype=="INT16")return run<std::int16_t>(c);if(c.dtype=="INT8")return run<std::int8_t>(c);throw std::invalid_argument("dtype");}catch(const std::exception&e){std::cerr<<"REMAINING_OPERATOR_CASE_CORRELATE2D_PROBE FAIL error="<<e.what()<<'\n';return 2;}}
