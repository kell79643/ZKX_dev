#include "accuracy.h"
#include <cusignal/runtime/cuda_utils.h>
#include <cusignal/operators/filtering/filtering_typed.h>
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
struct Cli{std::string dtype,padtype;int rows{},cols{},axis{},padlen{};std::uint64_t seed{};std::size_t warmup{},measured{};fs::path output;};
struct Phase{std::string name;int index{};zkx::common::ResourceSample heap,rss,gpu;};
Cli cli(int argc,char**argv){std::map<std::string,std::string>v;for(int i=1;i<argc;i+=2){if(i+1>=argc||std::string(argv[i]).rfind("--",0))throw std::invalid_argument("expected --key value");v.emplace(std::string(argv[i]).substr(2),argv[i+1]);}auto g=[&](const char*k)->const std::string&{auto i=v.find(k);if(i==v.end())throw std::invalid_argument(std::string("missing --")+k);return i->second;};Cli c{g("dtype"),g("padtype"),std::stoi(g("rows")),std::stoi(g("cols")),std::stoi(g("axis")),std::stoi(g("padlen")),std::stoull(g("seed")),std::stoull(g("warmup")),std::stoull(g("measured")),g("output")};if(v.size()!=10||c.rows<1||c.cols<1||c.padlen<0||!c.warmup||!c.measured)throw std::invalid_argument("invalid arguments");return c;}
zkx::common::ResourceSample gpu_sample(){return zkx::common::sample_gpu_used([]{auto i=cusignal::cuda_utils::current_memory_info();return static_cast<std::int64_t>(i.total_bytes-i.free_bytes);},"cudaMemGetInfo.total_minus_free");}
Phase phase(const char*n,int i,bool gpu){return{n,i,zkx::common::sample_cpu_heap(),zkx::common::sample_cpu_rss(),gpu?gpu_sample():zkx::common::ResourceSample{false,0,"NA","CPU path"}};}
template<class T>T cast(float x){return static_cast<T>(x);}template<>__half cast<__half>(float x){return __float2half_rn(x);}
template<class T>std::vector<T> input(std::size_t n,std::uint64_t seed){std::vector<T>v;v.reserve(n);for(std::size_t i=0;i<n;++i){int r=static_cast<int>((i*13+seed)%9)-4;v.push_back(cast<T>(std::is_integral_v<T>?float(r):float(r)*0.25F));}return v;}
template<class T>std::string digest(const std::vector<T>&v){return zkx::remaining_operator_case::sha256_hex(std::string(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T)));}
template<class F>std::vector<double> measure(std::size_t w,std::size_t n,F&&f){for(std::size_t i=0;i<w;++i)f();std::vector<double>s;for(std::size_t i=0;i<n;++i){auto a=std::chrono::steady_clock::now();f();auto b=std::chrono::steady_clock::now();s.push_back(std::chrono::duration<double,std::milli>(b-a).count());}return s;}
template<class T>std::vector<double>dv(const std::vector<T>&v){return std::vector<double>(v.begin(),v.end());}
void samples(std::ostream&o,const std::vector<double>&v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<std::setprecision(17)<<v[i];}o<<']';}void sample(std::ostream&o,const zkx::common::ResourceSample&s){o<<"{\"available\":"<<(s.available?"true":"false")<<",\"bytes\":"<<(s.available?std::to_string(s.bytes):"null")<<'}';}void phases(std::ostream&o,const std::vector<Phase>&v){o<<'[';for(std::size_t i=0;i<v.size();++i){if(i)o<<',';o<<"{\"phase\":\""<<v[i].name<<"\",\"phase_index\":"<<v[i].index<<",\"heap\":";sample(o,v[i].heap);o<<",\"rss\":";sample(o,v[i].rss);o<<",\"gpu\":";sample(o,v[i].gpu);o<<'}';}o<<']';}
cusignal::Firfilter2Options options(const Cli&c){cusignal::Firfilter2Options o;o.shape={c.rows,c.cols};o.axis=c.axis;o.padlen=c.padlen;o.method=cusignal::Firfilter2Method::pad;if(c.padtype=="odd")o.padtype=cusignal::Firfilter2PadType::odd;else if(c.padtype=="even")o.padtype=cusignal::Firfilter2PadType::even;else if(c.padtype=="constant")o.padtype=cusignal::Firfilter2PadType::constant;else throw std::invalid_argument("padtype");return o;}
template<class T>int run(const Cli&c){auto x=input<T>(static_cast<std::size_t>(c.rows)*c.cols,c.seed);std::vector<T>b{cast<T>(1),cast<T>(1),cast<T>(1)};auto opt=options(c);cusignal::Firfilter2CpuResult cr;cusignal::Firfilter2DeviceResult gr;std::vector<float>gf;std::vector<double>gd;constexpr bool fp64=std::is_same_v<T,std::int32_t>;
 auto cpu=measure(c.warmup,c.measured,[&]{cr=cusignal::firfilter2_typed_cpu(b,x,opt);});auto gpu=measure(c.warmup,c.measured,[&]{auto db=cusignal::DeviceArray<T>::from_host(b);auto dx=cusignal::DeviceArray<T>::from_host(x);cusignal::firfilter2_device(db,dx,gr,opt);if constexpr(fp64)gd=gr.fp64.to_host();else gf=gr.fp32.to_host();});
 std::vector<Phase>ct{phase("before",0,false),phase("allocate",1,false),phase("h2d",2,false)};auto tc=cusignal::firfilter2_typed_cpu(b,x,opt);ct.push_back(phase("execute_sync",3,false));ct.push_back(phase("d2h",4,false));tc={};ct.push_back(phase("release_sync",5,false));ct.push_back(phase("after",6,false));
 std::vector<Phase>gt;cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("before",0,true));auto db=cusignal::DeviceArray<T>::from_host(b);auto dx=cusignal::DeviceArray<T>::from_host(x);cusignal::Firfilter2DeviceResult tg;gt.push_back(phase("allocate",1,true));gt.push_back(phase("h2d",2,true));cusignal::firfilter2_device(db,dx,tg,opt);cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("execute_sync",3,true));if constexpr(fp64)gd=tg.fp64.to_host();else gf=tg.fp32.to_host();gt.push_back(phase("d2h",4,true));tg={};db={};dx={};cusignal::cuda_utils::synchronize_stream();gt.push_back(phase("release_sync",5,true));gt.push_back(phase("after",6,true));
 auto ref=fp64?dv(cr.fp64):dv(cr.fp32);auto actual=fp64?dv(gd):dv(gf);auto a=zkx::common::compare(ref,actual,1e-12);std::ostringstream manifest;manifest<<"x:"<<c.dtype<<":["<<c.rows<<'x'<<c.cols<<"]:bytes="<<x.size()*sizeof(T)<<":sha256="<<digest(x)<<"|b:"<<c.dtype<<":[3]:bytes="<<b.size()*sizeof(T)<<":sha256="<<digest(b);std::string in=zkx::remaining_operator_case::sha256_hex(manifest.str()),od=fp64?"FP64":"FP32";
 if(fs::exists(c.output))throw std::runtime_error("output exists");std::ofstream o(c.output);o<<"{\"schema_version\":1,\"operator_name\":\"firfilter2\",\"backend\":\"not_applicable\",\"requested_dtype\":\""<<c.dtype<<"\",\"compute_dtype\":\"FP32\",\"fft_dtype\":\"not_applicable\",\"output_dtype\":\""<<od<<"\",\"axis\":"<<c.axis<<",\"padtype\":\""<<c.padtype<<"\",\"padlen\":"<<c.padlen<<",\"typed_input_count\":2,\"typed_input_manifest\":\""<<manifest.str()<<"\",\"actual_input_digest\":\""<<in<<"\",\"cpu_output_digest\":\""<<(fp64?digest(cr.fp64):digest(cr.fp32))<<"\",\"gpu_output_digest\":\""<<(fp64?digest(gd):digest(gf))<<"\",\"output_shape\":["<<c.rows<<','<<c.cols<<"],\"mse\":"<<std::setprecision(17)<<a.mse<<",\"rmse\":"<<a.rmse<<",\"relative_l2\":"<<a.relative_l2<<",\"relative_linf\":"<<a.relative_linf<<",\"cpu_samples_ms\":";samples(o,cpu);o<<",\"gpu_samples_ms\":";samples(o,gpu);o<<",\"cpu_trace\":";phases(o,ct);o<<",\"gpu_trace\":";phases(o,gt);o<<"}\n";std::cout<<"REMAINING_OPERATOR_CASE_FIRFILTER2_PROBE PASS dtype="<<c.dtype<<" padtype="<<c.padtype<<" backend=not_applicable\n";return 0;}
}
int main(int argc,char**argv){try{auto c=cli(argc,argv);if(c.dtype=="FP32")return run<float>(c);if(c.dtype=="FP16")return run<__half>(c);if(c.dtype=="INT32")return run<std::int32_t>(c);if(c.dtype=="INT16")return run<std::int16_t>(c);if(c.dtype=="INT8")return run<std::int8_t>(c);throw std::invalid_argument("dtype");}catch(const std::exception&e){std::cerr<<"REMAINING_OPERATOR_CASE_FIRFILTER2_PROBE FAIL error="<<e.what()<<'\n';return 2;}}
