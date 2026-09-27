#include "task1_scale_config.h"
#include <demo/config_file.h>
#include "test_all/support/config/json_value.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
namespace task1 { namespace {
using zkx::config::JsonValue;
const JsonValue& field(const JsonValue& v,const char* n){const auto* f=v.find(n);if(!f)throw std::invalid_argument(std::string("missing field: ")+n);return *f;}
int integer(const JsonValue& v,const char* n){const double x=field(v,n).as_number();if(!std::isfinite(x)||std::floor(x)!=x)throw std::invalid_argument(std::string(n)+" must be integer");return static_cast<int>(x);}
std::uint32_t uint32(const JsonValue& v,const char* n){const double x=field(v,n).as_number();if(!std::isfinite(x)||std::floor(x)!=x||x<0||x>4294967295.0)throw std::invalid_argument(std::string(n)+" must be uint32");return static_cast<std::uint32_t>(x);}
float f32(const JsonValue& v,const char* n){return static_cast<float>(field(v,n).as_number());}
std::string file_sha(const std::filesystem::path& p){std::ifstream in(p,std::ios::binary);if(!in)throw std::invalid_argument("cannot open task scales: "+p.string());const std::vector<unsigned char>b{std::istreambuf_iterator<char>(in),{}};return demo_config::sha256(b);}
} Task1ScaleConfig load_task1_scale(const std::filesystem::path& requested,const std::string& wanted)
{
    const auto path=std::filesystem::absolute(requested);const auto root=JsonValue::parse_file(path.string());const std::string scale_set_id=field(root,"scale_set_id").as_string();
    const auto& scales=field(field(field(root,"tasks"),"Task1"),"scales").as_array();
    for(const auto& scale:scales){if(field(scale,"scale_id").as_string()!=wanted)continue;const auto& shape=field(scale,"input_shape").as_array();if(shape.size()!=2)throw std::invalid_argument("Task1 shape must have 2 axes");const auto& p=field(scale,"parameters");const std::string sha=file_sha(path);
        TaskConfig c{path,sha,integer(p,"num_pulses"),integer(p,"samples_per_pulse"),integer(p,"pulse_samples"),f32(p,"sample_rate_hz"),f32(p,"prf_hz"),f32(p,"bandwidth_hz"),field(p,"carrier_frequency_hz").as_number(),integer(p,"target_delay_samples"),integer(p,"doppler_bin"),f32(p,"target_amplitude"),f32(p,"noise_std"),uint32(p,"noise_seed"),f32(p,"pfa"),integer(p,"cfar_guard_doppler"),integer(p,"cfar_guard_range"),integer(p,"cfar_reference_doppler"),integer(p,"cfar_reference_range"),integer(p,"ambiguity_nfreq")};c.validate();
        const auto elements=integer(scale,"actual_elements");if(elements!=c.num_pulses*c.samples_per_pulse)throw std::invalid_argument("actual_elements mismatch");
        if(shape[0].as_number()!=c.num_pulses||shape[1].as_number()!=c.samples_per_pulse)throw std::invalid_argument("input_shape does not match Task1 parameters");
        std::ostringstream id;id<<wanted<<'|'<<c.num_pulses<<'|'<<c.samples_per_pulse<<'|'<<c.pulse_samples<<'|'<<c.sample_rate_hz<<'|'<<c.prf_hz<<'|'<<c.bandwidth_hz<<'|'<<c.target_delay_samples<<'|'<<c.doppler_bin<<'|'<<c.target_amplitude<<'|'<<c.noise_std<<'|'<<c.noise_seed;const auto text=id.str();const std::vector<unsigned char> bytes(text.begin(),text.end());
        return {wanted,scale_set_id,field(scale,"order_of_magnitude").as_string(),"["+std::to_string(c.num_pulses)+","+std::to_string(c.samples_per_pulse)+"]",demo_config::sha256(bytes),elements,std::move(c)};
    }throw std::invalid_argument("Task1 scale not found: "+wanted);
}
}  // namespace task1
