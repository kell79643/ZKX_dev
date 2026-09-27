#include "json_value.h"

#include <cmath>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <utility>

namespace zkx::config {
namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    JsonValue parse_document()
    {
        skip_space();
        JsonValue value = parse_value();
        skip_space();
        if (position_ != text_.size()) fail("unexpected trailing data");
        return value;
    }

private:
    JsonValue parse_value()
    {
        if (position_ >= text_.size()) fail("unexpected end of JSON");
        const char value = text_[position_];
        if (value == '{') return parse_object();
        if (value == '[') return parse_array();
        if (value == '"') return JsonValue(parse_string());
        if (value == 't') { consume_literal("true"); return JsonValue(true); }
        if (value == 'f') { consume_literal("false"); return JsonValue(false); }
        if (value == 'n') { consume_literal("null"); return JsonValue(); }
        if (value == '-' || (value >= '0' && value <= '9')) return JsonValue(parse_number());
        fail("unexpected token");
        return JsonValue();
    }

    JsonValue parse_object()
    {
        ++position_;
        JsonValue::Object object;
        skip_space();
        if (take('}')) return JsonValue(std::move(object));
        while (true) {
            skip_space();
            if (position_ >= text_.size() || text_[position_] != '"')
                fail("object key must be a string");
            std::string key = parse_string();
            if (object.find(key) != object.end()) fail("duplicate object key: " + key);
            skip_space();
            require(':');
            skip_space();
            object.emplace(std::move(key), parse_value());
            skip_space();
            if (take('}')) break;
            require(',');
        }
        return JsonValue(std::move(object));
    }

    JsonValue parse_array()
    {
        ++position_;
        JsonValue::Array array;
        skip_space();
        if (take(']')) return JsonValue(std::move(array));
        while (true) {
            skip_space();
            array.push_back(parse_value());
            skip_space();
            if (take(']')) break;
            require(',');
        }
        return JsonValue(std::move(array));
    }

    std::string parse_string()
    {
        require('"');
        std::string output;
        while (position_ < text_.size()) {
            const unsigned char value = static_cast<unsigned char>(text_[position_++]);
            if (value == '"') return output;
            if (value < 0x20) fail("control character in string");
            if (value != '\\') { output.push_back(static_cast<char>(value)); continue; }
            if (position_ >= text_.size()) fail("unfinished escape");
            const char escaped = text_[position_++];
            switch (escaped) {
            case '"': output.push_back('"'); break;
            case '\\': output.push_back('\\'); break;
            case '/': output.push_back('/'); break;
            case 'b': output.push_back('\b'); break;
            case 'f': output.push_back('\f'); break;
            case 'n': output.push_back('\n'); break;
            case 'r': output.push_back('\r'); break;
            case 't': output.push_back('\t'); break;
            case 'u': append_unicode(output); break;
            default: fail("invalid string escape");
            }
        }
        fail("unterminated string");
        return {};
    }

    void append_unicode(std::string& output)
    {
        unsigned code = 0;
        for (int index = 0; index < 4; ++index) {
            if (position_ >= text_.size()) fail("unfinished unicode escape");
            const char value = text_[position_++];
            code <<= 4;
            if (value >= '0' && value <= '9') code += value - '0';
            else if (value >= 'a' && value <= 'f') code += value - 'a' + 10;
            else if (value >= 'A' && value <= 'F') code += value - 'A' + 10;
            else fail("invalid unicode escape");
        }
        if (code <= 0x7f) output.push_back(static_cast<char>(code));
        else if (code <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (code >> 6)));
            output.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xe0 | (code >> 12)));
            output.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (code & 0x3f)));
        }
    }

    double parse_number()
    {
        const std::size_t begin = position_;
        if (take('-') && position_ == text_.size()) fail("unfinished number");
        if (take('0')) {
            if (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])) != 0)
                fail("leading zero in number");
        } else {
            if (!take_digits()) fail("number needs integer digits");
        }
        if (take('.')) {
            if (!take_digits()) fail("number needs fraction digits");
        }
        if (take('e') || take('E')) {
            (void)(take('+') || take('-'));
            if (!take_digits()) fail("number needs exponent digits");
        }
        const std::string token = text_.substr(begin, position_ - begin);
        char* end = nullptr;
        const double value = std::strtod(token.c_str(), &end);
        if (end == nullptr || *end != '\0' || !std::isfinite(value)) fail("invalid/non-finite number");
        return value;
    }

    bool take_digits()
    {
        const std::size_t begin = position_;
        while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_])) != 0)
            ++position_;
        return position_ != begin;
    }

    void consume_literal(const char* literal)
    {
        while (*literal != '\0') {
            if (position_ >= text_.size() || text_[position_++] != *literal++) fail("invalid literal");
        }
    }

    void skip_space()
    {
        while (position_ < text_.size()) {
            const char value = text_[position_];
            if (value != ' ' && value != '\n' && value != '\r' && value != '\t') break;
            ++position_;
        }
    }

    bool take(char expected)
    {
        if (position_ < text_.size() && text_[position_] == expected) { ++position_; return true; }
        return false;
    }

    void require(char expected)
    {
        if (!take(expected)) fail(std::string("expected '") + expected + "'");
    }

    [[noreturn]] void fail(const std::string& message) const { throw JsonError(position_, message); }

    const std::string& text_;
    std::size_t position_ = 0;
};

[[noreturn]] void type_error(const char* expected)
{
    throw std::logic_error(std::string("JSON type is not ") + expected);
}

}  // namespace

JsonError::JsonError(std::size_t offset, const std::string& message)
    : std::runtime_error(message), offset_(offset) {}
JsonValue::JsonValue() = default;
JsonValue::JsonValue(bool value) : type_(Type::boolean), boolean_(value) {}
JsonValue::JsonValue(double value) : type_(Type::number), number_(value) {}
JsonValue::JsonValue(std::string value) : type_(Type::string), string_(std::move(value)) {}
JsonValue::JsonValue(Array value) : type_(Type::array), array_(std::move(value)) {}
JsonValue::JsonValue(Object value) : type_(Type::object), object_(std::move(value)) {}
bool JsonValue::as_boolean() const { if (!is_boolean()) type_error("boolean"); return boolean_; }
double JsonValue::as_number() const { if (!is_number()) type_error("number"); return number_; }
const std::string& JsonValue::as_string() const { if (!is_string()) type_error("string"); return string_; }
const JsonValue::Array& JsonValue::as_array() const { if (!is_array()) type_error("array"); return array_; }
const JsonValue::Object& JsonValue::as_object() const { if (!is_object()) type_error("object"); return object_; }
const JsonValue* JsonValue::find(const std::string& key) const
{
    if (!is_object()) return nullptr;
    const auto found = object_.find(key);
    return found == object_.end() ? nullptr : &found->second;
}
JsonValue JsonValue::parse(const std::string& text) { return Parser(text).parse_document(); }
JsonValue JsonValue::parse_file(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open JSON file: " + path);
    std::ostringstream contents;
    contents << input.rdbuf();
    return parse(contents.str());
}

std::string json_escape(const std::string& value)
{
    std::ostringstream output;
    for (const unsigned char character : value) {
        switch (character) {
        case '"': output << "\\\""; break;
        case '\\': output << "\\\\"; break;
        case '\n': output << "\\n"; break;
        case '\r': output << "\\r"; break;
        case '\t': output << "\\t"; break;
        default:
            if (character < 0x20) output << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character);
            else output << static_cast<char>(character);
        }
    }
    return output.str();
}

}  // namespace zkx::config
