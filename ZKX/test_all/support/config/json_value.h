#pragma once

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace zkx::config {

class JsonError : public std::runtime_error {
public:
    JsonError(std::size_t offset, const std::string& message);
    std::size_t offset() const noexcept { return offset_; }
private:
    std::size_t offset_;
};

class JsonValue {
public:
    enum class Type { null_value, boolean, number, string, array, object };
    using Array = std::vector<JsonValue>;
    using Object = std::map<std::string, JsonValue>;

    JsonValue();
    explicit JsonValue(bool value);
    explicit JsonValue(double value);
    explicit JsonValue(std::string value);
    explicit JsonValue(Array value);
    explicit JsonValue(Object value);

    Type type() const noexcept { return type_; }
    bool is_null() const noexcept { return type_ == Type::null_value; }
    bool is_boolean() const noexcept { return type_ == Type::boolean; }
    bool is_number() const noexcept { return type_ == Type::number; }
    bool is_string() const noexcept { return type_ == Type::string; }
    bool is_array() const noexcept { return type_ == Type::array; }
    bool is_object() const noexcept { return type_ == Type::object; }
    bool as_boolean() const;
    double as_number() const;
    const std::string& as_string() const;
    const Array& as_array() const;
    const Object& as_object() const;
    const JsonValue* find(const std::string& key) const;

    static JsonValue parse(const std::string& text);
    static JsonValue parse_file(const std::string& path);

private:
    Type type_ = Type::null_value;
    bool boolean_ = false;
    double number_ = 0.0;
    std::string string_;
    Array array_;
    Object object_;
};

std::string json_escape(const std::string& value);

}  // namespace zkx::config
