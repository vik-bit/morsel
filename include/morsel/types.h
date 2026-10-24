#pragma once

// Logical types and value formatting. Every type has a fixed width, so a
// column of any type is one flat array and a vector of it is a pointer.

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace morsel {

using int128 = __int128;

enum class TypeId : std::uint8_t {
  Int32,   // integers; with scale > 0, a fixed-point decimal
  Int64,   // decimals are int64 scaled by 10^scale, as in DuckDB's DECIMAL(15,2)
  Int128,  // sums of int32/int64, which cannot overflow
  Double,
  Date,    // days since 1970-01-01, int32
  Dict,    // uint8 code into a sorted dictionary (code order == string order)
  String,  // std::string_view into immutable storage
};

struct Dictionary {
  std::vector<std::string> values;  // sorted, at most 256 entries

  // Index of value, or -1 if absent.
  int find(std::string_view value) const;
};

struct Type {
  TypeId id = TypeId::Int64;
  std::uint8_t scale = 0;
  const Dictionary* dict = nullptr;  // only for Dict

  static Type int32(int s = 0) { return {TypeId::Int32, static_cast<std::uint8_t>(s), nullptr}; }
  static Type int64(int s = 0) { return {TypeId::Int64, static_cast<std::uint8_t>(s), nullptr}; }
  static Type int128(int s = 0) { return {TypeId::Int128, static_cast<std::uint8_t>(s), nullptr}; }
  static Type float64() { return {TypeId::Double, 0, nullptr}; }
  static Type date() { return {TypeId::Date, 0, nullptr}; }
  static Type dict_of(const Dictionary* d) { return {TypeId::Dict, 0, d}; }
  static Type string() { return {TypeId::String, 0, nullptr}; }

  bool is_integer() const {
    return id == TypeId::Int32 || id == TypeId::Int64 || id == TypeId::Int128;
  }
  bool is_numeric() const { return is_integer() || id == TypeId::Double; }
};

std::size_t type_width(TypeId id);
std::string type_name(const Type& type);

std::int64_t pow10_i64(int exponent);

// "1994-01-01" <-> days since 1970-01-01 (proleptic Gregorian calendar).
std::int32_t parse_date(std::string_view text);
std::string format_date(std::int32_t days);

// Exact decimal text: format_decimal(12345, 2) == "123.45".
std::string format_decimal(int128 value, int scale);

// Parses "0.05" into {5, 2}; throws on malformed input.
struct DecimalLiteral {
  std::int64_t value;
  int scale;
};
DecimalLiteral parse_decimal(std::string_view text);

// One value of a column as text: what the CLI prints and the tests compare.
std::string format_value(const Type& type, const void* column, std::size_t row);

}  // namespace morsel
