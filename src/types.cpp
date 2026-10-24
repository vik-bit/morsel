#include "morsel/types.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

namespace morsel {

int Dictionary::find(std::string_view value) const {
  const auto it = std::lower_bound(values.begin(), values.end(), value,
                                   [](const std::string& a, std::string_view b) { return a < b; });
  if (it == values.end() || *it != value) return -1;
  return static_cast<int>(it - values.begin());
}

std::size_t type_width(TypeId id) {
  switch (id) {
    case TypeId::Int32:
    case TypeId::Date:
      return 4;
    case TypeId::Int64:
    case TypeId::Double:
      return 8;
    case TypeId::Int128:
    case TypeId::String:
      return 16;
    case TypeId::Dict:
      return 1;
  }
  return 0;
}

std::string type_name(const Type& type) {
  const std::string scale = type.scale ? "(scale " + std::to_string(type.scale) + ")" : "";
  switch (type.id) {
    case TypeId::Int32:
      return "int32" + scale;
    case TypeId::Int64:
      return "int64" + scale;
    case TypeId::Int128:
      return "int128" + scale;
    case TypeId::Double:
      return "double";
    case TypeId::Date:
      return "date";
    case TypeId::Dict:
      return "dict";
    case TypeId::String:
      return "string";
  }
  return "?";
}

std::int64_t pow10_i64(int exponent) {
  if (exponent < 0 || exponent > 18) throw std::invalid_argument("pow10 out of range");
  std::int64_t v = 1;
  for (int i = 0; i < exponent; ++i) v *= 10;
  return v;
}

// Howard Hinnant's days_from_civil / civil_from_days.
std::int32_t parse_date(std::string_view text) {
  int y = 0, m = 0, d = 0;
  if (text.size() != 10 || text[4] != '-' || text[7] != '-' ||
      std::sscanf(std::string(text).c_str(), "%4d-%2d-%2d", &y, &m, &d) != 3 || m < 1 || m > 12 ||
      d < 1 || d > 31) {
    throw std::invalid_argument("bad date: " + std::string(text));
  }
  y -= m <= 2;
  const int era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = static_cast<unsigned>(y - era * 400);
  const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + static_cast<unsigned>(d) - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + static_cast<int>(doe) - 719468;
}

std::string format_date(std::int32_t days) {
  const int z = days + 719468;
  const int era = (z >= 0 ? z : z - 146096) / 146097;
  const unsigned doe = static_cast<unsigned>(z - era * 146097);
  const unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
  const int y = static_cast<int>(yoe) + era * 400;
  const unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
  const unsigned mp = (5 * doy + 2) / 153;
  const unsigned d = doy - (153 * mp + 2) / 5 + 1;
  const unsigned m = mp < 10 ? mp + 3 : mp - 9;
  char buf[16];
  std::snprintf(buf, sizeof buf, "%04d-%02u-%02u", y + (m <= 2), m, d);
  return buf;
}

std::string format_decimal(int128 value, int scale) {
  const bool negative = value < 0;
  unsigned __int128 v = negative ? -static_cast<unsigned __int128>(value)
                                 : static_cast<unsigned __int128>(value);
  std::string digits;
  do {
    digits.push_back(static_cast<char>('0' + static_cast<int>(v % 10)));
    v /= 10;
  } while (v != 0);
  while (static_cast<int>(digits.size()) <= scale) digits.push_back('0');
  std::reverse(digits.begin(), digits.end());
  if (scale > 0) digits.insert(digits.end() - scale, '.');
  return negative ? "-" + digits : digits;
}

DecimalLiteral parse_decimal(std::string_view text) {
  if (text.empty()) throw std::invalid_argument("empty decimal");
  bool negative = false;
  std::size_t i = 0;
  if (text[0] == '-' || text[0] == '+') {
    negative = text[0] == '-';
    i = 1;
  }
  std::int64_t value = 0;
  int scale = -1;
  bool any_digit = false;
  for (; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '.' && scale < 0) {
      scale = 0;
    } else if (c >= '0' && c <= '9') {
      value = value * 10 + (c - '0');
      any_digit = true;
      if (scale >= 0) ++scale;
    } else {
      throw std::invalid_argument("bad decimal: " + std::string(text));
    }
  }
  if (!any_digit || scale > 18) throw std::invalid_argument("bad decimal: " + std::string(text));
  return {negative ? -value : value, scale < 0 ? 0 : scale};
}

std::string format_value(const Type& type, const void* column, std::size_t row) {
  switch (type.id) {
    case TypeId::Int32:
      return format_decimal(static_cast<const std::int32_t*>(column)[row], type.scale);
    case TypeId::Int64:
      return format_decimal(static_cast<const std::int64_t*>(column)[row], type.scale);
    case TypeId::Int128:
      return format_decimal(static_cast<const int128*>(column)[row], type.scale);
    case TypeId::Double: {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%.17g", static_cast<const double*>(column)[row]);
      return buf;
    }
    case TypeId::Date:
      return format_date(static_cast<const std::int32_t*>(column)[row]);
    case TypeId::Dict:
      return type.dict->values[static_cast<const std::uint8_t*>(column)[row]];
    case TypeId::String:
      return std::string(static_cast<const std::string_view*>(column)[row]);
  }
  return "";
}

}  // namespace morsel
