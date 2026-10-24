#include "morsel/storage.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace morsel {

Buffer::~Buffer() { std::free(data_); }

Buffer::Buffer(Buffer&& other) noexcept
    : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
  other.data_ = nullptr;
  other.size_ = other.capacity_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    std::free(data_);
    data_ = other.data_;
    size_ = other.size_;
    capacity_ = other.capacity_;
    other.data_ = nullptr;
    other.size_ = other.capacity_ = 0;
  }
  return *this;
}

void Buffer::reserve(std::size_t bytes) {
  if (bytes <= capacity_) return;
  // Round up to whole cache lines, plus one line of slack so SIMD kernels may
  // store a full register past the last element.
  const std::size_t cap = ((std::max(bytes, capacity_ * 2) + 63) / 64) * 64 + 64;
  auto* fresh = static_cast<std::uint8_t*>(std::aligned_alloc(64, cap));
  if (!fresh) throw std::bad_alloc();
  if (size_) std::memcpy(fresh, data_, size_);
  std::free(data_);
  data_ = fresh;
  capacity_ = cap;
}

void Buffer::resize(std::size_t bytes) {
  reserve(bytes);
  size_ = bytes;
}

int Relation::index_of(std::string_view name) const {
  for (std::size_t i = 0; i < names.size(); ++i) {
    if (names[i] == name) return static_cast<int>(i);
  }
  return -1;
}

namespace {

constexpr char kMagic[8] = {'M', 'C', 'O', 'L', '1', 0, 0, 0};

struct File {
  std::FILE* f;
  std::string path;
  ~File() {
    if (f) std::fclose(f);
  }
  void read(void* dst, std::size_t bytes) {
    if (bytes && std::fread(dst, 1, bytes, f) != bytes) {
      throw std::runtime_error("truncated column file: " + path);
    }
  }
};

}  // namespace

std::unique_ptr<Column> load_column(const std::string& path, const std::string& name) {
  File file{std::fopen(path.c_str(), "rb"), path};
  if (!file.f) throw std::runtime_error("cannot open column file: " + path);
  char magic[8];
  std::uint32_t type_code = 0, scale = 0;
  std::uint64_t rows = 0, aux = 0;
  file.read(magic, 8);
  if (std::memcmp(magic, kMagic, 8) != 0) throw std::runtime_error("not a column file: " + path);
  file.read(&type_code, 4);
  file.read(&scale, 4);
  file.read(&rows, 8);
  file.read(&aux, 8);

  auto column = std::make_unique<Column>();
  column->name = name;
  column->rows = rows;
  switch (type_code) {
    case 0:
    case 2:
      column->type = type_code == 0 ? Type::int32(static_cast<int>(scale)) : Type::date();
      column->values.resize(rows * 4);
      file.read(column->values.data(), rows * 4);
      break;
    case 1:
      column->type = Type::int64(static_cast<int>(scale));
      column->values.resize(rows * 8);
      file.read(column->values.data(), rows * 8);
      break;
    case 3: {
      if (aux > 256) throw std::runtime_error("dictionary too large: " + path);
      column->dict = std::make_unique<Dictionary>();
      for (std::uint64_t i = 0; i < aux; ++i) {
        std::uint32_t len = 0;
        file.read(&len, 4);
        std::string s(len, '\0');
        file.read(s.data(), len);
        column->dict->values.push_back(std::move(s));
      }
      column->type = Type::dict_of(column->dict.get());
      column->values.resize(rows);
      file.read(column->values.data(), rows);
      break;
    }
    case 4: {
      std::vector<std::uint64_t> offsets(rows + 1);
      file.read(offsets.data(), (rows + 1) * 8);
      column->string_bytes.resize(aux);
      file.read(column->string_bytes.data(), aux);
      column->type = Type::string();
      column->values.resize(rows * sizeof(std::string_view));
      auto* views = column->values.as<std::string_view>();
      const char* base = reinterpret_cast<const char*>(column->string_bytes.data());
      for (std::uint64_t i = 0; i < rows; ++i) {
        views[i] = std::string_view(base + offsets[i], offsets[i + 1] - offsets[i]);
      }
      break;
    }
    default:
      throw std::runtime_error("unknown column type in " + path);
  }
  return column;
}

Table::Table(std::string name, std::string dir) : name_(std::move(name)), dir_(std::move(dir)) {}

bool Table::has_column(const std::string& name) const {
  if (columns_.count(name)) return true;
  if (dir_.empty()) return false;
  std::FILE* f = std::fopen((dir_ + "/" + name + ".col").c_str(), "rb");
  if (f) std::fclose(f);
  return f != nullptr;
}

const Column& Table::column(const std::string& name) {
  auto it = columns_.find(name);
  if (it != columns_.end()) return *it->second;
  if (dir_.empty()) throw std::runtime_error("no column " + name + " in table " + name_);
  add_column(load_column(dir_ + "/" + name + ".col", name));
  return *columns_.at(name);
}

void Table::add_column(std::unique_ptr<Column> column) {
  if (rows_known_ && column->rows != rows_) {
    throw std::runtime_error("column " + column->name + " has a different row count");
  }
  rows_ = column->rows;
  rows_known_ = true;
  columns_[column->name] = std::move(column);
}

void Table::add_dict_column(const std::string& name, const std::vector<std::string>& values) {
  auto column = std::make_unique<Column>();
  column->name = name;
  column->rows = values.size();
  column->dict = std::make_unique<Dictionary>();
  column->dict->values = values;
  std::sort(column->dict->values.begin(), column->dict->values.end());
  column->dict->values.erase(std::unique(column->dict->values.begin(), column->dict->values.end()),
                             column->dict->values.end());
  if (column->dict->values.size() > 256) throw std::invalid_argument("too many distinct values");
  column->type = Type::dict_of(column->dict.get());
  column->values.resize(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    column->values.data()[i] = static_cast<std::uint8_t>(column->dict->find(values[i]));
  }
  add_column(std::move(column));
}

void Table::add_string_column(const std::string& name, const std::vector<std::string>& values) {
  auto column = std::make_unique<Column>();
  column->name = name;
  column->rows = values.size();
  column->type = Type::string();
  std::size_t total = 0;
  for (const auto& v : values) total += v.size();
  column->string_bytes.resize(total);
  column->values.resize(values.size() * sizeof(std::string_view));
  char* chars = reinterpret_cast<char*>(column->string_bytes.data());
  std::size_t pos = 0;
  for (std::size_t i = 0; i < values.size(); ++i) {
    std::memcpy(chars + pos, values[i].data(), values[i].size());
    column->values.as<std::string_view>()[i] = std::string_view(chars + pos, values[i].size());
    pos += values[i].size();
  }
  add_column(std::move(column));
}

Database::Database(std::string dir) : dir_(std::move(dir)) {}

Table& Database::table(const std::string& name) {
  auto it = tables_.find(name);
  if (it != tables_.end()) return *it->second;
  if (dir_.empty()) throw std::runtime_error("no table " + name);
  auto table = std::make_unique<Table>(name, dir_ + "/" + name);
  Table& ref = *table;
  tables_[name] = std::move(table);
  return ref;
}

Table& Database::add_table(const std::string& name) {
  auto table = std::make_unique<Table>(name, "");
  Table& ref = *table;
  tables_[name] = std::move(table);
  return ref;
}

Relation MaterializedTable::view() const {
  Relation r;
  r.names = names;
  r.types = types;
  r.rows = rows;
  for (const auto& c : columns) r.data.push_back(c.data());
  return r;
}

std::string MaterializedTable::to_tsv() const {
  std::string out;
  for (std::size_t c = 0; c < names.size(); ++c) out += (c ? "\t" : "") + names[c];
  out += '\n';
  for (std::size_t r = 0; r < rows; ++r) {
    for (std::size_t c = 0; c < names.size(); ++c) out += (c ? "\t" : "") + value(c, r);
    out += '\n';
  }
  return out;
}

}  // namespace morsel
