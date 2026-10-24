#pragma once

// Columnar storage. A table is a set of typed flat arrays loaded lazily from
// the binary column files that scripts/gen_tpch.py writes. A Relation is a
// non-owning view of columns, so base tables and intermediate results (hash
// aggregation output, sorted results) are scanned by the same code.

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "morsel/types.h"

namespace morsel {

// 64-byte aligned, growable byte buffer. Growing keeps the old contents.
class Buffer {
 public:
  Buffer() = default;
  explicit Buffer(std::size_t bytes) { resize(bytes); }
  ~Buffer();
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  void resize(std::size_t bytes);  // never shrinks the allocation
  void reserve(std::size_t bytes);
  std::size_t size() const { return size_; }
  std::uint8_t* data() { return data_; }
  const std::uint8_t* data() const { return data_; }
  template <class T>
  T* as() {
    return reinterpret_cast<T*>(data_);
  }
  template <class T>
  const T* as() const {
    return reinterpret_cast<const T*>(data_);
  }

 private:
  std::uint8_t* data_ = nullptr;
  std::size_t size_ = 0;
  std::size_t capacity_ = 0;
};

struct Relation {
  std::vector<std::string> names;
  std::vector<Type> types;
  std::vector<const void*> data;
  std::size_t rows = 0;

  int index_of(std::string_view name) const;  // -1 if absent
};

struct Column {
  std::string name;
  Type type;
  std::size_t rows = 0;
  Buffer values;         // rows x type_width(type.id)
  Buffer string_bytes;   // String columns: the characters `values` points into
  std::unique_ptr<Dictionary> dict;
};

class Table {
 public:
  Table(std::string name, std::string dir);  // dir empty: in-memory table
  const std::string& name() const { return name_; }
  std::size_t rows() const { return rows_; }

  // Loads the column on first use. Not thread-safe: plans load their columns
  // when they are built, before any query runs.
  const Column& column(const std::string& name);
  bool has_column(const std::string& name) const;

  // For tests and in-memory tables.
  void add_column(std::unique_ptr<Column> column);
  template <class T>
  void add_column(const std::string& name, Type type, const std::vector<T>& values);
  void add_dict_column(const std::string& name, const std::vector<std::string>& values);
  void add_string_column(const std::string& name, const std::vector<std::string>& values);

 private:
  std::string name_;
  std::string dir_;
  std::size_t rows_ = 0;
  bool rows_known_ = false;
  std::map<std::string, std::unique_ptr<Column>> columns_;
};

class Database {
 public:
  explicit Database(std::string dir = "");
  Table& table(const std::string& name);
  Table& add_table(const std::string& name);  // in-memory

 private:
  std::string dir_;
  std::map<std::string, std::unique_ptr<Table>> tables_;
};

// Owns the columns of an intermediate or final result.
struct MaterializedTable {
  std::vector<std::string> names;
  std::vector<Type> types;
  std::vector<Buffer> columns;
  std::size_t rows = 0;

  Relation view() const;
  std::string value(std::size_t column, std::size_t row) const {
    return format_value(types[column], columns[column].data(), row);
  }
  // Tab-separated, header first: the format the correctness check reads.
  std::string to_tsv() const;
};

std::unique_ptr<Column> load_column(const std::string& path, const std::string& name);

template <class T>
void Table::add_column(const std::string& name, Type type, const std::vector<T>& values) {
  auto column = std::make_unique<Column>();
  column->name = name;
  column->type = type;
  column->rows = values.size();
  column->values.resize(values.size() * sizeof(T));
  for (std::size_t i = 0; i < values.size(); ++i) column->values.as<T>()[i] = values[i];
  add_column(std::move(column));
}

}  // namespace morsel
