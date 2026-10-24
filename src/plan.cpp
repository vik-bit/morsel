#include "morsel/plan.h"

#include <stdexcept>

namespace morsel {

namespace {

Agg make_agg(AggKind kind, const E& x, const std::string& name) { return {kind, x.ast, name}; }

std::shared_ptr<PlanNode> derive(const std::shared_ptr<PlanNode>& child, PlanNode::Kind kind) {
  auto node = std::make_shared<PlanNode>();
  node->kind = kind;
  node->children.push_back(child);
  return node;
}

bool is_key_type(const Type& t) {
  return t.id == TypeId::Int32 || t.id == TypeId::Date || t.id == TypeId::Int64 ||
         t.id == TypeId::Dict;
}

void check_key_width(const Schema& schema, const std::vector<int>& cols, const char* what) {
  std::size_t bytes = 0;
  for (int c : cols) {
    const Type& t = schema.types[static_cast<std::size_t>(c)];
    if (!is_key_type(t)) {
      throw std::invalid_argument(std::string(what) + " column " + schema.names[static_cast<std::size_t>(c)] +
                                  " must be an integer, date or dictionary column");
    }
    bytes += type_width(t.id);
  }
  if (bytes > 16) throw std::invalid_argument(std::string(what) + " keys wider than 16 bytes");
}

void add_column(Schema& schema, const std::string& name, const Type& type) {
  if (schema.index_of(name) >= 0) throw std::invalid_argument("duplicate column name " + name);
  schema.names.push_back(name);
  schema.types.push_back(type);
}

}  // namespace

Agg sum(const E& x, const std::string& name) { return make_agg(AggKind::Sum, x, name); }
Agg avg(const E& x, const std::string& name) { return make_agg(AggKind::Avg, x, name); }
Agg min(const E& x, const std::string& name) { return make_agg(AggKind::Min, x, name); }
Agg max(const E& x, const std::string& name) { return make_agg(AggKind::Max, x, name); }
Agg any_value(const E& x, const std::string& name) { return make_agg(AggKind::Any, x, name); }
Agg count_star(const std::string& name) { return {AggKind::Count, nullptr, name}; }

Plan Plan::scan(Database& db, const std::string& table, const std::vector<std::string>& columns) {
  Table& t = db.table(table);
  Relation r;
  for (const auto& name : columns) {
    const Column& c = t.column(name);
    r.names.push_back(name);
    r.types.push_back(c.type);
    r.data.push_back(c.values.data());
    r.rows = c.rows;
  }
  return scan(r, table);
}

Plan Plan::scan(const Relation& relation, const std::string& label) {
  auto node = std::make_shared<PlanNode>();
  node->kind = PlanNode::Scan;
  node->relation = relation;
  node->schema.names = relation.names;
  node->schema.types = relation.types;
  node->label = label;
  return Plan{node};
}

Plan Plan::filter(const P& pred) const {
  auto n = derive(node, PlanNode::Filter);
  n->schema = schema();
  n->pred = bind_pred(*pred.ast, schema());
  return Plan{n};
}

Plan Plan::project(const std::vector<std::pair<std::string, E>>& exprs) const {
  auto n = derive(node, PlanNode::Project);
  for (const auto& [name, e] : exprs) {
    auto bound = std::shared_ptr<BoundExpr>(bind_expr(*e.ast, schema()));
    add_column(n->schema, name, bound->type);
    n->exprs.push_back(std::move(bound));
  }
  return Plan{n};
}

Plan Plan::select(const std::vector<std::string>& columns) const {
  std::vector<std::pair<std::string, E>> exprs;
  for (const auto& c : columns) exprs.emplace_back(c, col(c));
  return project(exprs);
}

Plan Plan::join(const Plan& build, const std::vector<std::string>& probe_keys,
                const std::vector<std::string>& build_keys, JoinType type) const {
  if (probe_keys.empty() || probe_keys.size() != build_keys.size()) {
    throw std::invalid_argument("join needs the same number of probe and build keys");
  }
  auto n = derive(node, PlanNode::Join);
  n->children.push_back(build.node);
  n->join_type = type;
  const Schema& ps = schema();
  const Schema& bs = build.schema();
  for (std::size_t k = 0; k < probe_keys.size(); ++k) {
    const int p = ps.require(probe_keys[k]);
    const int b = bs.require(build_keys[k]);
    const Type& pt = ps.types[static_cast<std::size_t>(p)];
    const Type& bt = bs.types[static_cast<std::size_t>(b)];
    if (pt.id != bt.id || pt.scale != bt.scale || (pt.id == TypeId::Dict && pt.dict != bt.dict)) {
      throw std::invalid_argument("join key types differ: " + probe_keys[k] + " vs " + build_keys[k]);
    }
    n->probe_keys.push_back(p);
    n->build_keys.push_back(b);
  }
  check_key_width(ps, n->probe_keys, "join");
  n->schema = ps;
  if (type == JoinType::Inner) {
    for (std::size_t c = 0; c < bs.size(); ++c) {
      bool is_key = false;
      for (int b : n->build_keys) is_key |= b == static_cast<int>(c);
      if (is_key) continue;
      n->build_payload.push_back(static_cast<int>(c));
      add_column(n->schema, bs.names[c], bs.types[c]);
    }
  }
  return Plan{n};
}

Plan Plan::aggregate(const std::vector<std::string>& group_by, const std::vector<Agg>& aggs) const {
  // A projection first computes the aggregate inputs as columns, so the
  // aggregation itself only reads columns. Plain column references pass
  // through without copying.
  std::vector<std::pair<std::string, E>> inputs;
  for (const auto& g : group_by) inputs.emplace_back(g, col(g));
  for (std::size_t a = 0; a < aggs.size(); ++a) {
    if (aggs[a].input) inputs.emplace_back("__agg" + std::to_string(a), E(aggs[a].input));
  }
  const Plan input = project(inputs);
  const Schema& is = input.schema();

  auto n = derive(input.node, PlanNode::Aggregate);
  for (std::size_t g = 0; g < group_by.size(); ++g) {
    n->group_cols.push_back(static_cast<int>(g));
    add_column(n->schema, group_by[g], is.types[g]);
  }
  check_key_width(is, n->group_cols, "group-by");
  int next_input = static_cast<int>(group_by.size());
  for (const auto& a : aggs) {
    BoundAgg b;
    b.kind = a.kind;
    b.name = a.name;
    if (a.input) {
      b.input = next_input++;
      b.input_type = is.types[static_cast<std::size_t>(b.input)];
    }
    const Type in = b.input_type;
    switch (a.kind) {
      case AggKind::Count:
        b.output_type = Type::int64();
        break;
      case AggKind::Sum:
        if (in.id == TypeId::Double) {
          b.output_type = Type::float64();
        } else if (in.is_integer()) {
          b.output_type = Type::int128(in.scale);
        } else {
          throw std::invalid_argument("sum needs a numeric input: " + a.name);
        }
        break;
      case AggKind::Avg:
        if (!in.is_numeric()) throw std::invalid_argument("avg needs a numeric input: " + a.name);
        b.output_type = Type::float64();
        break;
      case AggKind::Min:
      case AggKind::Max:
        if (in.id == TypeId::String || in.id == TypeId::Int128) {
          throw std::invalid_argument("min/max not supported for " + type_name(in));
        }
        b.output_type = in;
        break;
      case AggKind::Any:
        b.output_type = in;
        break;
    }
    add_column(n->schema, a.name, b.output_type);
    n->aggs.push_back(b);
  }
  return Plan{n};
}

Plan Plan::order_by(const std::vector<SortKey>& keys, std::size_t limit) const {
  auto n = derive(node, PlanNode::OrderBy);
  n->schema = schema();
  for (const auto& k : keys) {
    n->sort_cols.push_back(schema().require(k.column));
    n->sort_desc.push_back(k.descending);
  }
  n->limit = limit;
  return Plan{n};
}

namespace {

void explain_node(const PlanNode& n, int depth, std::string& out) {
  std::string line(static_cast<std::size_t>(depth) * 2, ' ');
  auto names = [&](const std::vector<int>& cols, const Schema& s) {
    std::string r;
    for (int c : cols) r += (r.empty() ? "" : ", ") + s.names[static_cast<std::size_t>(c)];
    return r;
  };
  switch (n.kind) {
    case PlanNode::Scan:
      line += "Scan " + n.label + " (" + std::to_string(n.relation.rows) + " rows, " +
              std::to_string(n.schema.size()) + " columns)";
      break;
    case PlanNode::Filter:
      line += "Filter";
      break;
    case PlanNode::Project:
      line += "Project [";
      for (std::size_t i = 0; i < n.schema.size(); ++i) line += (i ? ", " : "") + n.schema.names[i];
      line += "]";
      break;
    case PlanNode::Join: {
      const char* type = n.join_type == JoinType::Inner ? "Inner" : n.join_type == JoinType::Semi ? "Semi" : "Anti";
      line += std::string("HashJoin ") + type + " probe(" + names(n.probe_keys, n.children[0]->schema) +
              ") = build(" + names(n.build_keys, n.children[1]->schema) + ")";
      break;
    }
    case PlanNode::Aggregate:
      line += "HashAggregate by [" + names(n.group_cols, n.children[0]->schema) + "] -> " +
              std::to_string(n.aggs.size()) + " aggregates";
      break;
    case PlanNode::OrderBy:
      line += "OrderBy [" + names(n.sort_cols, n.schema) + "]";
      if (n.limit) line += " limit " + std::to_string(n.limit);
      break;
  }
  out += line + "\n";
  for (const auto& c : n.children) explain_node(*c, depth + 1, out);
}

}  // namespace

std::string Plan::explain() const {
  std::string out;
  explain_node(*node, 0, out);
  return out;
}

}  // namespace morsel
