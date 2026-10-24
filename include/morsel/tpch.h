#pragma once

// Hand-written physical plans for TPC-H queries, with the standard
// substitution parameters (the ones DuckDB's tpch extension uses).

#include <vector>

#include "morsel/plan.h"
#include "morsel/storage.h"

namespace morsel {

const std::vector<int>& tpch_supported_queries();

// Builds the plan, loading the columns it needs. Throws for unsupported queries.
Plan tpch_plan(Database& db, int query);

}  // namespace morsel
