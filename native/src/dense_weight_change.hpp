#pragma once

#include <iosfwd>

#include "dense_train_internal.hpp"

namespace lkjai {

DenseWeightChangeReport dense_weight_change_report(
    const DenseTrainState& before, const DenseTrainState& after);
void append_dense_weight_change_json(std::ostream& out,
                                     const DenseWeightChangeReport& report);

}  // namespace lkjai
