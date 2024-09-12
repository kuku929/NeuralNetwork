#pragma once
#include "dev_vector.h"
#include <memory>

std::shared_ptr<dev_vector<float>> shared_dev_vector(size_t second, size_t first,
                                                     int line = __LINE__,
                                                     std::string = __FILE_NAME__);
