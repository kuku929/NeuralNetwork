#include "debug.h"

std::shared_ptr<dev_vector<float>> shared_dev_vector(size_t second, size_t first, int line,
                                                     std::string filename)
{
    return std::make_shared<dev_vector<float>>(second * first, line, filename);
}
