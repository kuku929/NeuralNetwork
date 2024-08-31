#ifndef DIMENSION_H 
#define DIMENSION_H
namespace nnet{
class Dimension{
public:
	Dimension(): first(0), second(0){};
	Dimension(size_t first, size_t second): 
		first(first),
		second(second){};
	Dimension(size_t first): 
		first(first),
		second(first){};
	size_t first;
	size_t second;
};
}
#endif //DIMENSION_H
