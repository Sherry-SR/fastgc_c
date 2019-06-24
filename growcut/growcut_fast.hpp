#include <numeric>
#include <math.h>
#include <vector>

//#include "boost/heap/fibonacci_heap.hpp"
#include "pybind11/pybind11.h"
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xindex_view.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xmasked_view.hpp"
#include "myutils/fibonacci.hpp"

using namespace std;
namespace py = pybind11;

