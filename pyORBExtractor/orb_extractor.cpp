#include <pybind11/pybind11.h>

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "opencv_type_casters.h"
#include "ORBextractor.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace ORB_SLAM2;


PYBIND11_MODULE(pyORBExtractor, m)
{

    m.doc() = "pybind11 plugin for ORBSLAM2 features";

    // bindings to ORBextractor class
    py::class_<ORBextractor>(m, "ORBextractor")
        .def(py::init<int, float, int, int, int>(),"nfeatures"_a, "scaleFactor"_a, "nlevels"_a, "iniThFAST"_a, "minThFAST"_a)
        .def("GetLevels", &ORBextractor::GetLevels)
        .def("GetScaleFactor", &ORBextractor::GetScaleFactor)
        .def("GetInverseScaleFactors", &ORBextractor::GetInverseScaleFactors)
        .def("GetScaleSigmaSquares", &ORBextractor::GetScaleSigmaSquares)
        .def("GetInverseScaleSigmaSquares", &ORBextractor::GetInverseScaleSigmaSquares)
        .def("operator_kd", [](ORBextractor& o, cv::Mat& image)
            {
                cv::Mat mask = cv::Mat();  // input mask is not actually used by the implementation
                std::vector<cv::KeyPoint> keypoints;
                cv::Mat descriptors;
                o.operator_kd(image, mask, keypoints, descriptors);
                return std::make_tuple(keypoints, descriptors);
            });

}
