#include "Mat.hpp"
#include <algorithm>



Mat::Mat(Mat&& other) : size(other.size),
                        batches(other.batches),
                        capacity{other.capacity},
                        data{other.data} {
    other.data = nullptr;
}

Mat::Mat(Mat const& other) : Mat(other.batches, 
                                 other.size.rows, 
                                 other.size.cols, 
                                 other.size.channels, false) {
    std::copy(other.begin(), other.end(), data);
}

Mat& Mat::operator=(Mat&& other) {
    if (this == &other) [[unlikely]] return *this;
    if (data) {
        operator delete[](data, mat_alignment);
    }
    size = other.size;
    batches = other.batches;
    capacity = other.capacity;
    data = other.data;
    other.data = nullptr;
    return *this;
}

Mat& Mat::operator=(Mat const& other) {
    if (this == &other) [[unlikely]] return *this;
    if (!data || capacity < other.total()) {
        if (data) operator delete[](data, mat_alignment);
        capacity = other.total();
        data = new (mat_alignment) Scalar[capacity];
    }
    size = other.size;
    batches = other.batches;
    std::copy(other.begin(), other.end(), data);
    return *this;
}
#include <cassert>
Mat Mat::operator+(Mat const& other) const {
    assert(other.size == size && batches == other.batches);
    Mat result(batches, size, false);
    for (int i = 0; i < total(); ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}


Mat Mat::operator-(Mat const& other) const  {
    
    assert(other.size == size && batches == other.batches);
    Mat result(batches, size, false);
    for (int i = 0; i < total(); ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

Mat Mat::operator*(Mat const& other) const  {
    
    assert(other.size == size && batches == other.batches);
    Mat result(batches, size, false);
    for (int i = 0; i < total(); ++i) {
        result.data[i] = data[i] * other.data[i];
    }
    return result;
}

Mat Mat::operator/(Mat const& other) const {
    assert(other.size == size && batches == other.batches);
    Mat result(batches, size, false);
    for (int i = 0; i < total(); ++i) {
        result.data[i] = data[i] / other.data[i];
    }
    return result;
}



Mat& Mat::operator+=(Mat const& other) {
    
    assert(other.size == size && batches == other.batches);
    for (int i = 0; i < other.total(); ++i) {
        data[i] += other.data[i];
    }
    return *this;
}


Mat& Mat::operator-=(Mat const& other) {

    
    assert(other.size == size && batches == other.batches);
    
    for (int i = 0; i < other.total(); ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

Mat& Mat::operator*=(Mat const& other) {
    
    assert(other.size == size && batches == other.batches);
    for (int i = 0; i < other.total(); ++i) {
        data[i] *= other.data[i];
    }
    return *this;
}

Mat& Mat::operator/=(Scalar scale)  {
    
    for (int i = 0; i < total(); ++i) {
        data[i] /= scale;
    }
    return *this;
}

void Mat::dump(std::string filename) {
    FILE* file = fopen(filename.c_str(), "wb");
    fwrite(begin(), sizeof(Scalar), total(), file);
    fclose(file);
}
void Mat::load(std::string filename) {
    FILE* file = fopen(filename.c_str(), "rb");
    fread(begin(), sizeof(Scalar), total(), file);
    fclose(file);
}


Mat Mat::apply(std::function<Scalar(Scalar)>&& func) const {
    Mat result(batches, size, false);
    std::transform(begin(), end(), result.begin(), func);
    return result;
}


// non-member functions

Mat operator+(Mat&& temp, Mat const& other) {
    
    assert(other.size == temp.size && other.batches == temp.batches);
    for (int i = 0; i < other.total(); ++i) {
        temp.data[i] += other.data[i];
    }
    return temp;
}


Mat operator*(Mat&& temp, Mat const& other)  {
    
    assert(other.size == temp.size && other.batches == temp.batches);
    for (int i = 0; i < other.total(); ++i) {
        temp.data[i] *= other.data[i];
    }
    return temp;
}


Mat operator*(Scalar scale, Mat const& mat) {
    
    Mat result(mat.batches, mat.size, false);
    for (int i = 0; i < mat.total(); ++i) {
        result.data[i] = scale * mat.data[i];
    }
    return result;
}

Mat operator-(Scalar offset, Mat const& mat)  {
    Mat result(mat.batches, mat.size, false);
    for (int i = 0; i < mat.total(); ++i) {
        result.data[i] = offset - mat.data[i];
    }
    return result;
}

Mat operator-(Mat const& mat, Scalar offset)  {
    Mat result(mat.batches, mat.size, false);
    for (int i = 0; i < mat.total(); ++i) {
        result.data[i] = mat.data[i] - offset;
    }
    return result;
}


void accumulate_v_mul_vT(Mat& acc, Mat const& v1, Mat const& v2) {

    assert(v1.size.cols == 1 && v2.size.cols == 1 && 1 == v2.batches && 1 == v1.batches);
    for (int i = 0; i < v1.total(); ++i) {
        for (int j = 0; j < v2.total(); ++j) {
            acc[i * v2.total() + j] += v1[i] * v2[j];
        }
    }
}

Mat mat_mul_vec(Mat const& mat, Mat const& vec) {
    assert(mat.size.cols == vec.size.rows && vec.batches == 1 && mat.batches == 1);
    Mat result(1, mat.size.rows, 1, 1, false);
    for (int i = 0; i < mat.size.rows; ++i) {
        result[i] = std::transform_reduce(vec.begin(), vec.end(), 
                                          mat.begin() + i * vec.total(), static_cast<Scalar>(0));
    }
    return result;
}

Mat matT_mul_vec(Mat const& mat, Mat const& vec) {
    assert(mat.size.rows == vec.size.rows && vec.batches == 1 && mat.batches == 1);
    Mat result(1, mat.size.cols, 1, 1, true);

    for (int k = 0; k < mat.size.rows; ++k) { 
        for (int i = 0; i < mat.size.cols; ++i) {
            result[i] += mat(k, i) * vec[k];
        }
    }
        
    return result;
}

#include <random>
static Scalar get_random_scalar() {
    static std::mt19937_64 rng((std::random_device())());
    static std::uniform_real_distribution dist(static_cast<Scalar>(-.03), 
                                               static_cast<Scalar>(.03));
    return dist(rng);
}



Mat::Mat(int batches, Size size, bool zero_fill, bool random_fill) :
    size(size), batches{batches}, capacity{batches * size.total()},
    data{zero_fill ? new (mat_alignment) Scalar[batches * size.total()]() :
                     new (mat_alignment) Scalar[batches * size.total()]} 
{
    if (random_fill)
        std::for_each(begin(), end(), [](Scalar& x) { x = get_random_scalar(); });
}

Mat::Mat(int batches, int rows, int cols, int channels, bool zero_fill, bool random_fill) : 
    size(rows, cols, channels), batches{batches}, capacity{batches * channels * rows * cols},
    data{zero_fill ? new (mat_alignment) Scalar[batches * channels * rows * cols]() :
                     new (mat_alignment) Scalar[batches * channels * rows * cols]} 
{
    if (random_fill)
        std::for_each(begin(), end(), [](Scalar& x) { x = get_random_scalar(); });
}

Mat::~Mat() {
    if (data) operator delete[](data, mat_alignment);
}



void shuffle_data(std::vector<Mat>& inputs, std::vector<Mat>& outputs) {
    std::vector<int> idx(inputs.size());
    std::generate(idx.begin(), idx.end(), [x = 0]() mutable { return x++; });
    std::shuffle(idx.begin(), idx.end(), std::mt19937((std::random_device())()));
    std::vector<Mat> new_inputs, new_outputs;
    for (int i : idx) {
        new_inputs.push_back(std::move(inputs[i]));
        new_outputs.push_back(std::move(outputs[i]));
    }
    inputs = std::move(new_inputs);
    outputs = std::move(new_outputs);
}