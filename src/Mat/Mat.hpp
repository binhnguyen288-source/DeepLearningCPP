

#pragma once
#include <numeric>
#include <algorithm>
#include <cmath>
#include <functional>
#include <fstream>
#include <vector>
#include <numeric>


using Scalar = float;
constexpr auto mat_alignment = std::align_val_t(32);
struct Size {
    int rows, cols, channels;
    
    Size(int rows, int cols, int channels) : rows{rows}, 
                                             cols{cols}, 
                                             channels{channels} {}
    ~Size()                         = default;
    Size(Size&&)                    = default;
    Size(Size const&)               = default;
    Size& operator=(Size&&)         = default;
    Size& operator=(Size const&)    = default;
    bool operator==(Size const& other) const {
        return rows == other.rows && cols == other.cols && channels == other.channels;
    }
    int total() const {
        return rows * cols * channels;
    }
};

class Mat {
public:
    Mat(int batches, Size size, bool zero_fill, bool random_fill = false);
    Mat(int batches, int rows, int cols, int channels, bool zero_fill, bool random_fill = false);
    ~Mat();
    Mat(Mat&&);
    Mat(Mat const&);
    Mat& operator=(Mat&&);
    Mat& operator=(Mat const&);

    Mat operator+(Mat const& other) const;
    Mat operator-(Mat const& other) const;
    Mat operator*(Mat const& other) const;
    Mat operator/(Mat const& other) const;
    Mat& operator+=(Mat const& other);
    Mat& operator-=(Mat const& other);
    Mat& operator*=(Mat const& other);
    Mat& operator/=(Scalar scale);

    friend Mat  operator*(Mat&& temp, Mat const& other);
    friend Mat  operator*(Scalar scale, Mat const& mat);
    friend Mat  operator-(Scalar offset, Mat const& mat);
    friend Mat  operator-(Mat const& mat, Scalar offset);
    friend Mat  operator+(Mat&& temp, Mat const& other);

    
    int           total()   const   { return size.total() * batches; }
    Scalar*       begin()           { return data; }
    Scalar*       end()             { return data + total(); }
    Scalar const* begin()   const   { return data; }
    Scalar const* end()     const   { return data + total(); }

    void dump(std::string filename);
    void load(std::string filename);
    Mat  apply(std::function<Scalar(Scalar)>&& func) const;
    Scalar& operator()(int b, int i, int j, int c)       { return data[((b * size.rows + i) * size.cols + j) * size.channels + c]; }
    Scalar  operator()(int b, int i, int j, int c) const { return data[((b * size.rows + i) * size.cols + j) * size.channels + c]; }
    Scalar& operator()(int i, int j, int c)              { return data[(i * size.cols + j) * size.channels + c]; }
    Scalar  operator()(int i, int j, int c)        const { return data[(i * size.cols + j) * size.channels + c]; }
    Scalar& operator()(int i, int j)                     { return data[i * size.cols + j]; }
    Scalar  operator()(int i, int j)               const { return data[i * size.cols + j]; }
    Scalar& operator[](int idx)                          { return data[idx]; }
    Scalar  operator[](int idx)                    const { return data[idx]; }

    Scalar sum()        const   { return std::accumulate(begin(), end   (), 0.0f); }
    Scalar max()        const   { return *std::max_element(begin(), end()); }
    int argmax()        const   { return std::distance(begin(), std::max_element(begin(), end())); }
    void fill_zero()            { std::fill(begin(), end(), 0.0f); }

public:
    Size size;
    int batches;
private:
    int capacity;
    Scalar* data;
};

void accumulate_v_mul_vT(Mat& acc, Mat const& v1, Mat const& v2);
Mat  mat_mul_vec(Mat const& mat, Mat const& vec);
Mat  matT_mul_vec(Mat const& mat, Mat const& vec);
void shuffle_data(std::vector<Mat>& inputs, std::vector<Mat>& outputs);