#include "adaptive_manifold_filter.hpp"
#include <cmath>
#include <limits>
#include <opencv2/core/internal.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

namespace cv
{
    CV_INIT_ALGORITHM(AdaptiveManifoldFilter, "AdaptiveManifoldFilter",
                      obj.info()->addParam(obj, "sigma_s", obj.sigma_s_, false, 0, 0, "Filter spatial standard deviation");
                      obj.info()->addParam(obj, "sigma_r", obj.sigma_r_, false, 0, 0, "Filter range standard deviation");
                      obj.info()->addParam(obj, "tree_height", obj.tree_height_, false, 0, 0, "Height of the manifold tree (default = -1 : automatically computed)");
                      obj.info()->addParam(obj, "num_pca_iterations", obj.num_pca_iterations_, false, 0, 0, "Number of iterations to computed the eigenvector v1"));
}

cv::AdaptiveManifoldFilter::AdaptiveManifoldFilter()
{
    sigma_s_ = 16.0;
    sigma_r_ = 0.2;
    tree_height_ = -1;
    num_pca_iterations_ = 1;
}

namespace
{
    double Log2( double n )
    {
        return log(n) / log(2.0);
    }

    int computeManifoldTreeHeight(double sigma_s, double sigma_r)
    {
        const double Hs = floor(Log2(sigma_s)) - 1.0;
        const double Lr = 1.0 - sigma_r;
        return max(2, static_cast<int>(ceil(Hs * Lr)));
    }

    template <typename T>
    Mat_<T> h_filter_horizontal(const Mat_<T>& src, double sigma)
    {
        CV_Assert( src.depth() == CV_64F );

        const double a = exp(-sqrt(2.0) / sigma);

        Mat_<T> dst = src.clone();

        for (int y = 0; y < src.rows; ++y)
        {
            const T* src_row = src[y];
            T* dst_row = dst[y];

            for (int x = 1; x < src.cols; ++x)
            {
                dst_row[x] = src_row[x] + a * (src_row[x - 1] - src_row[x]);
            }

            for (int x = src.cols - 2; x >= 0; --x)
            {
                dst_row[x] = dst_row[x] + a * (dst_row[x + 1] - dst_row[x]);
            }
        }

        return dst;
    }

    template <typename T>
    Mat_<T> h_filter(const Mat_<T> src, double sigma)
    {
        Mat_<T> dst = h_filter_horizontal(src, sigma);
        dst = dst.t();
        dst = h_filter_horizontal(dst, sigma);
        return dst.t();
    }

    template <typename T>
    Mat_<T> rdivide(const Mat_<T>& a, const Mat_<double>& b)
    {
        CV_Assert( a.depth() == CV_64F );
        CV_Assert( a.size() == b.size() );

        Mat_<T> dst(a.size());

        for (int y = 0; y < a.rows; ++y)
        {
            const T* a_row = a[y];
            const double* b_row = b[y];
            T* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                dst_row[x] = a_row[x] * (1.0 / b_row[x]);
            }
        }

        return dst;
    }

    template <typename T>
    Mat_<T> times(const Mat_<T>& a, const Mat_<double>& b)
    {
        CV_Assert( a.depth() == CV_64F );
        CV_Assert( a.size() == b.size() );

        Mat_<T> dst(a.size());

        for (int y = 0; y < a.rows; ++y)
        {
            const T* a_row = a[y];
            const double* b_row = b[y];
            T* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                dst_row[x] = a_row[x] * b_row[x];
            }
        }

        return dst;
    }
}

void cv::AdaptiveManifoldFilter::apply(InputArray src, OutputArray _dst, OutputArray _tilde_dst, InputArray src_joint)
{
    CV_Assert( src.type() == CV_8UC3 );
    CV_Assert( src_joint.empty() || (src_joint.type() == src.type() && src_joint.size() == src.size()) );

    src.getMat().convertTo(src_f_, src_f_.type(), 1.0 / 255.0);

    // Use the center pixel as seed to random number generation.
    const Point3d centralPix = src_f_(src_f_.rows / 2, src_f_.cols / 2);
    rng_ = RNG(static_cast<uint64>(centralPix.ddot(centralPix) * numeric_limits<uint64>::max()));

    sum_w_ki_Psi_blur_.create(src_f_.size());
    sum_w_ki_Psi_blur_.setTo(Scalar::all(0));

    sum_w_ki_Psi_blur_0_.create(src_f_.size());
    sum_w_ki_Psi_blur_0_.setTo(Scalar::all(0));

    min_pixel_dist_to_manifold_squared_.create(src_f_.size());
    min_pixel_dist_to_manifold_squared_.setTo(Scalar::all(numeric_limits<double>::max()));

    // If the tree_height was not specified, compute it using Eq. (10) of our paper.
    cur_tree_height_ = tree_height_ > 0 ? tree_height_ : computeManifoldTreeHeight(sigma_s_, sigma_r_);

    // If no joint signal was specified, use the original signal
    if (src_joint.empty())
        src_f_.copyTo(src_joint_f_);
    else
        src_joint.getMat().convertTo(src_joint_f_, CV_64F, 1.0 / 255.0);

    // Algorithm 1, Step 1: compute the first manifold by low-pass filtering.
    const Mat_<Point3d> eta_1 = h_filter(src_joint_f_, sigma_s_);
    const Mat_<uchar> cluster_1(src_f_.size(), 1);

    buildManifoldsAndPerformFiltering(eta_1, cluster_1, 1);

    // Compute the filter response by normalized convolution -- Eq. (4)
    const Mat_<Point3d> tilde_dst = rdivide(sum_w_ki_Psi_blur_, sum_w_ki_Psi_blur_0_);

    // Adjust the filter response for outlier pixels -- Eq. (10)
    Mat_<double> alpha;
    exp(min_pixel_dist_to_manifold_squared_ * (-0.5 / sigma_r_ / sigma_r_), alpha);

    const Mat_<Point3d> diff = tilde_dst - src_f_;
    const Mat_<Point3d> dst = src_f_ + times(diff, alpha);

    dst.convertTo(_dst, CV_8U, 255.0);
    if (_tilde_dst.needed())
        tilde_dst.convertTo(_tilde_dst, CV_8U, 255.0);
}

namespace
{
    double floor_to_power_of_two(double r)
    {
        return pow(2.0, floor(Log2(r)));
    }

    Mat_<double> channelsSum(const Mat_<Point3d>& src)
    {
        Mat_<double> dst(src.size(), 0.0);

        for (int y = 0; y < src.rows; ++y)
        {
            const Point3d* src_row = src[y];
            double* dst_row = dst[y];

            for (int x = 0; x < src.cols; ++x)
            {
                Point3d src_val = src_row[x];
                dst_row[x] = src_val.x + src_val.y + src_val.z;
            }
        }

        return dst;
    }

    Mat_<double> phi(const Mat_<double>& src, double sigma)
    {
        Mat_<double> dst(src.size());

        for (int y = 0; y < dst.rows; ++y)
        {
            const double* src_row = src[y];
            double* dst_row = dst[y];

            for (int x = 0; x < dst.cols; ++x)
            {
                double src_val = src_row[x];
                double dst_val = exp(-0.5 * src_val / sigma / sigma);
                dst_row[x] = dst_val;
            }
        }

        return dst;
    }

    Mat_<Vec4d> catCn(const Mat_<Point3d>& a, const Mat_<double>& b)
    {
        Mat_<Vec4d> dst(a.size());
        CV_Assert( a.size() == b.size() );

        for (int y = 0; y < a.rows; ++y)
        {
            const Point3d* a_row = a[y];
            const double* b_row = b[y];
            Vec4d* dst_row = dst[y];

            for (int x = 0; x < a.cols; ++x)
            {
                Point3d a_val = a_row[x];
                double b_val = b_row[x];

                dst_row[x] = Vec4d(a_val.x, a_val.y, a_val.z, b_val);
            }
        }

        return dst;
    }

    Mat_<Point3d> diffY(const Mat_<Point3d>& src)
    {
        Mat_<Point3d> dst(src.rows - 1, src.cols);

        for (int y = 0; y < src.rows - 1; ++y)
        {
            const Point3d* src_cur_row = src[y];
            const Point3d* src_next_row = src[y + 1];
            Point3d* dst_row = dst[y];

            for (int x = 0; x < src.cols; ++x)
            {
                dst_row[x] = src_next_row[x] - src_cur_row[x];
            }
        }

        return dst;
    }

    Mat_<Point3d> diffX(const Mat_<Point3d>& src)
    {
        Mat_<Point3d> dst(src.rows, src.cols - 1);

        for (int y = 0; y < src.rows; ++y)
        {
            const Point3d* src_row = src[y];
            Point3d* dst_row = dst[y];

            for (int x = 0; x < src.cols - 1; ++x)
            {
                dst_row[x] = src_row[x + 1] - src_row[x];
            }
        }

        return dst;
    }

    Mat_<Vec4d> TransformedDomainRecursiveFilter_Horizontal(const Mat_<Vec4d>& I, const Mat_<double>& D, double sigma)
    {
        CV_Assert( I.size() == D.size() );

        const double a = exp(-sqrt(2.0) / sigma);

        Mat_<Vec4d> F = I.clone();

        Mat_<double> V(D.size());
        for (int y = 0; y < D.rows; ++y)
        {
            const double* D_row = D[y];
            double* V_row = V[y];

            for (int x = 0; x < D.cols; ++x)
            {
                V_row[x] = pow(a, D_row[x]);
            }
        }

        for (int y = 0; y < I.rows; ++y)
        {
            const double* V_row = V[y];
            Vec4d* F_row = F[y];

            for (int x = 1; x < I.cols; ++x)
            {
                Vec4d F_prev_val = F_row[x - 1];
                Vec4d F_cur_val = F_row[x];

                F_cur_val[0] += V_row[x] * (F_prev_val[0] - F_cur_val[0]);
                F_cur_val[1] += V_row[x] * (F_prev_val[1] - F_cur_val[1]);
                F_cur_val[2] += V_row[x] * (F_prev_val[2] - F_cur_val[2]);
                F_cur_val[3] += V_row[x] * (F_prev_val[3] - F_cur_val[3]);

                F_row[x] = F_cur_val;
            }
        }

        for (int y = 0; y < I.rows; ++y)
        {
            const double* V_row = V[y];
            Vec4d* F_row = F[y];

            for (int x = I.cols - 2; x >= 0; --x)
            {
                Vec4d F_prev_val = F_row[x + 1];
                Vec4d F_cur_val = F_row[x];

                F_cur_val[0] += V_row[x] * (F_prev_val[0] - F_cur_val[0]);
                F_cur_val[1] += V_row[x] * (F_prev_val[1] - F_cur_val[1]);
                F_cur_val[2] += V_row[x] * (F_prev_val[2] - F_cur_val[2]);
                F_cur_val[3] += V_row[x] * (F_prev_val[3] - F_cur_val[3]);

                F_row[x] = F_cur_val;
            }
        }

        return F;
    }

    Mat_<Vec4d> RF_filter(const Mat_<Vec4d>& src, const Mat_<Point3d>& src_joint, double sigma_s, double sigma_r)
    {
        CV_Assert( src_joint.size() == src.size() );

        const Mat_<Point3d> dIcdx = diffX(src_joint);
        const Mat_<Point3d> dIcdy = diffY(src_joint);

        Mat_<double> dIdx(src.size());
        Mat_<double> dIdy(src.size());
        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 1; x < src.cols; ++x)
            {
                const Point3d val = dIcdx(y, x - 1);
                dIdx.at<double>(y, x) = val.ddot(val);
            }
        }
        for (int y = 1; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                const Point3d val = dIcdy(y - 1, x);
                dIdy.at<double>(y, x) = val.ddot(val);
            }
        }

        Mat_<double> dHdx;
        dIdx.convertTo(dHdx, dHdx.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
        sqrt(dHdx, dHdx);

        Mat_<double> dVdy;
        dIdy.convertTo(dVdy, dVdy.type(), (sigma_s / sigma_r) * (sigma_s / sigma_r), (sigma_s / sigma_s) * (sigma_s / sigma_s));
        sqrt(dVdy, dVdy);
        dVdy = dVdy.t();

        Mat_<Vec4d> F = src.clone();

        F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_s);
        F = F.t();

        F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_s);
        F = F.t();

        return F;
    }

    void split_3_1(const Mat_<Vec4d>& src, Mat_<Point3d>& dst1, Mat_<double>& dst2)
    {
        dst1.create(src.size());
        dst2.create(src.size());

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                Vec4d val = src(y, x);
                dst1(y, x) = Point3d(val[0], val[1], val[2]);
                dst2(y, x) = val[3];
            }
        }
    }

    Mat_<double> computeEigenVector(const Mat_<double>& X, const Mat_<uchar>& mask, int num_pca_iterations, const Mat_<double>& rand_vec)
    {
        CV_Assert( X.cols == rand_vec.cols );
        CV_Assert( X.rows == mask.size().area() );
        CV_Assert( rand_vec.rows == 1 );

        Mat_<double> p = rand_vec.clone();

        for (int i = 0; i < num_pca_iterations; ++i)
        {
            Mat_<double> t(X.size(), 0.0);

            for (int y = 0; y < mask.rows; ++y)
            {
                for (int x = 0; x < mask.cols; ++x)
                {
                    if (mask(y, x))
                    {
                        int ind = y * mask.cols + x;

                        double dots = 0.0;
                        for (int c = 0; c < X.cols; ++c)
                            dots += p(0, c) * X(ind, c);

                        for (int c = 0; c < X.cols; ++c)
                            t(ind, c) = dots * X(ind, c);
                    }
                }
            }

            p.setTo(0.0);
            for (int i = 0; i < X.rows; ++i)
                for (int c = 0; c < X.cols; ++c)
                    p(0, c) += t(i, c);
        }

        return p / norm(p);
    }

    Mat_<Point3d> calcEta(const Mat_<Point3d>& src_joint_f, const Mat_<double>& theta, const Mat_<uchar>& cluster, double sigma_s, double df)
    {
        Mat_<double> theta_masked(theta.size(), 0.0);
        theta.copyTo(theta_masked, cluster);

        Mat_<Point3d> numerator;
        resize(times(src_joint_f, theta_masked), numerator, Size(), 1.0 / df, 1.0 / df);

        Mat_<double> denominator;
        resize(theta_masked, denominator, Size(), 1.0 / df, 1.0 / df);

        return rdivide(h_filter(numerator, sigma_s / df), h_filter(denominator, sigma_s / df));
    }
}

void cv::AdaptiveManifoldFilter::buildManifoldsAndPerformFiltering(const Mat_<Point3d>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level)
{
    // Dividing the covariance matrix by 2 is equivalent to dividing the standard deviations by sqrt(2).

    const double sigma_r_over_sqrt_2 = sigma_r_ / sqrt(2.0);

    // Compute downsampling factor

    double df = min(sigma_s_ / 4.0, 256.0 * sigma_r_);
    df = floor_to_power_of_two(df);
    df = max(1.0, df);

    // Splatting: project the pixel values onto the current manifold eta_k

    Mat_<Point3d> X;
    Mat_<Point3d> eta_k_small;

    if (eta_k.rows == src_joint_f_.rows)
    {
        subtract(src_joint_f_, eta_k, X);
        resize(eta_k, eta_k_small, Size(), 1.0 / df, 1.0 / df);
    }
    else
    {
        eta_k_small = eta_k;
        Mat eta_k_big;
        resize(eta_k, eta_k_big, src_joint_f_.size());
        subtract(src_joint_f_, eta_k_big, X);
    }

    // Project pixel colors onto the manifold -- Eq. (3), Eq. (5)

    Mat_<Point3d> X_squared;
    multiply(X, X, X_squared);
    const Mat_<double> pixel_dist_to_manifold_squared = channelsSum(X_squared);

    const Mat_<double> gaussian_distance_weights = phi(pixel_dist_to_manifold_squared, sigma_r_over_sqrt_2);

    const Mat_<Point3d> Psi_splat = times(src_f_, gaussian_distance_weights);
    const Mat_<double> Psi_splat_0 = gaussian_distance_weights;

    // Save min distance to later perform adjustment of outliers -- Eq. (10)

    min(min_pixel_dist_to_manifold_squared_, pixel_dist_to_manifold_squared, min_pixel_dist_to_manifold_squared_);

    // Blurring: perform filtering over the current manifold eta_k

    const Mat_<Vec4d> Psi_splat_joined = catCn(Psi_splat, Psi_splat_0);

    Mat_<Vec4d> Psi_splat_joined_resized;
    resize(Psi_splat_joined, Psi_splat_joined_resized, eta_k_small.size());

    const Mat_<Vec4d> blurred_projected_values = RF_filter(Psi_splat_joined_resized, eta_k_small, sigma_s_ / df, sigma_r_over_sqrt_2);

    Mat_<Point3d> w_ki_Psi_blur;
    Mat_<double> w_ki_Psi_blur_0;
    split_3_1(blurred_projected_values, w_ki_Psi_blur, w_ki_Psi_blur_0);

    // Slicing: gather blurred values from the manifold

    // Since we perform splatting and slicing at the same points over the manifolds,
    // the interpolation weights are equal to the gaussian weights used for splatting.

    const Mat_<double> w_ki = gaussian_distance_weights;

    Mat_<Point3d> w_ki_Psi_blur_resized;
    resize(w_ki_Psi_blur, w_ki_Psi_blur_resized, src_f_.size());
    add(sum_w_ki_Psi_blur_, times(w_ki_Psi_blur_resized, w_ki), sum_w_ki_Psi_blur_);

    Mat_<double> w_ki_Psi_blur_0_resized;
    resize(w_ki_Psi_blur_0, w_ki_Psi_blur_0_resized, src_f_.size());
    add(sum_w_ki_Psi_blur_0_, times(w_ki_Psi_blur_0_resized, w_ki), sum_w_ki_Psi_blur_0_);

    // Compute two new manifolds eta_minus and eta_plus

    if (current_tree_level < cur_tree_height_)
    {
        // Algorithm 1, Step 2: compute the eigenvector v1
        const Mat_<double> nX(src_joint_f_.size().area(), 3, (double*) X.data);

        Mat_<double> rand_vec(1, nX.cols);
        rng_.fill(rand_vec, RNG::UNIFORM, -0.5, 0.5);

        Mat_<double> v1 = computeEigenVector(nX, cluster_k, num_pca_iterations_, rand_vec);
        v1 = v1.t();

        // Algorithm 1, Step 3: Segment pixels into two clusters -- Eq. (6)

        Mat_<double> Nx_v1_mult;
        gemm(nX, v1, 1.0, noArray(), 0.0, Nx_v1_mult);

        const Mat_<double> dot(src_joint_f_.rows, src_joint_f_.cols, (double*) Nx_v1_mult.data);

        const Mat_<uchar> cluster_minus = (dot <  0) & cluster_k;
        const Mat_<uchar> cluster_plus  = (dot >= 0) & cluster_k;

        // Algorithm 1, Step 4: Compute new manifolds by weighted low-pass filtering -- Eq. (7-8)

        const Mat_<double> theta = Mat::ones(w_ki.size(), w_ki.type()) - w_ki;

        Mat_<Point3d> eta_minus = calcEta(src_joint_f_, theta, cluster_minus, sigma_s_, df);
        Mat_<Point3d> eta_plus = calcEta(src_joint_f_, theta, cluster_plus, sigma_s_, df);

        // Algorithm 1, Step 5: recursively build more manifolds.

        buildManifoldsAndPerformFiltering(eta_minus, cluster_minus, current_tree_level + 1);

        buildManifoldsAndPerformFiltering(eta_plus, cluster_plus, current_tree_level + 1);
    }
}
