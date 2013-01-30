#include <cmath>
#include <limits>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

void adaptiveManifoldFilter(InputArray f, OutputArray g, OutputArray tilde_g,
                            double sigma_s, double sigma_r, int tree_height = -1, int num_pca_iterations = 1,
                            InputArray f_joint = noArray());

int main(int argc, const char* argv[])
{
    Mat img = imread(argv[1]);

    Mat g, tilde_g;
    adaptiveManifoldFilter(img, g, tilde_g, 16.0, 0.2);

    imshow("img", img);
    imshow("g", g);
    imshow("tilde_g", tilde_g);
    waitKey();

    return 0;
}

namespace
{
    RNG rng;

    Mat sum_w_ki_Psi_blur;
    Mat sum_w_ki_Psi_blur_0;

    Mat min_pixel_dist_to_manifold_squared;

    int computeManifoldTreeHeight(double sigma_s, double sigma_r)
    {
        const double Hs = floor(log2(sigma_s)) - 1.0;
        const double Lr = 1.0 - sigma_r;
        return max(2, static_cast<int>(ceil(Hs * Lr)));
    }

    Mat h_filter_horizontal(Mat f, double sigma)
    {
        CV_Assert( f.depth() == CV_64F );

        const int cn = f.channels();

        const double a = exp(-sqrt(2.0) / sigma);

        Mat g = f.clone();

        for (int y = 0; y < f.rows; ++y)
        {
            const double* f_row = f.ptr<double>(y);
            double* g_row = g.ptr<double>(y);

            for (int x = 1; x < f.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    g_row[x * cn + c] = f_row[x * cn + c] + a * (f_row[(x - 1) * cn + c] - f_row[x * cn + c]);
                }
            }

            for (int x = f.cols - 2; x >= 0; --x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    g_row[x * cn + c] = g_row[x * cn + c] + a * (g_row[(x + 1) * cn + c] - g_row[x * cn + c]);
                }
            }
        }

        return g;
    }

    Mat h_filter(Mat f, double sigma)
    {
        Mat g = h_filter_horizontal(f, sigma);
        g = g.t();
        g = h_filter_horizontal(g, sigma);
        return g.t();
    }

    double floor_to_power_of_two(double r)
    {
        return pow(2.0, floor(log2(r)));
    }

    Mat phi(Mat x_squared, double sigma)
    {
        CV_Assert( x_squared.type() == CV_64FC1 );

        Mat res(x_squared.size(), x_squared.type());

        for (int y = 0; y < res.rows; ++y)
        {
            for (int x = 0; x < res.cols; ++x)
            {
                double x_val = x_squared.at<double>(y, x);
                double res_val = exp(-0.5 * x_val / sigma / sigma);
                res.at<double>(y, x) = res_val;
            }
        }

        return res;
    }

    Mat channelsSum(Mat a)
    {
        CV_Assert( a.depth() == CV_64F );

        const int cn = a.channels();

        Mat g(a.size(), CV_64FC1, Scalar::all(0));

        for (int y = 0; y < a.rows; ++y)
        {
            const double* a_row = a.ptr<double>(y);
            double* g_row = g.ptr<double>(y);

            for (int x = 0; x < a.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    double a_val = a_row[x * cn + c];
                    g_row[x] += a_val;
                }
            }
        }

        return g;
    }

    Mat times(Mat a, Mat b)
    {
        CV_Assert( a.depth() == CV_64F );
        CV_Assert( b.type() == CV_64FC1 );

        const int cn = a.channels();

        Mat g(a.size(), a.type());

        for (int y = 0; y < a.rows; ++y)
        {
            const double* a_row = a.ptr<double>(y);
            const double* b_row = b.ptr<double>(y);
            double* g_row = g.ptr<double>(y);

            for (int x = 0; x < a.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    g_row[x * cn + c] = a_row[x * cn + c] * b_row[x];
                }
            }
        }

        return g;
    }

    Mat rdivide(Mat a, Mat b)
    {
        CV_Assert( a.depth() == CV_64F );
        CV_Assert( b.type() == CV_64FC1 );

        const int cn = a.channels();

        Mat g(a.size(), a.type());

        for (int y = 0; y < a.rows; ++y)
        {
            const double* a_row = a.ptr<double>(y);
            const double* b_row = b.ptr<double>(y);
            double* g_row = g.ptr<double>(y);

            for (int x = 0; x < a.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    g_row[x * cn + c] = a_row[x * cn + c] / b_row[x];
                }
            }
        }

        return g;
    }

    Mat catCn(Mat a, Mat b)
    {
        CV_Assert( a.depth() == CV_64F );
        CV_Assert( b.depth() == CV_64F );

        const int a_cn = a.channels();
        const int b_cn = b.channels();
        const int g_cn = a_cn + b_cn;

        Mat g(a.size(), CV_64FC(g_cn));

        for (int y = 0; y < a.rows; ++y)
        {
            const double* a_row = a.ptr<double>(y);
            const double* b_row = b.ptr<double>(y);
            double* g_row = g.ptr<double>(y);

            for (int x = 0; x < a.cols; ++x)
            {
                int gc = 0;

                for (int c = 0; c < a_cn; ++c, ++gc)
                {
                    g_row[x * g_cn + gc] = a_row[x * a_cn + c];
                }

                for (int c = 0; c < b_cn; ++c, ++gc)
                {
                    g_row[x * g_cn + gc] = b_row[x * b_cn + c];
                }
            }
        }

        return g;
    }

    Mat diffY(Mat f)
    {
        CV_Assert( f.depth() == CV_64F );

        const int cn = f.channels();

        Mat g(f.rows - 1, f.cols, f.type());

        for (int y = 0; y < f.rows - 1; ++y)
        {
            const double* f_row = f.ptr<double>(y);
            const double* f_n_row = f.ptr<double>(y + 1);
            double* g_row = g.ptr<double>(y);

            for (int x = 0; x < f.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    g_row[x * cn + c] = f_n_row[x * cn + c] - f_row[x * cn + c];
                }
            }
        }

        return g;
    }

    Mat diffX(Mat f)
    {
        CV_Assert( f.depth() == CV_64F );

        const int cn = f.channels();

        Mat g(f.rows, f.cols - 1, f.type());

        for (int y = 0; y < f.rows; ++y)
        {
            const double* f_row = f.ptr<double>(y);
            double* g_row = g.ptr<double>(y);

            for (int x = 0; x < f.cols - 1; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    g_row[x * cn + c] = f_row[(x + 1) * cn + c] - f_row[x * cn + c];
                }
            }
        }

        return g;
    }

    Mat TransformedDomainRecursiveFilter_Horizontal(Mat I, Mat D, double sigma)
    {
        CV_Assert( I.depth() == CV_64F );
        CV_Assert( D.type() == CV_64FC1 );

        const int cn = I.channels();

        const double a = exp(-sqrt(2.0) / sigma);

        Mat F = I.clone();

        Mat V(D.size(), D.type());
        for (int y = 0; y < D.rows; ++y)
        {
            for (int x = 0; x < D.cols; ++x)
            {
                V.at<double>(y, x) = pow(a, D.at<double>(y, x));
            }
        }

        for (int y = 0; y < I.rows; ++y)
        {
            for (int x = 1; x < I.cols; ++x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    F.at<double>(y, x * cn + c) += V.at<double>(y, x) * (F.at<double>(y, (x - 1) * cn + c) - F.at<double>(y, x * cn + c));
                }
            }
        }

        for (int y = 0; y < I.rows; ++y)
        {
            for (int x = I.cols - 2; x >= 0; --x)
            {
                for (int c = 0; c < cn; ++c)
                {
                    F.at<double>(y, x * cn + c) += V.at<double>(y, x + 1) * (F.at<double>(y, (x + 1) * cn + c) - F.at<double>(y, x * cn + c));
                }
            }
        }

        return F;
    }

    Mat RF_filter(Mat img, Mat joint_image, double sigma_s, double sigma_r)
    {
        CV_Assert( joint_image.empty() || joint_image.size() == img.size() );

        Mat I;
        img.convertTo(I, CV_64F);

        Mat J;
        if (joint_image.empty())
            J = I.clone();
        else
            joint_image.convertTo(J, CV_64F);

        Mat dIcdx = diffX(J);
        Mat dIcdy = diffY(J);

        Mat dIdx = Mat::zeros(img.size(), CV_64F);
        Mat dIdy = Mat::zeros(img.size(), CV_64F);

        const int num_joint_channels = joint_image.channels();
        for (int c = 0; c < num_joint_channels; ++c)
        {
            for (int y = 0; y < img.rows; ++y)
            {
                for (int x = 1; x < img.cols; ++x)
                {
                    const double val = dIcdx.at<double>(y, (x - 1) * num_joint_channels + c);
                    dIdx.at<double>(y, x) += val * val;
                }
            }

            for (int y = 1; y < img.rows; ++y)
            {
                for (int x = 0; x < img.cols; ++x)
                {
                    const double val = dIcdy.at<double>(y - 1, x * num_joint_channels + c);
                    dIdy.at<double>(y, x) += val * val;
                }
            }
        }

        const double sigma_H = sigma_s;

        Mat dHdx;
        {
            Mat temp;
            dIdx.convertTo(temp, dIdx.type(), (sigma_H / sigma_r) * (sigma_H / sigma_r), (sigma_H / sigma_s) * (sigma_H / sigma_s));
            sqrt(temp, dHdx);
        }

        Mat dVdy;
        {
            Mat temp;
            dIdy.convertTo(temp, dIdx.type(), (sigma_H / sigma_r) * (sigma_H / sigma_r), (sigma_H / sigma_s) * (sigma_H / sigma_s));
            sqrt(temp, dVdy);
        }

        dVdy = dVdy.t();

        const int N = 1;
        Mat F = I.clone();

        for (int i = 0; i < N; ++i)
        {
            const double sigma_H_i = sigma_H * sqrt(3.0) * pow(2.0, (N - (i + 1))) / sqrt(pow(4.0, N) - 1.0);

            F = TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i);
            F = F.t();

            F = TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i);
            F = F.t();
        }

        Mat res;
        F.convertTo(res, img.type());

        return res;
    }

    void split_3_1(Mat src, Mat& dst1, Mat& dst2)
    {
        CV_Assert( src.type() == CV_64FC4 );

        dst1.create(src.size(), CV_64FC3);
        dst2.create(src.size(), CV_64FC1);

        for (int y = 0; y < src.rows; ++y)
        {
            for (int x = 0; x < src.cols; ++x)
            {
                Vec4d val = src.at<Vec4d>(y, x);
                dst1.at<Vec3d>(y, x) = Vec3d(val[0], val[1], val[2]);
                dst2.at<double>(y, x) = val[3];
            }
        }
    }

    Mat computeEigenVector(Mat X, Mat mask, int num_pca_iterations, Mat rand_vec)
    {
        Mat p = rand_vec.clone();

        for (int i = 0; i < num_pca_iterations; ++i)
        {
            Mat t(X.rows, X.cols, CV_64FC1, Scalar::all(0));

            for (int y = 0; y < mask.rows; ++y)
            {
                for (int x = 0; x < mask.cols; ++x)
                {
                    if (mask.at<uchar>(y, x))
                    {
                        int ind = y * mask.cols + x;

                        double dots = 0.0;
                        for (int c = 0; c < X.cols; ++c)
                            dots += p.at<double>(0, c) * X.at<double>(ind, c);

                        for (int c = 0; c < X.cols; ++c)
                            t.at<double>(ind, c) = dots * X.at<double>(ind, c);
                    }
                }
            }

            p.setTo(Scalar::all(0));
            for (int i = 0; i < X.rows; ++i)
                for (int c = 0; c < X.cols; ++c)
                    p.at<double>(0, c) += t.at<double>(i, c);
        }

        return p / norm(p);
    }

    void buildManifoldsAndPerformFiltering(Mat f, Mat f_joint, Mat eta_k, Mat cluster_k, double sigma_s, double sigma_r, int current_tree_level, int tree_height, int num_pca_iterations)
    {
        // Dividing the covariance matrix by 2 is equivalent to dividing the standard deviations by sqrt(2).

        const double sigma_r_over_sqrt_2 = sigma_r / sqrt(2.0);

        // Compute downsampling factor

        double df = min(sigma_s / 4.0, 256.0 * sigma_r);
        df = floor_to_power_of_two(df);
        df = max(1.0, df);

        // Splatting: project the pixel values onto the current manifold eta_k

        Mat X;
        if (eta_k.rows == f_joint.rows)
        {
            X = f_joint - eta_k;
            Mat temp;
            resize(eta_k, temp, Size(), 1.0 / df, 1.0 / df);
            temp.copyTo(eta_k);
        }
        else
        {
            Mat temp;
            resize(eta_k, temp, f.size());
            X = f_joint - temp;
        }

        // Project pixel colors onto the manifold -- Eq. (3), Eq. (5)

        Mat pixel_dist_to_manifold_squared;
        {
            Mat temp;
            multiply(X, X, temp);
            pixel_dist_to_manifold_squared = channelsSum(temp);
        }

        Mat gaussian_distance_weights = phi(pixel_dist_to_manifold_squared, sigma_r_over_sqrt_2);
        Mat Psi_splat = times(f, gaussian_distance_weights);
        Mat Psi_splat_0 = gaussian_distance_weights.clone();

        // Save min distance to later perform adjustment of outliers -- Eq. (10)

        min(min_pixel_dist_to_manifold_squared, pixel_dist_to_manifold_squared, min_pixel_dist_to_manifold_squared);

        // Blurring: perform filtering over the current manifold eta_k

        Mat blurred_projected_values ;
        {
            Mat temp;
            resize(catCn(Psi_splat, Psi_splat_0), temp, Size(), 1.0 / df, 1.0 / df);

            blurred_projected_values = RF_filter(temp, eta_k, sigma_s / df, sigma_r_over_sqrt_2);
        }

        Mat w_ki_Psi_blur, w_ki_Psi_blur_0;
        split_3_1(blurred_projected_values, w_ki_Psi_blur, w_ki_Psi_blur_0);

        // Slicing: gather blurred values from the manifold

        // Since we perform splatting and slicing at the same points over the manifolds,
        // the interpolation weights are equal to the gaussian weights used for splatting.

        Mat w_ki = gaussian_distance_weights.clone();

        {
            Mat temp;
            resize(w_ki_Psi_blur, temp, f.size());
            add(sum_w_ki_Psi_blur, times(temp, w_ki), sum_w_ki_Psi_blur);

            resize(w_ki_Psi_blur_0, temp, f.size());
            add(sum_w_ki_Psi_blur_0, times(temp, w_ki), sum_w_ki_Psi_blur_0);
        }

        // Compute two new manifolds eta_minus and eta_plus

        // Test stopping criterion

        if (current_tree_level < tree_height)
        {
            // Algorithm 1, Step 2: compute the eigenvector v1
            Mat nX = Mat(f_joint.size().area(), f_joint.channels(), CV_64FC1, X.data, f_joint.channels() * sizeof(double));

            Mat rand_vec(1, nX.cols, CV_64FC1);
            rng.fill(rand_vec, RNG::UNIFORM, -0.5, 0.5);

            Mat v1 = computeEigenVector(nX, cluster_k, num_pca_iterations, rand_vec);

            // Algorithm 1, Step 3: Segment pixels into two clusters -- Eq. (6)

            Mat cluster_minus, cluster_plus;
            {
                Mat temp;
                gemm(nX, v1.t(), 1.0, noArray(), 0.0, temp);

                Mat dot(f_joint.rows, f_joint.cols, CV_64FC1, temp.data, f_joint.cols * sizeof(double));

                cluster_minus = (dot <  0) & cluster_k;
                cluster_plus  = (dot >= 0) & cluster_k;
            }

            // Algorithm 1, Step 4: Compute new manifolds by weighted low-pass filtering -- Eq. (7-8)

            Mat theta = Mat::ones(w_ki.size(), w_ki.type()) - w_ki;

            Mat eta_minus;
            {
                Mat temp1(theta.size(), CV_64FC1);
                for (int y = 0; y < theta.rows; ++y)
                {
                    for (int x = 0; x < theta.cols; ++x)
                    {
                        temp1.at<double>(y, x) = cluster_minus.at<uchar>(y, x) * theta.at<double>(y, x);
                    }
                }

                Mat temp2;
                resize(times(f_joint, temp1), temp2, Size(), 1.0 / df, 1.0 / df);

                Mat temp3;
                resize(temp1, temp3, Size(), 1.0 / df, 1.0 / df);

                eta_minus = rdivide(h_filter(temp2, sigma_s / df), h_filter(temp3, sigma_s / df));
            }

            Mat eta_plus;
            {
                Mat temp1(theta.size(), CV_64FC1);
                for (int y = 0; y < theta.rows; ++y)
                {
                    for (int x = 0; x < theta.cols; ++x)
                    {
                        temp1.at<double>(y, x) = cluster_plus.at<uchar>(y, x) * theta.at<double>(y, x);
                    }
                }

                Mat temp2;
                resize(times(f_joint, temp1), temp2, Size(), 1.0 / df, 1.0 / df);

                Mat temp3;
                resize(temp1, temp3, Size(), 1.0 / df, 1.0 / df);

                eta_plus = rdivide(h_filter(temp2, sigma_s / df), h_filter(temp3, sigma_s / df));
            }

            // Algorithm 1, Step 5: recursively build more manifolds.

            buildManifoldsAndPerformFiltering(f, f_joint, eta_minus, cluster_minus, sigma_s, sigma_r, current_tree_level + 1, tree_height, num_pca_iterations);

            buildManifoldsAndPerformFiltering(f, f_joint, eta_plus,  cluster_plus,  sigma_s, sigma_r, current_tree_level + 1, tree_height, num_pca_iterations);
        }
    }
}

void adaptiveManifoldFilter(InputArray _f, OutputArray _g, OutputArray _tilde_g, double sigma_s, double sigma_r, int tree_height, int num_pca_iterations, InputArray _f_joint)
{
    Mat f;
    _f.getMat().convertTo(f, CV_64F, 1.0 / 255.0);

    // Use the center pixel as seed to random number generation.
    rng = RNG(static_cast<uint64>(f.at<double>(f.rows / 2, f.cols / 2 * f.channels())));

    sum_w_ki_Psi_blur = Mat::zeros(f.size(), f.type());
    sum_w_ki_Psi_blur_0 = Mat::zeros(f.size(), CV_64FC1);

    min_pixel_dist_to_manifold_squared = Mat(f.size(), CV_64FC1, Scalar::all(numeric_limits<double>::max()));

    // If the tree_height was not specified, compute it using Eq. (10) of our paper.
    if (tree_height < 0)
        tree_height = computeManifoldTreeHeight(sigma_s, sigma_r);

    // If no joint signal was specified, use the original signal
    Mat f_joint;
    if (_f_joint.empty())
        f_joint = f.clone();
    else
        _f_joint.getMat().convertTo(f_joint, CV_64F, 1.0 / 255.0);

    // By default we use only one iteration to compute the eigenvector v1 (Appendix B)
    if (num_pca_iterations < 1)
        num_pca_iterations = 1;

    // Algorithm 1, Step 1: compute the first manifold by low-pass filtering.
    Mat eta_1 = h_filter(f_joint, sigma_s);
    Mat cluster_1(f.size(), CV_8U, Scalar::all(1));

    int current_tree_level = 1;

    buildManifoldsAndPerformFiltering(f, f_joint, eta_1, cluster_1, sigma_s, sigma_r, current_tree_level, tree_height, num_pca_iterations);

    // Compute the filter response by normalized convolution -- Eq. (4)
    Mat tilde_g = rdivide(sum_w_ki_Psi_blur, sum_w_ki_Psi_blur_0);

    // Adjust the filter response for outlier pixels -- Eq. (10)
    Mat alpha;
    exp(min_pixel_dist_to_manifold_squared * (-0.5 / sigma_r / sigma_r), alpha);
    Mat g = f + times(tilde_g - f, alpha);

    g.convertTo(_g, CV_8U, 255.0);
    if (_tilde_g.needed())
        tilde_g.convertTo(_tilde_g, CV_8U, 255.0);
}
