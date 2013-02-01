#pragma once

#ifndef __ADAPTIVE_MANIFOLD_FILTER_HPP__
#define __ADAPTIVE_MANIFOLD_FILTER_HPP__

#include <opencv2/core/core.hpp>

namespace cv
{
    struct Buf
    {
        Mat_<Point3f> eta_1;
        Mat_<uchar> cluster_1;

        Mat_<Point3f> tilde_dst;
        Mat_<float> alpha;
        Mat_<Point3f> diff;
        Mat_<Point3f> dst;

        Mat_<float> V;

        Mat_<Point3f> dIcdx;
        Mat_<Point3f> dIcdy;
        Mat_<float> dIdx;
        Mat_<float> dIdy;
        Mat_<float> dHdx;
        Mat_<float> dVdy;

        Mat_<float> t;

        Mat_<float> theta_masked;
        Mat_<Point3f> mul;
        Mat_<Point3f> numerator;
        Mat_<float> denominator;
        Mat_<Point3f> numerator_filtered;
        Mat_<float> denominator_filtered;

        Mat_<Point3f> X;
        Mat_<Point3f> eta_k_small;
        Mat_<Point3f> eta_k_big;
        Mat_<Point3f> X_squared;
        Mat_<float> pixel_dist_to_manifold_squared;
        Mat_<float> gaussian_distance_weights;
        Mat_<Point3f> Psi_splat;
        Mat_<Vec4f> Psi_splat_joined;
        Mat_<Vec4f> Psi_splat_joined_resized;
        Mat_<Vec4f> blurred_projected_values;
        Mat_<Point3f> w_ki_Psi_blur;
        Mat_<float> w_ki_Psi_blur_0;
        Mat_<Point3f> w_ki_Psi_blur_resized;
        Mat_<float> w_ki_Psi_blur_0_resized;
        Mat_<float> rand_vec;
        Mat_<float> v1;
        Mat_<float> Nx_v1_mult;
        Mat_<float> theta;

        std::vector<Mat_<Point3f> > eta_minus;
        std::vector<Mat_<uchar> > cluster_minus;
        std::vector<Mat_<Point3f> > eta_plus;
        std::vector<Mat_<uchar> > cluster_plus;
    };

    class AdaptiveManifoldFilter : public Algorithm
    {
    public:
        AlgorithmInfo* info() const;

        AdaptiveManifoldFilter();

        /**
         * @brief Apply High-dimensional filtering using adaptive manifolds
         * @param src       Input image to be filtered.
         * @param dst       Adaptive-manifold filter response adjusted for outliers.
         * @param tilde_dst Adaptive-manifold filter response NOT adjusted for outliers.
         * @param src_joint Image for joint filtering (optional).
         */
        void apply(InputArray src, OutputArray dst, OutputArray tilde_dst = noArray(), InputArray src_joint = noArray());

    protected:
        double sigma_s_;
        double sigma_r_;
        int tree_height_;
        int num_pca_iterations_;

    private:
        void buildManifoldsAndPerformFiltering(const Mat_<Point3f>& eta_k, const Mat_<uchar>& cluster_k, int current_tree_level);

        Buf buf_;

        Mat_<Point3f> src_f_;
        Mat_<Point3f> src_joint_f_;

        Mat_<Point3f> sum_w_ki_Psi_blur_;
        Mat_<float> sum_w_ki_Psi_blur_0_;

        Mat_<float> min_pixel_dist_to_manifold_squared_;

        RNG rng_;

        int cur_tree_height_;
    };
}

#endif // __ADAPTIVE_MANIFOLD_FILTER_HPP__
