#pragma once

#ifndef __ADAPTIVE_MANIFOLD_FILTER_HPP__
#define __ADAPTIVE_MANIFOLD_FILTER_HPP__

#include <opencv2/core/core.hpp>

namespace cv
{
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
