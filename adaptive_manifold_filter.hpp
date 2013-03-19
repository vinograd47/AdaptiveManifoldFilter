#pragma once

#ifndef __ADAPTIVE_MANIFOLD_FILTER_HPP__
#define __ADAPTIVE_MANIFOLD_FILTER_HPP__

#include <opencv2/core/core.hpp>

namespace cv
{
    class AdaptiveManifoldFilter : public Algorithm
    {
    public:
        /**
         * @brief Apply High-dimensional filtering using adaptive manifolds
         * @param src       Input image to be filtered.
         * @param dst       Adaptive-manifold filter response adjusted for outliers.
         * @param tilde_dst Adaptive-manifold filter response NOT adjusted for outliers.
         * @param src_joint Image for joint filtering (optional).
         */
        virtual void apply(InputArray src, OutputArray dst, OutputArray tilde_dst = noArray(), InputArray src_joint = noArray()) = 0;

        virtual void collectGarbage() = 0;

        static Ptr<AdaptiveManifoldFilter> create();
    };
}

#endif // __ADAPTIVE_MANIFOLD_FILTER_HPP__
