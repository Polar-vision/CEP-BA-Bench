#pragma once

#include "ceres/ceres.h"

namespace cep {
namespace cost {

enum AnchorCameraRepresentation {
    kAnchorCameraXYZ = 0,
    kAnchorCameraXYInverseZ = 1,
    kAnchorCameraSphericalRange = 2,
    kAnchorCameraSphericalInverseRange = 3,
};

template <typename T>
inline void EulerToWorldToCamera(const T* euler_angles, T* rotation) {
    const T ey = euler_angles[0];
    const T ex = euler_angles[1];
    const T ez = euler_angles[2];
    const T c1 = cos(ey);
    const T c2 = cos(ex);
    const T c3 = cos(ez);
    const T s1 = sin(ey);
    const T s2 = sin(ex);
    const T s3 = sin(ez);
    rotation[0] = c1 * c3 - s1 * s2 * s3;
    rotation[1] = c2 * s3;
    rotation[2] = s1 * c3 + c1 * s2 * s3;
    rotation[3] = -c1 * s3 - s1 * s2 * c3;
    rotation[4] = c2 * c3;
    rotation[5] = -s1 * s3 + c1 * s2 * c3;
    rotation[6] = -s1 * c2;
    rotation[7] = -s2;
    rotation[8] = c1 * c2;
}

template <typename T>
inline void MatVec(const T* matrix, const T* vector, T* result) {
    result[0] = matrix[0] * vector[0] + matrix[1] * vector[1] +
                matrix[2] * vector[2];
    result[1] = matrix[3] * vector[0] + matrix[4] * vector[1] +
                matrix[5] * vector[2];
    result[2] = matrix[6] * vector[0] + matrix[7] * vector[1] +
                matrix[8] * vector[2];
}

template <typename T>
inline void MatTransposeVec(const T* matrix, const T* vector, T* result) {
    result[0] = matrix[0] * vector[0] + matrix[3] * vector[1] +
                matrix[6] * vector[2];
    result[1] = matrix[1] * vector[0] + matrix[4] * vector[1] +
                matrix[7] * vector[2];
    result[2] = matrix[2] * vector[0] + matrix[5] * vector[1] +
                matrix[8] * vector[2];
}

template <typename T>
inline void BearingFromAngles(const T& azimuth, const T& elevation, T* bearing) {
    bearing[0] = sin(azimuth) * cos(elevation);
    bearing[1] = sin(elevation);
    bearing[2] = cos(azimuth) * cos(elevation);
}

template <int Representation, typename T>
inline void DecodeAnchorCameraPoint(const T* parameters, T* point_camera) {
    if constexpr (Representation == kAnchorCameraXYZ) {
        point_camera[0] = parameters[0];
        point_camera[1] = parameters[1];
        point_camera[2] = parameters[2];
    } else if constexpr (Representation == kAnchorCameraXYInverseZ) {
        point_camera[0] = parameters[0] / parameters[2];
        point_camera[1] = parameters[1] / parameters[2];
        point_camera[2] = T(1) / parameters[2];
    } else {
        T bearing[3];
        BearingFromAngles(parameters[0], parameters[1], bearing);
        if constexpr (Representation == kAnchorCameraSphericalRange) {
            point_camera[0] = bearing[0] * parameters[2];
            point_camera[1] = bearing[1] * parameters[2];
            point_camera[2] = bearing[2] * parameters[2];
        } else {
            point_camera[0] = bearing[0] / parameters[2];
            point_camera[1] = bearing[1] / parameters[2];
            point_camera[2] = bearing[2] / parameters[2];
        }
    }
}

template <typename T>
inline void Project(const T* point_camera,
                    double fx,
                    double fy,
                    double cx,
                    double cy,
                    double observed_u,
                    double observed_v,
                    T* residuals) {
    const T predicted_u = T(fx) * point_camera[0] / point_camera[2] + T(cx);
    const T predicted_v = T(fy) * point_camera[1] / point_camera[2] + T(cy);
    residuals[0] = predicted_u - T(observed_u);
    residuals[1] = predicted_v - T(observed_v);
}

template <int Representation>
struct AnchorCameraMainResidual {
    AnchorCameraMainResidual(double observed_u,
                             double observed_v,
                             double fx,
                             double fy,
                             double cx,
                             double cy)
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const point_parameters, T* residuals) const {
        T point_camera[3];
        DecodeAnchorCameraPoint<Representation>(point_parameters, point_camera);
        Project(point_camera, fx, fy, cx, cy, u, v, residuals);
        return true;
    }

    static ceres::CostFunction* Create(double u,
                                       double v,
                                       double fx,
                                       double fy,
                                       double cx,
                                       double cy) {
        return new ceres::AutoDiffCostFunction<
            AnchorCameraMainResidual<Representation>,
            2,
            3>(new AnchorCameraMainResidual<Representation>(
            u, v, fx, fy, cx, cy));
    }

    double u;
    double v;
    double fx;
    double fy;
    double cx;
    double cy;
};

template <int Representation>
struct AnchorCameraOtherResidual {
    AnchorCameraOtherResidual(double observed_u,
                              double observed_v,
                              double fx,
                              double fy,
                              double cx,
                              double cy)
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const euler_current,
                    const T* const center_current,
                    const T* const euler_anchor,
                    const T* const center_anchor,
                    const T* const point_parameters,
                    T* residuals) const {
        T point_anchor[3];
        DecodeAnchorCameraPoint<Representation>(point_parameters, point_anchor);

        T rotation_anchor[9];
        T rotation_current[9];
        EulerToWorldToCamera(euler_anchor, rotation_anchor);
        EulerToWorldToCamera(euler_current, rotation_current);

        T point_anchor_world[3];
        MatTransposeVec(rotation_anchor, point_anchor, point_anchor_world);

        T ray_world[3];
        ray_world[0] = center_anchor[0] - center_current[0] +
                       point_anchor_world[0];
        ray_world[1] = center_anchor[1] - center_current[1] +
                       point_anchor_world[1];
        ray_world[2] = center_anchor[2] - center_current[2] +
                       point_anchor_world[2];

        T point_current[3];
        MatVec(rotation_current, ray_world, point_current);
        Project(point_current, fx, fy, cx, cy, u, v, residuals);
        return true;
    }

    static ceres::CostFunction* Create(double u,
                                       double v,
                                       double fx,
                                       double fy,
                                       double cx,
                                       double cy) {
        return new ceres::AutoDiffCostFunction<
            AnchorCameraOtherResidual<Representation>,
            2,
            3,
            3,
            3,
            3,
            3>(new AnchorCameraOtherResidual<Representation>(
            u, v, fx, fy, cx, cy));
    }

    double u;
    double v;
    double fx;
    double fy;
    double cx;
    double cy;
};

template <typename T>
inline void ParallaxCameraScaledRay(const T* euler_main,
                                    const T* center_main,
                                    const T* center_associate,
                                    const T* center_current,
                                    const T* parameters,
                                    T* scaled_ray_world) {
    T rotation_main[9];
    EulerToWorldToCamera(euler_main, rotation_main);

    T bearing_main[3];
    BearingFromAngles(parameters[0], parameters[1], bearing_main);

    T baseline_associate_world[3] = {
        center_associate[0] - center_main[0],
        center_associate[1] - center_main[1],
        center_associate[2] - center_main[2],
    };
    T baseline_current_world[3] = {
        center_current[0] - center_main[0],
        center_current[1] - center_main[1],
        center_current[2] - center_main[2],
    };

    T baseline_associate_main[3];
    T baseline_current_main[3];
    MatVec(rotation_main, baseline_associate_world, baseline_associate_main);
    MatVec(rotation_main, baseline_current_world, baseline_current_main);

    const T baseline_norm =
        sqrt(baseline_associate_main[0] * baseline_associate_main[0] +
             baseline_associate_main[1] * baseline_associate_main[1] +
             baseline_associate_main[2] * baseline_associate_main[2] +
             T(1e-24));
    T cos_beta =
        (bearing_main[0] * baseline_associate_main[0] +
         bearing_main[1] * baseline_associate_main[1] +
         bearing_main[2] * baseline_associate_main[2]) /
        baseline_norm;
    if (cos_beta > T(1)) {
        cos_beta = T(1);
    } else if (cos_beta < T(-1)) {
        cos_beta = T(-1);
    }
    T sin_beta_squared = T(1) - cos_beta * cos_beta;
    if (sin_beta_squared < T(0)) {
        sin_beta_squared = T(0);
    }
    // At exact ray-baseline collinearity the residual remains finite, but
    // sqrt(0) has an unbounded derivative and produces NaN Jets. The floor is
    // at machine scale in squared sine and only regularizes that singularity.
    const T sin_beta = sqrt(sin_beta_squared + T(1e-24));
    const T sin_omega = sin(parameters[2]);
    const T sin_beta_plus_omega =
        sin_beta * cos(parameters[2]) + cos_beta * sin_omega;

    T scaled_ray_main[3];
    scaled_ray_main[0] =
        baseline_norm * sin_beta_plus_omega * bearing_main[0] -
        sin_omega * baseline_current_main[0];
    scaled_ray_main[1] =
        baseline_norm * sin_beta_plus_omega * bearing_main[1] -
        sin_omega * baseline_current_main[1];
    scaled_ray_main[2] =
        baseline_norm * sin_beta_plus_omega * bearing_main[2] -
        sin_omega * baseline_current_main[2];
    MatTransposeVec(rotation_main, scaled_ray_main, scaled_ray_world);
}

struct ParallaxCameraMainResidual {
    ParallaxCameraMainResidual(double observed_u,
                               double observed_v,
                               double fx,
                               double fy,
                               double cx,
                               double cy)
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const parameters, T* residuals) const {
        T bearing[3];
        BearingFromAngles(parameters[0], parameters[1], bearing);
        Project(bearing, fx, fy, cx, cy, u, v, residuals);
        return true;
    }

    static ceres::CostFunction* Create(double u,
                                       double v,
                                       double fx,
                                       double fy,
                                       double cx,
                                       double cy) {
        return new ceres::AutoDiffCostFunction<ParallaxCameraMainResidual, 2, 3>(
            new ParallaxCameraMainResidual(u, v, fx, fy, cx, cy));
    }

    double u;
    double v;
    double fx;
    double fy;
    double cx;
    double cy;
};

struct ParallaxCameraAssociateResidual {
    ParallaxCameraAssociateResidual(double observed_u,
                                    double observed_v,
                                    double fx,
                                    double fy,
                                    double cx,
                                    double cy)
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const euler_associate,
                    const T* const center_associate,
                    const T* const euler_main,
                    const T* const center_main,
                    const T* const parameters,
                    T* residuals) const {
        T scaled_ray_world[3];
        ParallaxCameraScaledRay(euler_main,
                                center_main,
                                center_associate,
                                center_associate,
                                parameters,
                                scaled_ray_world);
        T rotation_associate[9];
        T scaled_ray_associate[3];
        EulerToWorldToCamera(euler_associate, rotation_associate);
        MatVec(rotation_associate, scaled_ray_world, scaled_ray_associate);
        Project(scaled_ray_associate, fx, fy, cx, cy, u, v, residuals);
        return true;
    }

    static ceres::CostFunction* Create(double u,
                                       double v,
                                       double fx,
                                       double fy,
                                       double cx,
                                       double cy) {
        return new ceres::AutoDiffCostFunction<
            ParallaxCameraAssociateResidual,
            2,
            3,
            3,
            3,
            3,
            3>(new ParallaxCameraAssociateResidual(u, v, fx, fy, cx, cy));
    }

    double u;
    double v;
    double fx;
    double fy;
    double cx;
    double cy;
};

struct ParallaxCameraOtherResidual {
    ParallaxCameraOtherResidual(double observed_u,
                                double observed_v,
                                double fx,
                                double fy,
                                double cx,
                                double cy)
        : u(observed_u), v(observed_v), fx(fx), fy(fy), cx(cx), cy(cy) {}

    template <typename T>
    bool operator()(const T* const euler_current,
                    const T* const center_current,
                    const T* const euler_main,
                    const T* const center_main,
                    const T* const center_associate,
                    const T* const parameters,
                    T* residuals) const {
        T scaled_ray_world[3];
        ParallaxCameraScaledRay(euler_main,
                                center_main,
                                center_associate,
                                center_current,
                                parameters,
                                scaled_ray_world);
        T rotation_current[9];
        T scaled_ray_current[3];
        EulerToWorldToCamera(euler_current, rotation_current);
        MatVec(rotation_current, scaled_ray_world, scaled_ray_current);
        Project(scaled_ray_current, fx, fy, cx, cy, u, v, residuals);
        return true;
    }

    static ceres::CostFunction* Create(double u,
                                       double v,
                                       double fx,
                                       double fy,
                                       double cx,
                                       double cy) {
        return new ceres::AutoDiffCostFunction<
            ParallaxCameraOtherResidual,
            2,
            3,
            3,
            3,
            3,
            3,
            3>(new ParallaxCameraOtherResidual(u, v, fx, fy, cx, cy));
    }

    double u;
    double v;
    double fx;
    double fy;
    double cx;
    double cy;
};

}  // namespace cost
}  // namespace cep
