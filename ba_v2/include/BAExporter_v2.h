#pragma once

#ifdef BA_EXPORTS
    #define BAapi __declspec(dllexport)
#elif defined(BA_STATIC)
    #define BAapi
#else
    #define BAapi __declspec(dllimport)
#endif

enum ObjectPointType {
    xyz = 0,
    xy_inverse_z = 1,
    depth = 2,
    inverse_depth = 3,
    archored_xyz = 4,
    archored_xy_inverse_z = 5,
    archored_depth = 6,
    archored_inverse_depth = 7,
    anchored_xyz = archored_xyz,
    anchored_xy_inverse_z = archored_xy_inverse_z,
    anchored_depth = archored_depth,
    anchored_inverse_depth = archored_inverse_depth,
    parallax = 8,
    anchor_camera_xyz = 9,
    anchor_camera_xy_inverse_z = 10,
    anchor_camera_spherical_range = 11,
    anchor_camera_spherical_inverse_range = 12,
    parallax_camera = 13,
};
using objectpointtype = ObjectPointType;

enum class MethodId {
    A0_XYZ = 0,
    A0_INV_DIST,
    A0_DEPTH,
    A0_INV_DEPTH,
    A1_XYZ,
    A1_INV_DIST,
    A1_DEPTH,
    A1_INV_DEPTH,
    A2_PA,
    A1_XYZ_AC,
    A1_XY_INV_Z_AC,
    A1_SPH_RANGE_AC,
    A1_SPH_INV_RANGE_AC,
    A2_PARALLAX_MC,
};

enum class BenchmarkOutputMode {
    CleanTiming = 0,
    Diagnostic = 1,
};

struct BARunMetrics {
    bool success = false;
    int cameras = 0;
    int points = 0;
    int observations = 0;
    int iterations = 0;
    int accepted_steps = 0;
    int rejected_steps = 0;
    int linear_solver_iterations = 0;
    int termination_type = 0;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    double initial_rmse_px = 0.0;
    double final_rmse_px = 0.0;
    double initial_gradient_max_norm = 0.0;
    double final_gradient_max_norm = 0.0;
    double final_gradient_norm = 0.0;
    double gradient_reduction_ratio_final = 0.0;
    int reached_gradient_tolerance = 0;
    int iterations_to_gradient_tolerance = -1;
    double final_relative_function_decrease = 0.0;
    double final_relative_step_size = 0.0;
    double final_lm_gain_ratio = 0.0;
    double final_gradient_lipschitz_estimate = 0.0;
    double final_direction_quality = 0.0;
    double solver_time_sec = 0.0;
    double linear_solver_time_sec = 0.0;
};

BAapi const char* method_id_name(MethodId method);
BAapi bool method_id_from_name(const char* name, MethodId* method);
BAapi objectpointtype method_object_point_type(MethodId method);
BAapi const char* benchmark_output_mode_name(BenchmarkOutputMode mode);

class IBA;

class BAapi BAExporter {
public:
    BAExporter();
    ~BAExporter();

    bool ba_run(const char* szCam = nullptr,
                const char* szFea = nullptr,
                const char* szXYZ = nullptr,
                const char* szCalib = nullptr,
                const char* szReport = nullptr,
                const char* szPose = nullptr,
                const char* sz3D = nullptr,
                MethodId method = MethodId::A0_XYZ,
                BenchmarkOutputMode output_mode = BenchmarkOutputMode::Diagnostic,
                int point_condition_sample = 0,
                int schur_sample = 0);

    bool ba_initialize(const char* szCamera,
                       const char* szFeature,
                       const char* szCalib = nullptr,
                       const char* szXYZ = nullptr);

    const BARunMetrics& last_metrics() const;

private:
    IBA* ptr;
};
