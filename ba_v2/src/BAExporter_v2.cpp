#include "BAExporter_v2.h"
#include "PBAImp_v2.h"

#include <algorithm>
#include <cctype>
#include <string>

namespace {

std::string normalize_method_name(const char* name) {
    std::string value = name == nullptr ? "" : name;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        if (ch == '_' || ch == ' ') {
            return '-';
        }
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

}  // namespace

const char* method_id_name(MethodId method) {
    switch (method) {
    case MethodId::A0_XYZ:
        return "A0-XYZ-W";
    case MethodId::A0_INV_DIST:
        return "A0-XYInvZ-W";
    case MethodId::A0_DEPTH:
        return "A0-SphRange-W";
    case MethodId::A0_INV_DEPTH:
        return "A0-SphInvRange-W";
    case MethodId::A1_XYZ:
        return "A1-XYZ-Aw";
    case MethodId::A1_INV_DIST:
        return "A1-XYInvZ-Aw";
    case MethodId::A1_DEPTH:
        return "A1-SphRange-Aw";
    case MethodId::A1_INV_DEPTH:
        return "A1-SphInvRange-Aw";
    case MethodId::A2_PA:
        return "A2-Parallax-Mw";
    case MethodId::A1_XYZ_AC:
        return "A1-XYZ-Ac";
    case MethodId::A1_XY_INV_Z_AC:
        return "A1-XYInvZ-Ac";
    case MethodId::A1_SPH_RANGE_AC:
        return "A1-SphRange-Ac";
    case MethodId::A1_SPH_INV_RANGE_AC:
        return "A1-SphInvRange-Ac";
    case MethodId::A2_PARALLAX_MC:
        return "A2-Parallax-Mc";
    }
    return "A0-XYZ-W";
}

bool method_id_from_name(const char* name, MethodId* method) {
    if (method == nullptr) {
        return false;
    }

    const std::string value = normalize_method_name(name);
    if (value == "A0-XYZ-W" || value == "A0-XYZ" || value == "XYZ") {
        *method = MethodId::A0_XYZ;
    } else if (value == "A0-XYINVZ-W" || value == "A0-INVDIST" ||
               value == "A0-XY-INVERSE-Z" || value == "XY-INVERSE-Z") {
        *method = MethodId::A0_INV_DIST;
    } else if (value == "A0-SPHRANGE-W" || value == "A0-DEPTH" || value == "DEPTH") {
        *method = MethodId::A0_DEPTH;
    } else if (value == "A0-SPHINVRANGE-W" || value == "A0-INVDEPTH" ||
               value == "A0-INVERSE-DEPTH" || value == "INVERSE-DEPTH") {
        *method = MethodId::A0_INV_DEPTH;
    } else if (value == "A1-XYZ-AW" || value == "A1-XYZ") {
        *method = MethodId::A1_XYZ;
    } else if (value == "A1-XYINVZ-AW" || value == "A1-INVDIST" || value == "A1-XY-INVERSE-Z") {
        *method = MethodId::A1_INV_DIST;
    } else if (value == "A1-SPHRANGE-AW" || value == "A1-DEPTH") {
        *method = MethodId::A1_DEPTH;
    } else if (value == "A1-SPHINVRANGE-AW" || value == "A1-INVDEPTH" || value == "A1-INVERSE-DEPTH") {
        *method = MethodId::A1_INV_DEPTH;
    } else if (value == "A2-PARALLAX-MW" || value == "A2-PA" || value == "A2-PARALLAX" || value == "PARALLAX") {
        *method = MethodId::A2_PA;
    } else if (value == "A1-XYZ-AC") {
        *method = MethodId::A1_XYZ_AC;
    } else if (value == "A1-XYINVZ-AC" || value == "A1-XY-INVERSE-Z-AC" ||
               value == "A1-IDP-CIVERA-AC" || value == "A1-IDP-AC") {
        *method = MethodId::A1_XY_INV_Z_AC;
    } else if (value == "A1-SPHRANGE-AC") {
        *method = MethodId::A1_SPH_RANGE_AC;
    } else if (value == "A1-SPHINVRANGE-AC") {
        *method = MethodId::A1_SPH_INV_RANGE_AC;
    } else if (value == "A2-PARALLAX-MC") {
        *method = MethodId::A2_PARALLAX_MC;
    } else {
        return false;
    }
    return true;
}

objectpointtype method_object_point_type(MethodId method) {
    switch (method) {
    case MethodId::A0_XYZ:
        return xyz;
    case MethodId::A0_INV_DIST:
        return xy_inverse_z;
    case MethodId::A0_DEPTH:
        return depth;
    case MethodId::A0_INV_DEPTH:
        return inverse_depth;
    case MethodId::A1_XYZ:
        return archored_xyz;
    case MethodId::A1_INV_DIST:
        return archored_xy_inverse_z;
    case MethodId::A1_DEPTH:
        return archored_depth;
    case MethodId::A1_INV_DEPTH:
        return archored_inverse_depth;
    case MethodId::A2_PA:
        return parallax;
    case MethodId::A1_XYZ_AC:
        return anchor_camera_xyz;
    case MethodId::A1_XY_INV_Z_AC:
        return anchor_camera_xy_inverse_z;
    case MethodId::A1_SPH_RANGE_AC:
        return anchor_camera_spherical_range;
    case MethodId::A1_SPH_INV_RANGE_AC:
        return anchor_camera_spherical_inverse_range;
    case MethodId::A2_PARALLAX_MC:
        return parallax_camera;
    }
    return xyz;
}

const char* benchmark_output_mode_name(BenchmarkOutputMode mode) {
    switch (mode) {
    case BenchmarkOutputMode::CleanTiming:
        return "clean";
    case BenchmarkOutputMode::Diagnostic:
        return "diagnostic";
    }
    return "diagnostic";
}

BAExporter::BAExporter() {
    ptr = new PBA;
}

BAExporter::~BAExporter() {
    delete ptr;
}

bool BAExporter::ba_run(const char* szCam,
                        const char* szFea,
                        const char* szXYZ,
                        const char* szCalib,
                        const char* szReport,
                        const char* szPose,
                        const char* sz3D,
                        MethodId method,
                        BenchmarkOutputMode output_mode,
                        int point_condition_sample,
                        int schur_sample) {
    return ptr->ba_run(const_cast<char*>(szCam),
                       const_cast<char*>(szFea),
                       const_cast<char*>(szXYZ),
                       const_cast<char*>(szCalib),
                       const_cast<char*>(szReport),
                       const_cast<char*>(szPose),
                       const_cast<char*>(sz3D),
                       method,
                       output_mode,
                       point_condition_sample,
                       schur_sample);
}

bool BAExporter::ba_initialize(const char* szCamera,
                               const char* szFeature,
                               const char* szCalib,
                               const char* szXYZ) {
    return ptr->ba_initialize(const_cast<char*>(szCamera),
                              const_cast<char*>(szFeature),
                              const_cast<char*>(szCalib),
                              const_cast<char*>(szXYZ));
}

const BARunMetrics& BAExporter::last_metrics() const {
    return ptr->last_metrics();
}
