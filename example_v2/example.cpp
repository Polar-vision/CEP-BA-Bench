#include "BAExporter_v2.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct DatasetPaths {
    std::string base_name;
    std::string name;
    fs::path base_root;
    fs::path quality_root;
    fs::path reference_original;
    fs::path cam;
    fs::path feature;
    fs::path xyz;
    fs::path calib;
};

struct InputFiles {
    fs::path cam;
    fs::path feature;
    fs::path xyz;
    fs::path calib;
};

const char* summary_header() {
    return "base_dataset,quality_dataset,method,mode,status,wall_time_sec,"
           "solver_time_sec,linear_solver_time_sec,cameras,points,observations,"
           "initial_cost,final_cost,initial_rmse_px,final_rmse_px,iterations,"
           "accepted_steps,rejected_steps,linear_solver_iterations,"
           "initial_gradient_max_norm,final_gradient_max_norm,final_gradient_norm,"
           "gradient_reduction_ratio_final,reached_gradient_tolerance,"
           "iterations_to_gradient_tolerance,final_relative_function_decrease,"
           "final_relative_step_size,final_lm_gain_ratio,"
           "final_gradient_lipschitz_estimate,final_direction_quality,"
           "termination_type,report";
}

std::string run_key(const std::string& base_dataset,
                    const std::string& quality_dataset,
                    const std::string& method,
                    const std::string& mode) {
    constexpr char separator = '\x1f';
    return base_dataset + separator + quality_dataset + separator + method + separator + mode;
}

bool parse_summary_prefix(const std::string& line, std::vector<std::string>* fields) {
    if (fields == nullptr) {
        return false;
    }
    fields->clear();
    std::size_t begin = 0;
    for (int field = 0; field < 5; ++field) {
        const std::size_t end = line.find(',', begin);
        if (end == std::string::npos) {
            fields->clear();
            return false;
        }
        fields->push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return true;
}

const std::vector<MethodId>& all_methods() {
    static const std::vector<MethodId> methods = {
        MethodId::A0_XYZ,
        MethodId::A0_INV_DIST,
        MethodId::A0_DEPTH,
        MethodId::A0_INV_DEPTH,
        MethodId::A1_XYZ_AC,
        MethodId::A1_XY_INV_Z_AC,
        MethodId::A1_SPH_RANGE_AC,
        MethodId::A1_SPH_INV_RANGE_AC,
        MethodId::A2_PARALLAX_MC,
        MethodId::A1_XYZ,
        MethodId::A1_INV_DIST,
        MethodId::A1_DEPTH,
        MethodId::A1_INV_DEPTH,
        MethodId::A2_PA,
    };
    return methods;
}

bool output_mode_from_name(const std::string& name, BenchmarkOutputMode* mode) {
    if (mode == nullptr) {
        return false;
    }
    std::string value = name;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "clean" || value == "timing" || value == "clean-timing") {
        *mode = BenchmarkOutputMode::CleanTiming;
        return true;
    }
    if (value == "diagnostic" || value == "diagnostics" || value == "debug") {
        *mode = BenchmarkOutputMode::Diagnostic;
        return true;
    }
    return false;
}

void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe << " [--problems <input-root>] [--out <results>]\n"
        << "                  [--method <MethodId>] [--limit <N>] [--dataset <name>]\n"
        << "                  [--mode clean|diagnostic] [--resume]\n"
        << "                  [--no-xyz]\n"
        << "                  [--point-condition-sample <N>]\n"
        << "                  [--schur-sample <N>]\n"
        << "\n"
        << "Input root may be a BA Datasets tree with Initial Value/Ground Truth\n"
        << "folders, or a prepared original/quality benchmark tree.\n"
        << "Method IDs: A0-XYZ-W, A0-XYInvZ-W, A0-SphRange-W, A0-SphInvRange-W,\n"
        << "            A1-XYZ-Ac, A1-XYInvZ-Ac, A1-SphRange-Ac,\n"
        << "            A1-SphInvRange-Ac, A2-Parallax-Mc,\n"
        << "            A1-XYZ-Aw, A1-XYInvZ-Aw, A1-SphRange-Aw,\n"
        << "            A1-SphInvRange-Aw, A2-Parallax-Mw\n"
        << "Modes: clean writes only summary.csv; diagnostic also writes report,\n"
        << "       metrics.json, convergence.txt, FinalPose.txt, and Final3D.ply.\n"
        << "--resume retains successful rows, reruns failures, and skips completed runs.\n";
}

bool is_cam_file(const fs::path& path) {
    const std::string name = path.filename().string();
    return fs::is_regular_file(path) &&
           name.rfind("Cam", 0) == 0 &&
           path.extension() == ".txt";
}

std::optional<fs::path> find_camera_file(const fs::path& original) {
    const fs::path canonical = original / "Cam.txt";
    if (fs::is_regular_file(canonical)) {
        return canonical;
    }

    for (const auto& entry : fs::directory_iterator(original)) {
        if (is_cam_file(entry.path())) {
            return entry.path();
        }
    }
    return std::nullopt;
}

std::optional<InputFiles> find_input_files(const fs::path& dataset_root) {
    const fs::path feature = dataset_root / "Feature.txt";
    const fs::path xyz = dataset_root / "XYZ.txt";
    const fs::path calib = dataset_root / "cal.txt";
    const auto cam = find_camera_file(dataset_root);
    if (!cam || !fs::exists(feature) || !fs::exists(xyz) || !fs::exists(calib)) {
        return std::nullopt;
    }
    return InputFiles{*cam, feature, xyz, calib};
}

std::vector<DatasetPaths> discover_datasets(const fs::path& problems_root,
                                            const std::string& dataset_filter) {
    std::vector<DatasetPaths> datasets;
    if (!fs::exists(problems_root)) {
        return datasets;
    }

    std::unordered_set<std::string> seen;
    auto add_dataset = [&](const std::string& base_name,
                           const std::string& problem_name,
                           const std::string& run_name,
                           const fs::path& base_root,
                           const fs::path& run_root,
                           const fs::path& reference_original) {
        if (!dataset_filter.empty() &&
            base_name != dataset_filter &&
            problem_name != dataset_filter &&
            run_name != dataset_filter) {
            return;
        }

        const auto files = find_input_files(run_root);
        if (!files) {
            std::cerr << "Skip malformed dataset: " << base_name
                      << " / " << run_name << "\n";
            return;
        }

        const std::string key = run_key(base_name, run_name, "", "");
        if (!seen.insert(key).second) {
            return;
        }

        datasets.push_back({base_name,
                            run_name,
                            base_root,
                            run_root,
                            reference_original,
                            files->cam,
                            files->feature,
                            files->xyz,
                            files->calib});
    };

    auto maybe_add_initial_value_problem = [&](const fs::path& problem_root,
                                               const std::string& category_name) {
        const fs::path initial_value = problem_root / "Initial Value";
        if (!fs::is_directory(initial_value)) {
            return;
        }
        const std::string problem_name = problem_root.filename().string();
        const std::string base_name =
            category_name.empty() ? problem_name : category_name + "__" + problem_name;
        add_dataset(base_name,
                    problem_name,
                    "Initial Value",
                    problem_root,
                    initial_value,
                    problem_root / "Ground Truth");
    };

    // Native BA Datasets layout:
    // <root>/<category>/<problem>/{Ground Truth, Initial Value}/...
    maybe_add_initial_value_problem(problems_root, "");
    for (const auto& first_entry : fs::directory_iterator(problems_root)) {
        if (!first_entry.is_directory()) {
            continue;
        }
        const std::string first_name = first_entry.path().filename().string();
        maybe_add_initial_value_problem(first_entry.path(), "");
        for (const auto& second_entry : fs::directory_iterator(first_entry.path())) {
            if (second_entry.is_directory()) {
                maybe_add_initial_value_problem(second_entry.path(), first_name);
            }
        }
    }

    // Prepared benchmark layout:
    // <root>/<base-dataset>/original and <root>/<base-dataset>/quality/<run-name>/...
    for (const auto& base_entry : fs::directory_iterator(problems_root)) {
        if (!base_entry.is_directory()) {
            continue;
        }

        const std::string base_name = base_entry.path().filename().string();
        const fs::path reference_original = base_entry.path() / "original";
        const fs::path quality_root = base_entry.path() / "quality";
        if (!fs::is_directory(quality_root)) {
            continue;
        }

        for (const auto& quality_entry : fs::directory_iterator(quality_root)) {
            if (!quality_entry.is_directory()) {
                continue;
            }

            const std::string quality_name = quality_entry.path().filename().string();
            if (!dataset_filter.empty() &&
                base_name != dataset_filter &&
                quality_name != dataset_filter) {
                continue;
            }

            add_dataset(base_name,
                        base_name,
                        quality_name,
                        base_entry.path(),
                        quality_entry.path(),
                        reference_original);
        }
    }

    std::sort(datasets.begin(), datasets.end(), [](const DatasetPaths& a, const DatasetPaths& b) {
        if (a.base_name != b.base_name) {
            return a.base_name < b.base_name;
        }
        return a.name < b.name;
    });
    return datasets;
}

}  // namespace

int main(int argc, char* argv[]) {
    fs::path problems_root = "E:/zuo/projects/CEP/BA Datasets";
    fs::path output_root = "E:/zuo/projects/CEP/benchmark_initial_value_clean";
    std::vector<MethodId> selected_methods = all_methods();
    std::string dataset_filter;
    BenchmarkOutputMode output_mode = BenchmarkOutputMode::CleanTiming;
    int point_condition_sample = 0;
    int schur_sample = 0;
    std::size_t limit = 0;
    bool resume = false;
    bool use_xyz = true;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--problems" && i + 1 < argc) {
            problems_root = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            output_root = argv[++i];
        } else if (arg == "--method" && i + 1 < argc) {
            MethodId method;
            if (!method_id_from_name(argv[++i], &method)) {
                std::cerr << "Unknown Method ID.\n";
                print_usage(argv[0]);
                return 2;
            }
            selected_methods = {method};
        } else if (arg == "--limit" && i + 1 < argc) {
            limit = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--dataset" && i + 1 < argc) {
            dataset_filter = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            if (!output_mode_from_name(argv[++i], &output_mode)) {
                std::cerr << "Unknown output mode.\n";
                print_usage(argv[0]);
                return 2;
            }
        } else if (arg == "--diagnostic") {
            output_mode = BenchmarkOutputMode::Diagnostic;
        } else if (arg == "--clean-timing") {
            output_mode = BenchmarkOutputMode::CleanTiming;
        } else if (arg == "--resume") {
            resume = true;
        } else if (arg == "--no-xyz") {
            use_xyz = false;
        } else if (arg == "--point-condition-sample" && i + 1 < argc) {
            point_condition_sample = std::max(0, std::stoi(argv[++i]));
        } else if (arg == "--schur-sample" && i + 1 < argc) {
            schur_sample = std::max(0, std::stoi(argv[++i]));
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            print_usage(argv[0]);
            return 2;
        }
    }

    const auto datasets = discover_datasets(problems_root, dataset_filter);
    if (datasets.empty()) {
        std::cerr << "No valid BA datasets found in " << problems_root << "\n";
        return 1;
    }

    fs::create_directories(output_root);
    const fs::path summary_path = output_root / "summary.csv";
    std::unordered_set<std::string> completed_runs;
    std::vector<std::string> retained_rows;
    if (resume && fs::exists(summary_path)) {
        std::ifstream existing_summary(summary_path);
        std::string line;
        std::getline(existing_summary, line);
        std::vector<std::string> fields;
        while (std::getline(existing_summary, line)) {
            if (!parse_summary_prefix(line, &fields) || fields[4] != "ok") {
                continue;
            }
            const std::string key = run_key(fields[0], fields[1], fields[2], fields[3]);
            if (completed_runs.insert(key).second) {
                retained_rows.push_back(line);
            }
        }
    }

    std::ofstream summary(summary_path, std::ios::trunc);
    if (!summary) {
        std::cerr << "Cannot open benchmark summary: " << summary_path << "\n";
        return 1;
    }
    summary << summary_header() << "\n";
    for (const auto& row : retained_rows) {
        summary << row << "\n";
    }
    summary.flush();
    if (resume) {
        std::cout << "Resume retained " << retained_rows.size()
                  << " successful runs from " << summary_path << "\n";
    }

    const std::size_t dataset_count =
        limit > 0 ? std::min(limit, datasets.size()) : datasets.size();
    const std::size_t total_runs = dataset_count * selected_methods.size();
    std::size_t processed = 0;
    std::size_t run_index = 0;
    std::size_t skipped_runs = 0;
    std::size_t attempted_runs = 0;
    for (const auto& dataset : datasets) {
        if (limit > 0 && processed >= limit) {
            break;
        }
        ++processed;

        for (const MethodId method : selected_methods) {
            ++run_index;
            const std::string method_name = method_id_name(method);
            const std::string mode_name = benchmark_output_mode_name(output_mode);
            const std::string key =
                run_key(dataset.base_name, dataset.name, method_name, mode_name);
            if (resume && completed_runs.find(key) != completed_runs.end()) {
                ++skipped_runs;
                continue;
            }
            ++attempted_runs;
            const bool diagnostics = output_mode == BenchmarkOutputMode::Diagnostic;
            const fs::path run_dir = output_root / dataset.base_name / dataset.name / method_name;
            if (diagnostics) {
                fs::create_directories(run_dir);
            }

            const fs::path report = run_dir / "report.txt";
            const fs::path pose = run_dir / "FinalPose.txt";
            const fs::path points = run_dir / "Final3D.ply";

            const std::string cam_s = dataset.cam.string();
            const std::string feature_s = dataset.feature.string();
            const std::string xyz_s = dataset.xyz.string();
            const std::string calib_s = dataset.calib.string();
            const std::string report_s = report.string();
            const std::string pose_s = pose.string();
            const std::string points_s = points.string();

            std::cout << "\n[" << run_index << "/" << total_runs << "] "
                      << dataset.base_name << " / " << dataset.name << " / "
                      << method_name << " (" << mode_name << ")" << std::endl;
            const auto t0 = std::chrono::steady_clock::now();
            bool ok = false;
            BARunMetrics metrics;
            try {
                BAExporter ba;
                ok = ba.ba_run(cam_s.c_str(),
                               feature_s.c_str(),
                               use_xyz ? xyz_s.c_str() : nullptr,
                               calib_s.c_str(),
                               diagnostics ? report_s.c_str() : nullptr,
                               diagnostics ? pose_s.c_str() : nullptr,
                               diagnostics ? points_s.c_str() : nullptr,
                               method,
                               output_mode,
                               diagnostics ? point_condition_sample : 0,
                               diagnostics ? schur_sample : 0);
                metrics = ba.last_metrics();
            } catch (const std::exception& ex) {
                std::cerr << "Run failed: " << ex.what() << "\n";
            }
            const auto t1 = std::chrono::steady_clock::now();
            const double elapsed = std::chrono::duration<double>(t1 - t0).count();

            summary << dataset.base_name << ','
                    << dataset.name << ','
                    << method_name << ','
                    << mode_name << ','
                    << (ok ? "ok" : "failed") << ','
                    << std::fixed << std::setprecision(6) << elapsed << ','
                    << metrics.solver_time_sec << ','
                    << metrics.linear_solver_time_sec << ','
                    << metrics.cameras << ','
                    << metrics.points << ','
                    << metrics.observations << ','
                    << std::setprecision(15)
                    << metrics.initial_cost << ','
                    << metrics.final_cost << ','
                    << metrics.initial_rmse_px << ','
                    << metrics.final_rmse_px << ','
                    << metrics.iterations << ','
                    << metrics.accepted_steps << ','
                    << metrics.rejected_steps << ','
                    << metrics.linear_solver_iterations << ','
                    << metrics.initial_gradient_max_norm << ','
                    << metrics.final_gradient_max_norm << ','
                    << metrics.final_gradient_norm << ','
                    << metrics.gradient_reduction_ratio_final << ','
                    << metrics.reached_gradient_tolerance << ','
                    << metrics.iterations_to_gradient_tolerance << ','
                    << metrics.final_relative_function_decrease << ','
                    << metrics.final_relative_step_size << ','
                    << metrics.final_lm_gain_ratio << ','
                    << metrics.final_gradient_lipschitz_estimate << ','
                    << metrics.final_direction_quality << ','
                    << metrics.termination_type << ','
                    << (diagnostics ? report.string() : "") << "\n";
            summary.flush();
            if (ok) {
                completed_runs.insert(key);
            }
        }
    }

    std::cout << "\nAttempted runs: " << attempted_runs << "\n"
              << "Skipped completed runs: " << skipped_runs << "\n"
              << "Benchmark summary: " << summary_path << "\n";
    return 0;
}
