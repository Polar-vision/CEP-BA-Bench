#include "PBAImp_v2.h"

#include <algorithm>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <Eigen/SVD>

#include "ceres/version.h"

namespace {

double clamp_acos_arg(double value) {
	return std::max(-1.0, std::min(1.0, value));
}

bool environment_flag(const char* name, bool fallback = false) {
	const char* value = std::getenv(name);
	if (value == nullptr) {
		return fallback;
	}
	const std::string text(value);
	if (text == "1" || text == "true" || text == "TRUE" ||
		text == "yes" || text == "YES") {
		return true;
	}
	if (text == "0" || text == "false" || text == "FALSE" ||
		text == "no" || text == "NO") {
		return false;
	}
	return fallback;
}

int environment_int(const char* name, int fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr) {
		return fallback;
	}
	try {
		return std::stoi(value);
	} catch (...) {
		return fallback;
	}
}

double environment_double(const char* name, double fallback) {
	const char* value = std::getenv(name);
	if (value == nullptr) {
		return fallback;
	}
	try {
		return std::stod(value);
	} catch (...) {
		return fallback;
	}
}

constexpr double kDiagnosticEpsilon = 1e-30;

struct DiagnosticParameterBlock {
	double* values = nullptr;
	int ambient_size = 0;
	int tangent_size = 0;
	const ceres::Manifold* manifold = nullptr;
};

struct DiagnosticIterationState {
	ceres::IterationSummary iteration;
	std::vector<double> parameters;
	double parameter_norm = 0.0;
};

struct StrictDiagnosticMetrics {
	bool vector_metrics_available = false;
	double initial_gradient_max_norm = 0.0;
	double final_gradient_reduction_ratio = 0.0;
	int reached_gradient_tolerance = 0;
	int iterations_to_gradient_tolerance = -1;
	double final_relative_function_decrease = 0.0;
	double final_relative_step_size = 0.0;
	double final_lm_gain_ratio = 0.0;
	double final_gradient_lipschitz_estimate = 0.0;
	double final_direction_quality = 0.0;
};

double nan_value() {
	return std::numeric_limits<double>::quiet_NaN();
}

double squared_norm(const std::vector<double>& values) {
	double sum = 0.0;
	for (double value : values) {
		sum += value * value;
	}
	return sum;
}

double l2_norm(const std::vector<double>& values) {
	return std::sqrt(squared_norm(values));
}

double dot_product(const std::vector<double>& lhs, const std::vector<double>& rhs) {
	if (lhs.size() != rhs.size()) {
		return nan_value();
	}
	double sum = 0.0;
	for (std::size_t i = 0; i < lhs.size(); ++i) {
		sum += lhs[i] * rhs[i];
	}
	return sum;
}

std::string diagnostic_parent_directory(const char* report_path) {
	if (report_path == nullptr) {
		return ".";
	}
	const std::string path(report_path);
	const size_t separator = path.find_last_of("/\\");
	return separator == std::string::npos ? "." : path.substr(0, separator);
}

std::vector<double> default_gradient_thresholds(double solver_gradient_tolerance) {
	std::vector<double> thresholds = {1e-2, 1e-4, 1e-6, 1e-8, 1e-10};
	if (solver_gradient_tolerance > 0.0 &&
		std::find(thresholds.begin(), thresholds.end(), solver_gradient_tolerance) ==
			thresholds.end()) {
		thresholds.push_back(solver_gradient_tolerance);
	}
	std::sort(thresholds.begin(), thresholds.end(), std::greater<double>());
	return thresholds;
}

std::vector<DiagnosticParameterBlock> collect_active_parameter_blocks(
	ceres::Problem& problem) {
	std::vector<double*> all_blocks;
	problem.GetParameterBlocks(&all_blocks);
	std::vector<DiagnosticParameterBlock> blocks;
	blocks.reserve(all_blocks.size());
	for (double* block : all_blocks) {
		if (block == nullptr || problem.IsParameterBlockConstant(block)) {
			continue;
		}
		const int tangent_size = problem.ParameterBlockTangentSize(block);
		if (tangent_size <= 0) {
			continue;
		}
		blocks.push_back(DiagnosticParameterBlock{
			block,
			problem.ParameterBlockSize(block),
			tangent_size,
			problem.GetManifold(block)});
	}
	return blocks;
}

std::size_t ambient_parameter_size(
	const std::vector<DiagnosticParameterBlock>& blocks) {
	std::size_t size = 0;
	for (const auto& block : blocks) {
		size += static_cast<std::size_t>(block.ambient_size);
	}
	return size;
}

std::size_t tangent_parameter_size(
	const std::vector<DiagnosticParameterBlock>& blocks) {
	std::size_t size = 0;
	for (const auto& block : blocks) {
		size += static_cast<std::size_t>(block.tangent_size);
	}
	return size;
}

void snapshot_parameters(
	const std::vector<DiagnosticParameterBlock>& blocks,
	std::vector<double>* values) {
	if (values == nullptr) {
		return;
	}
	values->clear();
	values->reserve(ambient_parameter_size(blocks));
	for (const auto& block : blocks) {
		values->insert(values->end(), block.values, block.values + block.ambient_size);
	}
}

double active_parameter_norm(const std::vector<double>& values) {
	return l2_norm(values);
}

void restore_parameters(
	const std::vector<DiagnosticParameterBlock>& blocks,
	const std::vector<double>& values) {
	std::size_t offset = 0;
	for (const auto& block : blocks) {
		if (offset + static_cast<std::size_t>(block.ambient_size) > values.size()) {
			return;
		}
		std::copy(values.begin() + offset,
			values.begin() + offset + block.ambient_size,
			block.values);
		offset += static_cast<std::size_t>(block.ambient_size);
	}
}

bool tangent_delta(
	const std::vector<DiagnosticParameterBlock>& blocks,
	const std::vector<double>& previous,
	const std::vector<double>& current,
	std::vector<double>* delta) {
	if (delta == nullptr || previous.size() != current.size()) {
		return false;
	}
	delta->assign(tangent_parameter_size(blocks), 0.0);
	std::size_t ambient_offset = 0;
	std::size_t tangent_offset = 0;
	for (const auto& block : blocks) {
		if (ambient_offset + static_cast<std::size_t>(block.ambient_size) >
			previous.size()) {
			return false;
		}
		if (block.manifold != nullptr) {
			if (!block.manifold->Minus(
					current.data() + ambient_offset,
					previous.data() + ambient_offset,
					delta->data() + tangent_offset)) {
				return false;
			}
		} else {
			for (int i = 0; i < block.ambient_size; ++i) {
				(*delta)[tangent_offset + i] =
					current[ambient_offset + i] - previous[ambient_offset + i];
			}
		}
		ambient_offset += static_cast<std::size_t>(block.ambient_size);
		tangent_offset += static_cast<std::size_t>(block.tangent_size);
	}
	return true;
}

class StrictDiagnosticIterationCallback final : public ceres::IterationCallback {
public:
	explicit StrictDiagnosticIterationCallback(
		std::vector<DiagnosticParameterBlock> blocks)
		: blocks_(std::move(blocks)) {}

	ceres::CallbackReturnType operator()(
		const ceres::IterationSummary& summary) override {
		DiagnosticIterationState state;
		state.iteration = summary;
		snapshot_parameters(blocks_, &state.parameters);
		state.parameter_norm = active_parameter_norm(state.parameters);
		states_.push_back(std::move(state));
		return ceres::SOLVER_CONTINUE;
	}

	const std::vector<DiagnosticIterationState>& states() const {
		return states_;
	}

private:
	std::vector<DiagnosticParameterBlock> blocks_;
	std::vector<DiagnosticIterationState> states_;
};

StrictDiagnosticMetrics write_strict_diagnostics(
	ceres::Problem& problem,
	const std::vector<DiagnosticParameterBlock>& blocks,
	const std::vector<DiagnosticIterationState>& states,
	const ceres::Solver::Summary& summary,
	int nobs,
	double gradient_tolerance,
	const std::string& parent) {
	StrictDiagnosticMetrics result;
	if (summary.iterations.empty()) {
		return result;
	}

	const double initial_gradient =
		summary.iterations.front().gradient_max_norm;
	result.initial_gradient_max_norm = initial_gradient;
	if (!summary.iterations.empty()) {
		const auto& final_iteration = summary.iterations.back();
		result.final_gradient_reduction_ratio =
			(initial_gradient - final_iteration.gradient_max_norm) /
			(initial_gradient + kDiagnosticEpsilon);
		result.final_lm_gain_ratio = final_iteration.relative_decrease;
	}

	const std::vector<double> thresholds =
		default_gradient_thresholds(gradient_tolerance);
	const std::string thresholds_path = parent + "/gradient_thresholds.csv";
	FILE* threshold_fp = nullptr;
	fopen_s(&threshold_fp, thresholds_path.c_str(), "w");
	if (threshold_fp != nullptr) {
		fprintf(threshold_fp,
			"threshold,reached,iteration,cost,rmse_px\n");
		for (double threshold : thresholds) {
			int reached = 0;
			int iteration = -1;
			double cost = nan_value();
			double rmse = nan_value();
			for (const auto& it : summary.iterations) {
				if (it.gradient_max_norm <= threshold) {
					reached = 1;
					iteration = it.iteration;
					cost = it.cost;
					rmse = nobs > 0 ? std::sqrt(2.0 * it.cost / nobs) : nan_value();
					break;
				}
			}
			if (threshold == gradient_tolerance) {
				result.reached_gradient_tolerance = reached;
				result.iterations_to_gradient_tolerance = iteration;
			}
			fprintf(threshold_fp, "%.17g,%d,%d,%.17g,%.17g\n",
				threshold, reached, iteration, cost, rmse);
		}
		fclose(threshold_fp);
	}

	std::vector<std::vector<double>> gradients;
	bool vector_metrics_available =
		!blocks.empty() && states.size() == summary.iterations.size();
	std::vector<double> final_parameters;
	if (vector_metrics_available) {
		snapshot_parameters(blocks, &final_parameters);
		gradients.reserve(states.size());
		ceres::Problem::EvaluateOptions evaluation_options;
		evaluation_options.apply_loss_function = true;
		evaluation_options.num_threads = 1;
		evaluation_options.parameter_blocks.reserve(blocks.size());
		for (const auto& block : blocks) {
			evaluation_options.parameter_blocks.push_back(block.values);
		}
		for (const auto& state : states) {
			restore_parameters(blocks, state.parameters);
			std::vector<double> gradient;
			double evaluated_cost = 0.0;
			if (!problem.Evaluate(
					evaluation_options, &evaluated_cost, nullptr, &gradient, nullptr) ||
				gradient.size() != tangent_parameter_size(blocks)) {
				vector_metrics_available = false;
				break;
			}
			gradients.push_back(std::move(gradient));
		}
		restore_parameters(blocks, final_parameters);
	}
	result.vector_metrics_available = vector_metrics_available;

	const double final_cost = summary.final_cost;
	const double initial_cost = summary.initial_cost;
	const double total_cost_reduction =
		std::max(std::abs(initial_cost - final_cost), kDiagnosticEpsilon);

	const std::string convergence_path = parent + "/convergence.txt";
	FILE* convergence_fp = nullptr;
	fopen_s(&convergence_fp, convergence_path.c_str(), "w");
	const std::string strict_path = parent + "/convergence_strict.csv";
	FILE* strict_fp = nullptr;
	fopen_s(&strict_fp, strict_path.c_str(), "w");
	const char* header =
		"iteration,cost,rmse_px,cost_change,relative_function_decrease,"
		"normalized_convergence_progress,gradient_max_norm,gradient_norm,"
		"gradient_reduction_ratio,step_norm,step_tangent_norm,x_norm,"
		"relative_step_size,gradient_lipschitz_estimate,direction_quality,"
		"lm_gain_ratio,gain_ratio,trust_region_radius,lm_damping,"
		"linear_solver_eta,linear_solver_iterations,step_valid,step_successful\n";
	if (convergence_fp != nullptr) {
		fprintf(convergence_fp, "%s", header);
	}
	if (strict_fp != nullptr) {
		fprintf(strict_fp, "%s", header);
	}

	for (std::size_t i = 0; i < summary.iterations.size(); ++i) {
		const auto& it = summary.iterations[i];
		const double rmse = nobs > 0 ? std::sqrt(2.0 * it.cost / nobs) : nan_value();
		const double previous_cost =
			i > 0 ? summary.iterations[i - 1].cost : it.cost;
		const double relative_function_decrease =
			i > 0 ? (previous_cost - it.cost) /
				std::max(std::abs(previous_cost), kDiagnosticEpsilon) : 0.0;
		const double normalized_progress =
			(initial_cost - it.cost) / total_cost_reduction;
		const double gradient_reduction_ratio =
			(initial_gradient - it.gradient_max_norm) /
			(initial_gradient + kDiagnosticEpsilon);
		const double lm_damping =
			it.trust_region_radius > 0.0 ? 1.0 / it.trust_region_radius : 0.0;
		double x_norm = nan_value();
		double step_tangent_norm = nan_value();
		double relative_step_size = nan_value();
		double lipschitz = nan_value();
		double direction_quality = nan_value();
		if (i < states.size()) {
			x_norm = states[i].parameter_norm;
		}
		if (i > 0 && i < states.size()) {
			std::vector<double> delta;
			if (tangent_delta(blocks, states[i - 1].parameters,
					states[i].parameters, &delta)) {
				step_tangent_norm = l2_norm(delta);
				const double previous_norm = states[i - 1].parameter_norm;
				relative_step_size =
					step_tangent_norm / (previous_norm + kDiagnosticEpsilon);
				if (vector_metrics_available && gradients.size() == states.size()) {
					std::vector<double> gradient_delta(gradients[i].size(), 0.0);
					for (std::size_t j = 0; j < gradient_delta.size(); ++j) {
						gradient_delta[j] = gradients[i][j] - gradients[i - 1][j];
					}
					lipschitz =
						l2_norm(gradient_delta) /
						(step_tangent_norm + kDiagnosticEpsilon);
					direction_quality =
						-dot_product(gradients[i - 1], delta) /
						((l2_norm(gradients[i - 1]) * step_tangent_norm) +
						 kDiagnosticEpsilon);
				}
			}
		}
		if (i + 1 == summary.iterations.size()) {
			result.final_relative_function_decrease =
				relative_function_decrease;
			result.final_relative_step_size = relative_step_size;
			result.final_gradient_lipschitz_estimate = lipschitz;
			result.final_direction_quality = direction_quality;
		}

		const auto write_row = [&](FILE* fp) {
			if (fp == nullptr) {
				return;
			}
			fprintf(fp,
				"%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
				"%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
				"%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%d\n",
				it.iteration,
				it.cost,
				rmse,
				it.cost_change,
				relative_function_decrease,
				normalized_progress,
				it.gradient_max_norm,
				it.gradient_norm,
				gradient_reduction_ratio,
				it.step_norm,
				step_tangent_norm,
				x_norm,
				relative_step_size,
				lipschitz,
				direction_quality,
				it.relative_decrease,
				it.relative_decrease,
				it.trust_region_radius,
				lm_damping,
				it.eta,
				it.linear_solver_iterations,
				it.step_is_valid ? 1 : 0,
				it.step_is_successful ? 1 : 0);
		};
		write_row(convergence_fp);
		write_row(strict_fp);
	}
	if (convergence_fp != nullptr) {
		fclose(convergence_fp);
	}
	if (strict_fp != nullptr) {
		fclose(strict_fp);
	}

	return result;
}

enum class AnchorPolicy {
	kCurrent,
	kCurrentPairPaperOrder,
	kCurrentPairPaperOrderRefine,
	kTrueMaxParallax,
	kMaxParallaxReproj,
	kScoreReproj,
	kScoreBaselineReproj
};

std::string normalized_environment_string(const char* name) {
	const char* value = std::getenv(name);
	if (value == nullptr) {
		return "";
	}
	std::string text(value);
	for (char& ch : text) {
		if (ch == '-' || ch == '.' || ch == ' ') {
			ch = '_';
		} else {
			ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
		}
	}
	return text;
}

AnchorPolicy anchor_policy_from_environment() {
	const std::string text = normalized_environment_string("CEP_ANCHOR_POLICY");
	if (text.empty() || text == "current" || text == "default") {
		return AnchorPolicy::kCurrent;
	}
	if (text == "current_pair_paper_order" || text == "paper_order") {
		return AnchorPolicy::kCurrentPairPaperOrder;
	}
	if (text == "current_pair_paper_order_refine" || text == "paper_order_refine") {
		return AnchorPolicy::kCurrentPairPaperOrderRefine;
	}
	if (text == "true_max_parallax" || text == "max_parallax") {
		return AnchorPolicy::kTrueMaxParallax;
	}
	if (text == "max_parallax_reproj") {
		return AnchorPolicy::kMaxParallaxReproj;
	}
	if (text == "score_reproj") {
		return AnchorPolicy::kScoreReproj;
	}
	if (text == "score_baseline_reproj") {
		return AnchorPolicy::kScoreBaselineReproj;
	}
	return AnchorPolicy::kCurrent;
}

}  // namespace

PBA::PBA(void)
{
	m_ncams = m_n3Dpts = m_n2Dprojs = m_nS = nc_ = 0;
	m_archor = nullptr;
	m_photo = m_feature = nullptr;
	m_motstruct = m_imgpts = nullptr;
	m_XYZ = nullptr;
	m_K = nullptr;
	m_V = nullptr;
	m_vmask = m_umask = m_smask = nullptr;
	m_imgptsSum = m_struct = m_pnt2main = m_archorSort = nullptr;
	m_KR = m_KdA = m_KdB = m_KdG = m_P = nullptr;
	m_bProvideXYZ = false;
	m_bFocal = false;
	m_szCameraInit = m_szFeatures = m_szCalibration = m_szXYZ = m_sz3Dpts = m_szCamePose = m_szReport = nullptr;
	m_last_metrics = BARunMetrics{};
}


PBA::~PBA(void)
{
	free(m_archor);
	free(m_photo);
	free(m_feature);
	free(m_motstruct);
	free(m_imgpts);
	free(m_XYZ);
	free(m_K);
	free(m_V);
	free(m_vmask);
	free(m_umask);
	free(m_smask);
	free(m_imgptsSum);
	free(m_struct);
	free(m_pnt2main);
	free(m_archorSort);
	free(m_KR);
	free(m_KdA);
	free(m_KdB);
	free(m_KdG);
	free(m_P);
}

void PBA::parallax2xyz(){
	double xj[3], xk[3];
	double Tik[3];
	int nM, nA;
	double Dik;
	double w, w2;
	for (int i = 0; i < points.size(); i++)
	{
		w = points[i].parallax_world[2];

		xj[0] = sin(points[i].parallax_world[0]) * cos(points[i].parallax_world[1]);
		xj[1] = sin(points[i].parallax_world[1]);
		xj[2] = cos(points[i].parallax_world[0]) * cos(points[i].parallax_world[1]);

		nM = points[i].nM;
		nA = points[i].nA;

		Tik[0] = cams[nA].camera_center[0]-cams[nM].camera_center[0];
		Tik[1] = cams[nA].camera_center[1]-cams[nM].camera_center[1];
		Tik[2] = cams[nA].camera_center[2]-cams[nM].camera_center[2];
		
		Dik = sqrt(Tik[0] * Tik[0] + Tik[1] * Tik[1] + Tik[2] * Tik[2]);
		
		w2 = acos(clamp_acos_arg((xj[0] * Tik[0] + xj[1] * Tik[1] + xj[2] * Tik[2]) / Dik));

		xk[0] = (Dik * sin(w2 + w) * xj[0]) / sin(w);
		xk[1] = (Dik * sin(w2 + w) * xj[1]) / sin(w);
		xk[2] = (Dik * sin(w2 + w) * xj[2]) / sin(w);

		points[i].xyz[0] = cams[nM].camera_center[0] + xk[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + xk[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + xk[2];
	}
}
void PBA::xy_inverse_z2xyz(){
	for(int i=0;i<points.size();i++){
		double rho = points[i].xy_inverse_z[2];
		if (std::abs(rho) < 1e-12) {
			rho = std::copysign(1e-12, rho == 0.0 ? 1.0 : rho);
		}
		points[i].xyz[0] = points[i].xy_inverse_z[0] / rho;
		points[i].xyz[1] = points[i].xy_inverse_z[1] / rho;
		points[i].xyz[2] = 1 / rho;
	}
}
void PBA::depth2xyz(){
	double xj[3];
	for(int i=0;i<points.size();i++){
		xj[0] = sin(points[i].world_depth[0]) * cos(points[i].world_depth[1]);
		xj[1] = sin(points[i].world_depth[1]);
		xj[2] = cos(points[i].world_depth[0]) * cos(points[i].world_depth[1]);

		points[i].xyz[0] = xj[0] * points[i].world_depth[2];
		points[i].xyz[1] = xj[1] * points[i].world_depth[2];
		points[i].xyz[2] = xj[2] * points[i].world_depth[2];
	}
}
void PBA::inverse_depth2xyz(){
	double xj[3];
	for(int i=0;i<points.size();i++){
		xj[0] = sin(points[i].world_inverse_depth[0]) * cos(points[i].world_inverse_depth[1]);
		xj[1] = sin(points[i].world_inverse_depth[1]);
		xj[2] = cos(points[i].world_inverse_depth[0]) * cos(points[i].world_inverse_depth[1]);

		points[i].xyz[0] = xj[0] / points[i].world_inverse_depth[2];
		points[i].xyz[1] = xj[1] / points[i].world_inverse_depth[2];
		points[i].xyz[2] = xj[2] / points[i].world_inverse_depth[2];
	}
}
void PBA::archored_inverse_depth2xyz(){
	double xj[3],xk[3];
	int nM;
	for(int i=0;i<points.size();i++){
		xj[0] = sin(points[i].archored_spherical_inverse_range[0]) *
			cos(points[i].archored_spherical_inverse_range[1]);
		xj[1] = sin(points[i].archored_spherical_inverse_range[1]);
		xj[2] = cos(points[i].archored_spherical_inverse_range[0]) *
			cos(points[i].archored_spherical_inverse_range[1]);

		nM = points[i].nM;
		xk[0] = xj[0] / points[i].archored_spherical_inverse_range[2];
		xk[1] = xj[1] / points[i].archored_spherical_inverse_range[2];
		xk[2] = xj[2] / points[i].archored_spherical_inverse_range[2];

		points[i].xyz[0] = cams[nM].camera_center[0] + xk[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + xk[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + xk[2];
	}
}
void PBA::archored_depth2xyz(){
	double xj[3],xk[3];
	int nM;
	for(int i=0;i<points.size();i++){
		xj[0] = sin(points[i].archored_spherical_range[0]) *
			cos(points[i].archored_spherical_range[1]);
		xj[1] = sin(points[i].archored_spherical_range[1]);
		xj[2] = cos(points[i].archored_spherical_range[0]) *
			cos(points[i].archored_spherical_range[1]);

		nM = points[i].nM;
		xk[0] = xj[0] * points[i].archored_spherical_range[2];
		xk[1] = xj[1] * points[i].archored_spherical_range[2];
		xk[2] = xj[2] * points[i].archored_spherical_range[2];

		points[i].xyz[0] = cams[nM].camera_center[0] + xk[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + xk[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + xk[2];
	}
}
void PBA::archored_xy_inverse_z2xyz(){
	int nM;
	for(int i=0;i<points.size();i++){
		nM = points[i].nM;
		double rho = points[i].archored_xy_inverse_z[2];
		if (std::abs(rho) < 1e-12) {
			rho = std::copysign(1e-12, rho == 0.0 ? 1.0 : rho);
		}
		points[i].xyz[0] = cams[nM].camera_center[0] +
			points[i].archored_xy_inverse_z[0] / rho;
		points[i].xyz[1] = cams[nM].camera_center[1] +
			points[i].archored_xy_inverse_z[1] / rho;
		points[i].xyz[2] = cams[nM].camera_center[2] + 1 / rho;
	}
}
void PBA::archored_xyz2xyz(){
	int nM;
	for(int i=0;i<points.size();i++){
		nM = points[i].nM;
		points[i].xyz[0] = cams[nM].camera_center[0] + points[i].archored_xyz[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + points[i].archored_xyz[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + points[i].archored_xyz[2];
	}
}
void PBA::anchor_camera_xyz2xyz() {
	for (int i = 0; i < static_cast<int>(points.size()); ++i) {
		const int nM = points[i].nM;
		double rotation_main[9];
		double point_world_relative[3];
		cep::cost::EulerToWorldToCamera(cams[nM].euler_angle, rotation_main);
		cep::cost::MatTransposeVec(rotation_main,
			points[i].anchor_camera_xyz,
			point_world_relative);
		points[i].xyz[0] = cams[nM].camera_center[0] + point_world_relative[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + point_world_relative[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + point_world_relative[2];
	}
}
void PBA::anchor_camera_xy_inverse_z2xyz() {
	for (int i = 0; i < static_cast<int>(points.size()); ++i) {
		const int nM = points[i].nM;
		double rho = points[i].anchor_camera_xy_inverse_z[2];
		if (std::abs(rho) < 1e-12) {
			rho = std::copysign(1e-12, rho == 0.0 ? 1.0 : rho);
		}
		double point_camera[3] = {
			points[i].anchor_camera_xy_inverse_z[0] / rho,
			points[i].anchor_camera_xy_inverse_z[1] / rho,
			1.0 / rho,
		};
		double rotation_main[9];
		double point_world_relative[3];
		cep::cost::EulerToWorldToCamera(cams[nM].euler_angle, rotation_main);
		cep::cost::MatTransposeVec(rotation_main, point_camera, point_world_relative);
		points[i].xyz[0] = cams[nM].camera_center[0] + point_world_relative[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + point_world_relative[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + point_world_relative[2];
	}
}
void PBA::anchor_camera_spherical_range2xyz() {
	for (int i = 0; i < static_cast<int>(points.size()); ++i) {
		const int nM = points[i].nM;
		double bearing[3];
		double point_camera[3];
		double rotation_main[9];
		double point_world_relative[3];
		cep::cost::BearingFromAngles(
			points[i].anchor_camera_spherical_range[0],
			points[i].anchor_camera_spherical_range[1],
			bearing);
		point_camera[0] = bearing[0] * points[i].anchor_camera_spherical_range[2];
		point_camera[1] = bearing[1] * points[i].anchor_camera_spherical_range[2];
		point_camera[2] = bearing[2] * points[i].anchor_camera_spherical_range[2];
		cep::cost::EulerToWorldToCamera(cams[nM].euler_angle, rotation_main);
		cep::cost::MatTransposeVec(rotation_main, point_camera, point_world_relative);
		points[i].xyz[0] = cams[nM].camera_center[0] + point_world_relative[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + point_world_relative[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + point_world_relative[2];
	}
}
void PBA::anchor_camera_spherical_inverse_range2xyz() {
	for (int i = 0; i < static_cast<int>(points.size()); ++i) {
		const int nM = points[i].nM;
		double rho = points[i].anchor_camera_spherical_inverse_range[2];
		if (std::abs(rho) < 1e-12) {
			rho = std::copysign(1e-12, rho == 0.0 ? 1.0 : rho);
		}
		double bearing[3];
		double point_camera[3];
		double rotation_main[9];
		double point_world_relative[3];
		cep::cost::BearingFromAngles(
			points[i].anchor_camera_spherical_inverse_range[0],
			points[i].anchor_camera_spherical_inverse_range[1],
			bearing);
		point_camera[0] = bearing[0] / rho;
		point_camera[1] = bearing[1] / rho;
		point_camera[2] = bearing[2] / rho;
		cep::cost::EulerToWorldToCamera(cams[nM].euler_angle, rotation_main);
		cep::cost::MatTransposeVec(rotation_main, point_camera, point_world_relative);
		points[i].xyz[0] = cams[nM].camera_center[0] + point_world_relative[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + point_world_relative[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + point_world_relative[2];
	}
}
void PBA::parallax_camera2xyz() {
	for (int i = 0; i < static_cast<int>(points.size()); ++i) {
		const int nM = points[i].nM;
		const int nA = points[i].nA;
		double rotation_main[9];
		double bearing_main[3];
		double baseline_world[3] = {
			cams[nA].camera_center[0] - cams[nM].camera_center[0],
			cams[nA].camera_center[1] - cams[nM].camera_center[1],
			cams[nA].camera_center[2] - cams[nM].camera_center[2],
		};
		double baseline_main[3];
		cep::cost::EulerToWorldToCamera(cams[nM].euler_angle, rotation_main);
		cep::cost::BearingFromAngles(
			points[i].parallax_camera[0],
			points[i].parallax_camera[1],
			bearing_main);
		cep::cost::MatVec(rotation_main, baseline_world, baseline_main);
		const double baseline_norm = sqrt(
			baseline_main[0] * baseline_main[0] +
			baseline_main[1] * baseline_main[1] +
			baseline_main[2] * baseline_main[2]);
		if (baseline_norm < 1e-12) {
			continue;
		}
		const double cos_beta = clamp_acos_arg(
			(bearing_main[0] * baseline_main[0] +
			 bearing_main[1] * baseline_main[1] +
			 bearing_main[2] * baseline_main[2]) /
			baseline_norm);
		const double beta = acos(cos_beta);
		const double omega = points[i].parallax_camera[2];
		double sin_omega = sin(omega);
		if (std::abs(sin_omega) < 1e-12) {
			sin_omega = std::copysign(1e-12, sin_omega == 0.0 ? 1.0 : sin_omega);
		}
		const double range =
			baseline_norm * sin(beta + omega) / sin_omega;
		double point_main[3] = {
			range * bearing_main[0],
			range * bearing_main[1],
			range * bearing_main[2],
		};
		double point_world_relative[3];
		cep::cost::MatTransposeVec(rotation_main, point_main, point_world_relative);
		points[i].xyz[0] = cams[nM].camera_center[0] + point_world_relative[0];
		points[i].xyz[1] = cams[nM].camera_center[1] + point_world_relative[1];
		points[i].xyz[2] = cams[nM].camera_center[2] + point_world_relative[2];
	}
}
bool PBA::ba_run(char* szCam,
	char* szFea,
	char* szXYZ,
	char* szCalib,
	char* szReport,
	char* szPose,
	char* sz3D,
    MethodId method,
	BenchmarkOutputMode output_mode,
	int point_condition_sample,
	int schur_sample)
{
	m_last_metrics = BARunMetrics{};
	m_szCameraInit = szCam;
	m_szFeatures = szFea;
	m_szCalibration = szCalib;
	m_szXYZ = szXYZ;
	m_szCamePose = szPose;
	m_sz3Dpts = sz3D;
	m_szReport = szReport;

	const bool diagnostics = output_mode == BenchmarkOutputMode::Diagnostic;
	const objectpointtype optype = method_object_point_type(method);
	const char* method_name = method_id_name(method);

	if (diagnostics) {
		printf("%s\n", m_szXYZ != nullptr ? m_szXYZ : "");
		printf("object-point parameterization benchmark\n");
		printf("method: %s\n", method_name);
	}

	if (!ba_initialize(m_szCameraInit, m_szFeatures, m_szCalibration, m_szXYZ)) {
		return false;
	}

	ceres::Problem problem;
	int nobs = 0;

	for (int i = 0; i < static_cast<int>(tracks.size()); i++) {
		const int nM = points[i].nM;
		const int nA = points[i].nA;

		for (int j = 0; j < tracks[i].nview; j++) {
			const int nP = tracks[i].obss[j].view_idx;
			const double u = tracks[i].obss[j].u;
			const double v = tracks[i].obss[j].v;

			if (nP < 0 || nP >= static_cast<int>(cams.size())) {
				fprintf(stderr, "BA error: observation references invalid camera index %d\n", nP);
				return false;
			}

			const int cam_idx = cams[nP].camidx;
			if (cam_idx <= 0 || cam_idx > static_cast<int>(intrs.size())) {
				fprintf(stderr, "BA error: camera %d references invalid calibration index %d\n", nP, cam_idx);
				return false;
			}

			const double fx = intrs[cam_idx - 1].fx;
			const double fy = intrs[cam_idx - 1].fy;
			const double cx = intrs[cam_idx - 1].cx;
			const double cy = intrs[cam_idx - 1].cy;
			nobs++;

			switch (optype) {
			case xyz: {
				ceres::CostFunction* cost_function = xyz_euler_angle_uv::Create(u, v, fx, fy, cx, cy);
				problem.AddResidualBlock(cost_function, nullptr,
					&cams[nP].euler_angle[0],
					&cams[nP].camera_center[0],
					&points[i].xyz[0]);
				break;
			}
			case xy_inverse_z: {
				ceres::CostFunction* cost_function = xy_inverse_z_euler_angle_uv::Create(u, v, fx, fy, cx, cy);
				problem.AddResidualBlock(cost_function, nullptr,
					&cams[nP].euler_angle[0],
					&cams[nP].camera_center[0],
					&points[i].xy_inverse_z[0]);
				break;
			}
			case depth: {
				ceres::CostFunction* cost_function = depth_euler_angle_uv::Create(u, v, fx, fy, cx, cy);
				problem.AddResidualBlock(cost_function, nullptr,
					&cams[nP].euler_angle[0],
					&cams[nP].camera_center[0],
					&points[i].world_depth[0]);
				break;
			}
			case inverse_depth: {
				ceres::CostFunction* cost_function = inverse_depth_euler_angle_uv::Create(u, v, fx, fy, cx, cy);
				problem.AddResidualBlock(cost_function, nullptr,
					&cams[nP].euler_angle[0],
					&cams[nP].camera_center[0],
					&points[i].world_inverse_depth[0]);
				break;
			}
			case anchor_camera_xyz: {
				if (nP == nM) {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraMainResidual<
							cep::cost::kAnchorCameraXYZ>::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(
						cost_function, nullptr, &points[i].anchor_camera_xyz[0]);
				} else {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraOtherResidual<
							cep::cost::kAnchorCameraXYZ>::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].euler_angle[0],
						&cams[nM].camera_center[0],
						&points[i].anchor_camera_xyz[0]);
				}
				break;
			}
			case anchor_camera_xy_inverse_z: {
				if (nP == nM) {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraMainResidual<
							cep::cost::kAnchorCameraXYInverseZ>::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(
						cost_function, nullptr, &points[i].anchor_camera_xy_inverse_z[0]);
				} else {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraOtherResidual<
							cep::cost::kAnchorCameraXYInverseZ>::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].euler_angle[0],
						&cams[nM].camera_center[0],
						&points[i].anchor_camera_xy_inverse_z[0]);
				}
				break;
			}
			case anchor_camera_spherical_range: {
				if (nP == nM) {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraMainResidual<
							cep::cost::kAnchorCameraSphericalRange>::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(
						cost_function, nullptr, &points[i].anchor_camera_spherical_range[0]);
				} else {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraOtherResidual<
							cep::cost::kAnchorCameraSphericalRange>::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].euler_angle[0],
						&cams[nM].camera_center[0],
						&points[i].anchor_camera_spherical_range[0]);
				}
				break;
			}
			case anchor_camera_spherical_inverse_range: {
				if (nP == nM) {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraMainResidual<
							cep::cost::kAnchorCameraSphericalInverseRange>::Create(
							u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(
						cost_function, nullptr, &points[i].anchor_camera_spherical_inverse_range[0]);
				} else {
					ceres::CostFunction* cost_function =
						cep::cost::AnchorCameraOtherResidual<
							cep::cost::kAnchorCameraSphericalInverseRange>::Create(
							u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].euler_angle[0],
						&cams[nM].camera_center[0],
						&points[i].anchor_camera_spherical_inverse_range[0]);
				}
				break;
			}
			case archored_xyz: {
				if (nP == nM) {
					ceres::CostFunction* cost_function = archored_xyz_euler_angle_uv_nM::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&points[i].archored_xyz[0]);
				} else {
					ceres::CostFunction* cost_function = archored_xyz_euler_angle_uv_nP::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].camera_center[0],
						&points[i].archored_xyz[0]);
				}
				break;
			}
			case archored_xy_inverse_z: {
				if (nP == nM) {
					ceres::CostFunction* cost_function = archored_xy_inverse_z_euler_angle_uv_nM::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&points[i].archored_xy_inverse_z[0]);
				} else {
					ceres::CostFunction* cost_function = archored_xy_inverse_z_euler_angle_uv_nP::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].camera_center[0],
						&points[i].archored_xy_inverse_z[0]);
				}
				break;
			}
			case archored_depth: {
				if (nP == nM) {
					ceres::CostFunction* cost_function = archored_depth_euler_angle_uv_nM::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&points[i].archored_spherical_range[0]);
				} else {
					ceres::CostFunction* cost_function = archored_depth_euler_angle_uv_nP::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].camera_center[0],
						&points[i].archored_spherical_range[0]);
				}
				break;
			}
			case archored_inverse_depth: {
				if (nP == nM) {
					ceres::CostFunction* cost_function = archored_inverse_depth_euler_angle_uv_nM::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&points[i].archored_spherical_inverse_range[0]);
				} else {
					ceres::CostFunction* cost_function = archored_inverse_depth_euler_angle_uv_nP::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].camera_center[0],
						&points[i].archored_spherical_inverse_range[0]);
				}
				break;
			}
			case parallax: {
				if (nP == nM) {
					ceres::CostFunction* cost_function = parallax_euler_angle_uv_nM::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&points[i].parallax_world[0]);
				} else if (nP == nA) {
					ceres::CostFunction* cost_function = parallax_euler_angle_uv_nA::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].camera_center[0],
						&points[i].parallax_world[0]);
				} else {
					ceres::CostFunction* cost_function = parallax_euler_angle_uv_nP::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].camera_center[0],
						&cams[nA].camera_center[0],
						&points[i].parallax_world[0]);
				}
				break;
			}
			case parallax_camera: {
				if (nP == nM) {
					ceres::CostFunction* cost_function =
						cep::cost::ParallaxCameraMainResidual::Create(u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(
						cost_function, nullptr, &points[i].parallax_camera[0]);
				} else if (nP == nA) {
					ceres::CostFunction* cost_function =
						cep::cost::ParallaxCameraAssociateResidual::Create(
							u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].euler_angle[0],
						&cams[nM].camera_center[0],
						&points[i].parallax_camera[0]);
				} else {
					ceres::CostFunction* cost_function =
						cep::cost::ParallaxCameraOtherResidual::Create(
							u, v, fx, fy, cx, cy);
					problem.AddResidualBlock(cost_function, nullptr,
						&cams[nP].euler_angle[0],
						&cams[nP].camera_center[0],
						&cams[nM].euler_angle[0],
						&cams[nM].camera_center[0],
						&cams[nA].camera_center[0],
						&points[i].parallax_camera[0]);
				}
				break;
			}
			default:
				fprintf(stderr, "BA error: unsupported object-point method %s\n", method_name);
				return false;
			}
		}
	}

	if (nobs == 0) {
		fprintf(stderr, "BA error: no observations were loaded\n");
		return false;
	}

	if (environment_flag("CEP_FIX_MONOCULAR_GAUGE", true) && !cams.empty()) {
		if (problem.HasParameterBlock(&cams[0].euler_angle[0])) {
			problem.SetParameterBlockConstant(&cams[0].euler_angle[0]);
		}
		if (problem.HasParameterBlock(&cams[0].camera_center[0])) {
			problem.SetParameterBlockConstant(&cams[0].camera_center[0]);
		}
		int scale_camera = -1;
		for (int camera_index = 1; camera_index < static_cast<int>(cams.size()); ++camera_index) {
			if (!problem.HasParameterBlock(&cams[camera_index].camera_center[0])) {
				continue;
			}
			double baseline_squared = 0.0;
			for (int axis = 0; axis < 3; ++axis) {
				const double component =
					cams[camera_index].camera_center[axis] - cams[0].camera_center[axis];
				baseline_squared += component * component;
			}
			if (baseline_squared > 1e-24) {
				scale_camera = camera_index;
				break;
			}
		}
		if (scale_camera >= 0) {
			int scale_axis = 0;
			double maximum_baseline_component = 0.0;
			for (int axis = 0; axis < 3; ++axis) {
				const double component = std::abs(
					cams[scale_camera].camera_center[axis] - cams[0].camera_center[axis]);
				if (component > maximum_baseline_component) {
					maximum_baseline_component = component;
					scale_axis = axis;
				}
			}
			problem.SetManifold(
				&cams[scale_camera].camera_center[0],
				new ceres::SubsetManifold(3, std::vector<int>{scale_axis}));
		}
	}

	ceres::Solver::Options options;
	options.logging_type = ceres::SILENT;
	options.minimizer_progress_to_stdout = false;
	const unsigned int hw_threads = std::thread::hardware_concurrency();
	options.num_threads = environment_int(
		"CEP_NUM_THREADS",
		static_cast<int>(hw_threads == 0 ? 1 : hw_threads));
	options.max_num_iterations =
		environment_int("CEP_MAX_NUM_ITERATIONS", 100);
	options.linear_solver_type = ceres::SPARSE_SCHUR;
	options.function_tolerance = environment_double(
		"CEP_FUNCTION_TOLERANCE",
		options.function_tolerance);
	options.gradient_tolerance = environment_double(
		"CEP_GRADIENT_TOLERANCE",
		options.gradient_tolerance);
	options.parameter_tolerance = environment_double(
		"CEP_PARAMETER_TOLERANCE",
		options.parameter_tolerance);
	options.initial_trust_region_radius = environment_double(
		"CEP_INITIAL_TRUST_REGION_RADIUS",
		options.initial_trust_region_radius);

	std::vector<DiagnosticParameterBlock> diagnostic_parameter_blocks;
	std::unique_ptr<StrictDiagnosticIterationCallback> strict_callback;
	const bool strict_vector_diagnostics =
		diagnostics && environment_flag("CEP_STRICT_VECTOR_DIAGNOSTICS", true);
	if (strict_vector_diagnostics) {
		diagnostic_parameter_blocks = collect_active_parameter_blocks(problem);
		strict_callback = std::make_unique<StrictDiagnosticIterationCallback>(
			diagnostic_parameter_blocks);
		options.update_state_every_iteration = true;
		options.callbacks.push_back(strict_callback.get());
	}

	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);

	const double initial_rmse = sqrt(2.0 * summary.initial_cost / nobs);
	const double final_rmse = sqrt(2.0 * summary.final_cost / nobs);
	const int iteration_count = static_cast<int>(summary.iterations.size() > 0 ? summary.iterations.size() - 1 : 0);
	int accepted_steps = 0;
	int rejected_steps = 0;
	int total_linear_solver_iterations = 0;
	for (const auto& iteration : summary.iterations) {
		total_linear_solver_iterations += iteration.linear_solver_iterations;
		if (iteration.iteration == 0) {
			continue;
		}
		if (iteration.step_is_successful) {
			++accepted_steps;
		} else {
			++rejected_steps;
		}
	}
	double final_gradient_max_norm = 0.0;
	double final_gradient_norm = 0.0;
	if (!summary.iterations.empty()) {
		final_gradient_max_norm = summary.iterations.back().gradient_max_norm;
		final_gradient_norm = summary.iterations.back().gradient_norm;
	}
	StrictDiagnosticMetrics strict_metrics;
	if (diagnostics && m_szReport != nullptr) {
		const std::string parent = diagnostic_parent_directory(m_szReport);
		const std::vector<DiagnosticIterationState> empty_states;
		strict_metrics = write_strict_diagnostics(
			problem,
			diagnostic_parameter_blocks,
			strict_callback ? strict_callback->states() : empty_states,
			summary,
			nobs,
			options.gradient_tolerance,
			parent);
	}

	m_last_metrics.success = summary.IsSolutionUsable();
	m_last_metrics.cameras = m_ncams;
	m_last_metrics.points = m_n3Dpts;
	m_last_metrics.observations = nobs;
	m_last_metrics.iterations = iteration_count;
	m_last_metrics.accepted_steps = accepted_steps;
	m_last_metrics.rejected_steps = rejected_steps;
	m_last_metrics.linear_solver_iterations = total_linear_solver_iterations;
	m_last_metrics.termination_type = static_cast<int>(summary.termination_type);
	m_last_metrics.initial_cost = summary.initial_cost;
	m_last_metrics.final_cost = summary.final_cost;
	m_last_metrics.initial_rmse_px = initial_rmse;
	m_last_metrics.final_rmse_px = final_rmse;
	m_last_metrics.initial_gradient_max_norm =
		strict_metrics.initial_gradient_max_norm;
	m_last_metrics.final_gradient_max_norm = final_gradient_max_norm;
	m_last_metrics.final_gradient_norm = final_gradient_norm;
	m_last_metrics.gradient_reduction_ratio_final =
		strict_metrics.final_gradient_reduction_ratio;
	m_last_metrics.reached_gradient_tolerance =
		strict_metrics.reached_gradient_tolerance;
	m_last_metrics.iterations_to_gradient_tolerance =
		strict_metrics.iterations_to_gradient_tolerance;
	m_last_metrics.final_relative_function_decrease =
		strict_metrics.final_relative_function_decrease;
	m_last_metrics.final_relative_step_size =
		strict_metrics.final_relative_step_size;
	m_last_metrics.final_lm_gain_ratio = strict_metrics.final_lm_gain_ratio;
	m_last_metrics.final_gradient_lipschitz_estimate =
		strict_metrics.final_gradient_lipschitz_estimate;
	m_last_metrics.final_direction_quality =
		strict_metrics.final_direction_quality;
	m_last_metrics.solver_time_sec = summary.total_time_in_seconds;
	m_last_metrics.linear_solver_time_sec = summary.linear_solver_time_in_seconds;

	if (diagnostics && point_condition_sample > 0 && m_szReport != nullptr) {
		const std::string report_path(m_szReport);
		const size_t separator = report_path.find_last_of("/\\");
		const std::string parent =
			separator == std::string::npos ? "." : report_path.substr(0, separator);
		const std::string conditioning_path =
			parent + "/point_block_conditioning.csv";
		FILE* fp = nullptr;
		fopen_s(&fp, conditioning_path.c_str(), "w");
		if (fp != nullptr) {
			fprintf(fp,
				"point_index,track_observations,jacobian_rows,jacobian_cols,"
				"eigen_min,eigen_mid,eigen_max,condition_number,numerical_rank\n");
			const int point_count = static_cast<int>(points.size());
			const int sample_count = std::min(point_condition_sample, point_count);
			for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
				const int point_index = sample_count == 1
					? 0
					: static_cast<int>(
						(static_cast<long long>(sample_index) * (point_count - 1)) /
						(sample_count - 1));

				std::vector<double*> parameter_blocks;
				switch (optype) {
				case xyz:
					parameter_blocks.push_back(&points[point_index].xyz[0]);
					break;
				case xy_inverse_z:
					parameter_blocks.push_back(&points[point_index].xy_inverse_z[0]);
					break;
				case depth:
					parameter_blocks.push_back(&points[point_index].world_depth[0]);
					break;
				case inverse_depth:
					parameter_blocks.push_back(&points[point_index].world_inverse_depth[0]);
					break;
				case archored_xyz:
					parameter_blocks.push_back(&points[point_index].archored_xyz[0]);
					break;
				case archored_xy_inverse_z:
					parameter_blocks.push_back(&points[point_index].archored_xy_inverse_z[0]);
					break;
				case archored_depth:
					parameter_blocks.push_back(
						&points[point_index].archored_spherical_range[0]);
					break;
				case archored_inverse_depth:
					parameter_blocks.push_back(
						&points[point_index].archored_spherical_inverse_range[0]);
					break;
				case parallax:
					parameter_blocks.push_back(&points[point_index].parallax_world[0]);
					break;
				case anchor_camera_xyz:
					parameter_blocks.push_back(&points[point_index].anchor_camera_xyz[0]);
					break;
				case anchor_camera_xy_inverse_z:
					parameter_blocks.push_back(&points[point_index].anchor_camera_xy_inverse_z[0]);
					break;
				case anchor_camera_spherical_range:
					parameter_blocks.push_back(
						&points[point_index].anchor_camera_spherical_range[0]);
					break;
				case anchor_camera_spherical_inverse_range:
					parameter_blocks.push_back(
						&points[point_index].anchor_camera_spherical_inverse_range[0]);
					break;
				case parallax_camera:
					parameter_blocks.push_back(&points[point_index].parallax_camera[0]);
					break;
				default:
					break;
				}

				std::set<ceres::ResidualBlockId> residual_set;
				for (double* parameter_block : parameter_blocks) {
					std::vector<ceres::ResidualBlockId> residual_blocks;
					problem.GetResidualBlocksForParameterBlock(
						parameter_block, &residual_blocks);
					residual_set.insert(residual_blocks.begin(), residual_blocks.end());
				}
				if (parameter_blocks.empty() || residual_set.empty()) {
					continue;
				}

				ceres::Problem::EvaluateOptions evaluation_options;
				evaluation_options.apply_loss_function = false;
				evaluation_options.num_threads = 1;
				evaluation_options.parameter_blocks = parameter_blocks;
				evaluation_options.residual_blocks.assign(
					residual_set.begin(), residual_set.end());
				ceres::CRSMatrix jacobian;
				if (!problem.Evaluate(
						evaluation_options, nullptr, nullptr, nullptr, &jacobian) ||
					jacobian.num_cols != 3) {
					continue;
				}

				Eigen::MatrixXd dense_jacobian =
					Eigen::MatrixXd::Zero(jacobian.num_rows, jacobian.num_cols);
				for (int row = 0; row < jacobian.num_rows; ++row) {
					for (int entry = jacobian.rows[row];
						 entry < jacobian.rows[row + 1];
						 ++entry) {
						dense_jacobian(row, jacobian.cols[entry]) =
							jacobian.values[entry];
					}
				}
				const Eigen::Matrix3d point_hessian =
					dense_jacobian.transpose() * dense_jacobian;
				const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(
					point_hessian);
				if (eigensolver.info() != Eigen::Success) {
					continue;
				}
				const Eigen::Vector3d eigenvalues = eigensolver.eigenvalues();
				const double eigen_max = eigenvalues[2];
				const double rank_threshold =
					std::max(1e-18, std::abs(eigen_max) * 1e-12);
				int numerical_rank = 0;
				for (int eigen_index = 0; eigen_index < 3; ++eigen_index) {
					if (eigenvalues[eigen_index] > rank_threshold) {
						++numerical_rank;
					}
				}
				const double condition_number =
					eigenvalues[0] > rank_threshold
						? eigen_max / eigenvalues[0]
						: std::numeric_limits<double>::infinity();
				fprintf(fp,
					"%d,%d,%d,%d,%.15lf,%.15lf,%.15lf,%.15lf,%d\n",
					point_index,
					tracks[point_index].nview,
					jacobian.num_rows,
					jacobian.num_cols,
					eigenvalues[0],
					eigenvalues[1],
					eigenvalues[2],
					condition_number,
					numerical_rank);
			}
			fclose(fp);
		}
	}

	if (diagnostics && schur_sample > 0 && m_szReport != nullptr) {
		const std::string report_path(m_szReport);
		const size_t separator = report_path.find_last_of("/\\");
		const std::string parent =
			separator == std::string::npos ? "." : report_path.substr(0, separator);
		const int point_count = static_cast<int>(points.size());
		const int sample_count = std::min(schur_sample, point_count);

		std::vector<double*> sampled_point_blocks;
		std::set<double*> sampled_point_block_set;
		std::set<ceres::ResidualBlockId> residual_set;
		for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
			const int point_index = sample_count == 1
				? 0
				: static_cast<int>(
					(static_cast<long long>(sample_index) * (point_count - 1)) /
					(sample_count - 1));
			std::vector<double*> point_blocks;
			switch (optype) {
			case xyz:
				point_blocks.push_back(&points[point_index].xyz[0]);
				break;
			case xy_inverse_z:
				point_blocks.push_back(&points[point_index].xy_inverse_z[0]);
				break;
			case depth:
				point_blocks.push_back(&points[point_index].world_depth[0]);
				break;
			case inverse_depth:
				point_blocks.push_back(&points[point_index].world_inverse_depth[0]);
				break;
			case archored_xyz:
				point_blocks.push_back(&points[point_index].archored_xyz[0]);
				break;
			case archored_xy_inverse_z:
				point_blocks.push_back(&points[point_index].archored_xy_inverse_z[0]);
				break;
			case archored_depth:
				point_blocks.push_back(
					&points[point_index].archored_spherical_range[0]);
				break;
			case archored_inverse_depth:
				point_blocks.push_back(
					&points[point_index].archored_spherical_inverse_range[0]);
				break;
			case parallax:
				point_blocks.push_back(&points[point_index].parallax_world[0]);
				break;
			case anchor_camera_xyz:
				point_blocks.push_back(&points[point_index].anchor_camera_xyz[0]);
				break;
			case anchor_camera_xy_inverse_z:
				point_blocks.push_back(&points[point_index].anchor_camera_xy_inverse_z[0]);
				break;
			case anchor_camera_spherical_range:
				point_blocks.push_back(
					&points[point_index].anchor_camera_spherical_range[0]);
				break;
			case anchor_camera_spherical_inverse_range:
				point_blocks.push_back(
					&points[point_index].anchor_camera_spherical_inverse_range[0]);
				break;
			case parallax_camera:
				point_blocks.push_back(&points[point_index].parallax_camera[0]);
				break;
			default:
				break;
			}
			for (double* point_block : point_blocks) {
				if (sampled_point_block_set.insert(point_block).second) {
					sampled_point_blocks.push_back(point_block);
				}
			std::vector<ceres::ResidualBlockId> point_residual_blocks;
			problem.GetResidualBlocksForParameterBlock(
				point_block, &point_residual_blocks);
			residual_set.insert(
				point_residual_blocks.begin(), point_residual_blocks.end());
			}
		}

		std::vector<double*> camera_blocks;
		std::set<double*> camera_block_set;
		for (ceres::ResidualBlockId residual_block : residual_set) {
			std::vector<double*> residual_parameter_blocks;
			problem.GetParameterBlocksForResidualBlock(
				residual_block, &residual_parameter_blocks);
			for (double* parameter_block : residual_parameter_blocks) {
				if (problem.IsParameterBlockConstant(parameter_block) ||
					problem.ParameterBlockTangentSize(parameter_block) <= 0) {
					continue;
				}
				if (sampled_point_block_set.count(parameter_block) == 0 &&
					camera_block_set.insert(parameter_block).second) {
					camera_blocks.push_back(parameter_block);
				}
			}
		}

		int camera_dimension = 0;
		for (double* camera_block : camera_blocks) {
			camera_dimension += problem.ParameterBlockTangentSize(camera_block);
		}
		int point_dimension = 0;
		for (double* point_block : sampled_point_blocks) {
			if (problem.IsParameterBlockConstant(point_block) ||
				problem.ParameterBlockTangentSize(point_block) <= 0) {
				continue;
			}
			point_dimension += problem.ParameterBlockTangentSize(point_block);
		}

		if (!camera_blocks.empty() && !sampled_point_blocks.empty() &&
			!residual_set.empty() && camera_dimension > 0 && point_dimension > 0) {
			ceres::Problem::EvaluateOptions evaluation_options;
			evaluation_options.apply_loss_function = false;
			evaluation_options.num_threads = 1;
			evaluation_options.parameter_blocks = camera_blocks;
			evaluation_options.parameter_blocks.insert(
				evaluation_options.parameter_blocks.end(),
				sampled_point_blocks.begin(),
				sampled_point_blocks.end());
			evaluation_options.residual_blocks.assign(
				residual_set.begin(), residual_set.end());

			ceres::CRSMatrix jacobian;
			if (problem.Evaluate(
					evaluation_options, nullptr, nullptr, nullptr, &jacobian) &&
				jacobian.num_cols == camera_dimension + point_dimension) {
				Eigen::MatrixXd dense_jacobian =
					Eigen::MatrixXd::Zero(jacobian.num_rows, jacobian.num_cols);
				for (int row = 0; row < jacobian.num_rows; ++row) {
					for (int entry = jacobian.rows[row];
						 entry < jacobian.rows[row + 1];
						 ++entry) {
						dense_jacobian(row, jacobian.cols[entry]) =
							jacobian.values[entry];
					}
				}
				const Eigen::MatrixXd normal_matrix =
					dense_jacobian.transpose() * dense_jacobian;
				const Eigen::MatrixXd camera_hessian =
					normal_matrix.topLeftCorner(camera_dimension, camera_dimension);
				const Eigen::MatrixXd cross_hessian =
					normal_matrix.topRightCorner(camera_dimension, point_dimension);
				const Eigen::MatrixXd point_hessian =
					normal_matrix.bottomRightCorner(point_dimension, point_dimension);
				const Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd>
					point_solver(point_hessian);
				Eigen::MatrixXd reduced_camera =
					camera_hessian -
					cross_hessian * point_solver.solve(cross_hessian.transpose());
				reduced_camera =
					0.5 * (reduced_camera + reduced_camera.transpose());

				const Eigen::JacobiSVD<Eigen::MatrixXd> svd(reduced_camera);
				const Eigen::VectorXd singular_values = svd.singularValues();
				if (singular_values.size() > 0) {
					const double sigma_max = singular_values[0];
					constexpr double rank_relative_tolerance = 1e-8;
					const double rank_threshold =
						std::max(
							1e-18,
							std::abs(sigma_max) * rank_relative_tolerance);
					int numerical_rank = 0;
					double sigma_min_nonzero = 0.0;
					for (int singular_index = 0;
						 singular_index < singular_values.size();
						 ++singular_index) {
						if (singular_values[singular_index] > rank_threshold) {
							++numerical_rank;
							sigma_min_nonzero = singular_values[singular_index];
						}
					}
					const int nullity = camera_dimension - numerical_rank;
					const double condition_number =
						sigma_min_nonzero > 0.0
							? sigma_max / sigma_min_nonzero
							: std::numeric_limits<double>::infinity();

					const std::string summary_path = parent + "/schur_summary.csv";
					FILE* fp = nullptr;
					fopen_s(&fp, summary_path.c_str(), "w");
					if (fp != nullptr) {
						fprintf(fp,
							"sampled_points,residual_blocks,camera_parameter_blocks,"
							"point_parameter_blocks,camera_dimension,point_dimension,"
							"numerical_rank,nullity,sigma_max,sigma_min_nonzero,"
							"condition_number,rank_relative_tolerance,rank_threshold\n");
						fprintf(fp,
							"%d,%d,%d,%d,%d,%d,%d,%d,%.15lf,%.15lf,%.15lf,"
							"%.15lf,%.15lf\n",
							sample_count,
							static_cast<int>(residual_set.size()),
							static_cast<int>(camera_blocks.size()),
							static_cast<int>(sampled_point_blocks.size()),
							camera_dimension,
							point_dimension,
							numerical_rank,
							nullity,
							sigma_max,
							sigma_min_nonzero,
							condition_number,
							rank_relative_tolerance,
							rank_threshold);
						fclose(fp);
					}

					const std::string spectrum_path = parent + "/schur_spectrum.csv";
					fopen_s(&fp, spectrum_path.c_str(), "w");
					if (fp != nullptr) {
						fprintf(fp, "index,singular_value,normalized_singular_value\n");
						for (int singular_index = 0;
							 singular_index < singular_values.size();
							 ++singular_index) {
							const double normalized =
								sigma_max > 0.0
									? singular_values[singular_index] / sigma_max
									: 0.0;
							fprintf(fp,
								"%d,%.15lf,%.15lf\n",
								singular_index,
								singular_values[singular_index],
								normalized);
						}
						fclose(fp);
					}
				}
			}
		}
	}

	if (diagnostics) {
		std::cout << summary.BriefReport() << "\n";
		printf("%s %.12lf %s %.12lf\n", "RMSE(px):", initial_rmse, "to", final_rmse);
	}

	if (diagnostics && m_szCamePose != nullptr) {
		FILE* fp = nullptr;
		fopen_s(&fp, m_szCamePose, "w");
		if (fp != nullptr) {
			for (int i = 0; i < static_cast<int>(cams.size()); i++) {
				fprintf(fp, "%.15lf %.15lf %.15lf %.15lf %.15lf %.15lf %d\n",
					cams[i].euler_angle[0], cams[i].euler_angle[1], cams[i].euler_angle[2],
					cams[i].camera_center[0], cams[i].camera_center[1], cams[i].camera_center[2],
					cams[i].camidx);
			}
			fclose(fp);
		}
	}

	if (diagnostics && m_sz3Dpts != nullptr) {
		if (optype == xy_inverse_z) {
			xy_inverse_z2xyz();
		} else if (optype == depth) {
			depth2xyz();
		} else if (optype == inverse_depth) {
			inverse_depth2xyz();
		} else if (optype == archored_xyz) {
			archored_xyz2xyz();
		} else if (optype == archored_xy_inverse_z) {
			archored_xy_inverse_z2xyz();
		} else if (optype == archored_depth) {
			archored_depth2xyz();
		} else if (optype == archored_inverse_depth) {
			archored_inverse_depth2xyz();
		} else if (optype == parallax) {
			parallax2xyz();
		} else if (optype == anchor_camera_xyz) {
			anchor_camera_xyz2xyz();
		} else if (optype == anchor_camera_xy_inverse_z) {
			anchor_camera_xy_inverse_z2xyz();
		} else if (optype == anchor_camera_spherical_range) {
			anchor_camera_spherical_range2xyz();
		} else if (optype == anchor_camera_spherical_inverse_range) {
			anchor_camera_spherical_inverse_range2xyz();
		} else if (optype == parallax_camera) {
			parallax_camera2xyz();
		}

		FILE* fp = nullptr;
		fopen_s(&fp, m_sz3Dpts, "w");
		if (fp != nullptr) {
			fprintf(fp, "%s\n", "ply");
			fprintf(fp, "%s\n", "format ascii 1.0");
			fprintf(fp, "%s %d\n", "element vertex", m_n3Dpts);
			fprintf(fp, "%s\n", "property float x");
			fprintf(fp, "%s\n", "property float y");
			fprintf(fp, "%s\n", "property float z");
			fprintf(fp, "%s\n", "end_header");
			for (int i = 0; i < static_cast<int>(points.size()); i++) {
				fprintf(fp, "%.15lf %.15lf %.15lf\n", points[i].xyz[0], points[i].xyz[1], points[i].xyz[2]);
			}
			fclose(fp);
		}
	}

	if (diagnostics && m_szReport != nullptr) {
		std::string report_path(m_szReport);
		FILE* fp = nullptr;
		fopen_s(&fp, report_path.c_str(), "w");
		if (fp != nullptr) {
			fprintf(fp, "method %s\n", method_name);
			fprintf(fp, "cameras %d\n", m_ncams);
			fprintf(fp, "points %d\n", m_n3Dpts);
			fprintf(fp, "observations %d\n", nobs);
			fprintf(fp, "initial_cost %.15lf\n", summary.initial_cost);
			fprintf(fp, "final_cost %.15lf\n", summary.final_cost);
			fprintf(fp, "initial_rmse_px %.15lf\n", initial_rmse);
			fprintf(fp, "final_rmse_px %.15lf\n", final_rmse);
			fprintf(fp, "iterations %d\n", iteration_count);
			fprintf(fp, "accepted_steps %d\n", accepted_steps);
			fprintf(fp, "rejected_steps %d\n", rejected_steps);
			fprintf(fp, "initial_gradient_max_norm %.15lf\n",
				strict_metrics.initial_gradient_max_norm);
			fprintf(fp, "final_gradient_max_norm %.15lf\n", final_gradient_max_norm);
			fprintf(fp, "final_gradient_norm %.15lf\n", final_gradient_norm);
			fprintf(fp, "gradient_reduction_ratio_final %.15lf\n",
				strict_metrics.final_gradient_reduction_ratio);
			fprintf(fp, "reached_gradient_tolerance %d\n",
				strict_metrics.reached_gradient_tolerance);
			fprintf(fp, "iterations_to_gradient_tolerance %d\n",
				strict_metrics.iterations_to_gradient_tolerance);
			fprintf(fp, "final_relative_function_decrease %.15lf\n",
				strict_metrics.final_relative_function_decrease);
			fprintf(fp, "final_relative_step_size %.15lf\n",
				strict_metrics.final_relative_step_size);
			fprintf(fp, "final_lm_gain_ratio %.15lf\n",
				strict_metrics.final_lm_gain_ratio);
			fprintf(fp, "final_gradient_lipschitz_estimate %.15lf\n",
				strict_metrics.final_gradient_lipschitz_estimate);
			fprintf(fp, "final_direction_quality %.15lf\n",
				strict_metrics.final_direction_quality);
			fprintf(fp, "linear_solver_iterations %d\n", total_linear_solver_iterations);
			fprintf(fp, "solver_time_sec %.15lf\n", summary.total_time_in_seconds);
			fprintf(fp, "linear_solver_time_sec %.15lf\n", summary.linear_solver_time_in_seconds);
			fprintf(fp, "termination_type %d\n", static_cast<int>(summary.termination_type));
			fprintf(fp, "brief_report %s\n", summary.BriefReport().c_str());
			fclose(fp);
		}

		const size_t pos = report_path.find_last_of("/\\");
		const std::string parent = (pos == std::string::npos) ? "." : report_path.substr(0, pos);
		const std::string metrics_path = parent + "/metrics.json";
		fopen_s(&fp, metrics_path.c_str(), "w");
		if (fp != nullptr) {
			fprintf(fp, "{\n");
			fprintf(fp, "  \"method\": \"%s\",\n", method_name);
			fprintf(fp, "  \"mode\": \"diagnostic\",\n");
			fprintf(fp, "  \"ceres_version\": \"%s\",\n", CERES_VERSION_STRING);
			fprintf(fp, "  \"solver\": {\n");
			fprintf(fp, "    \"trust_region_strategy\": \"LEVENBERG_MARQUARDT\",\n");
			fprintf(fp, "    \"linear_solver\": \"SPARSE_SCHUR\",\n");
			fprintf(fp, "    \"robust_loss\": \"none\",\n");
			fprintf(fp, "    \"max_num_iterations\": %d,\n", options.max_num_iterations);
			fprintf(fp, "    \"num_threads\": %d,\n", options.num_threads);
			fprintf(fp, "    \"function_tolerance\": %.17g,\n", options.function_tolerance);
			fprintf(fp, "    \"gradient_tolerance\": %.17g,\n", options.gradient_tolerance);
			fprintf(fp, "    \"parameter_tolerance\": %.17g,\n", options.parameter_tolerance);
			fprintf(fp, "    \"min_relative_decrease\": %.17g,\n", options.min_relative_decrease);
			fprintf(fp, "    \"initial_trust_region_radius\": %.17g,\n",
				options.initial_trust_region_radius);
			fprintf(fp, "    \"jacobi_scaling\": %s\n",
				options.jacobi_scaling ? "true" : "false");
			fprintf(fp, "  },\n");
			fprintf(fp, "  \"problem\": {\n");
			fprintf(fp, "    \"cameras\": %d,\n", m_ncams);
			fprintf(fp, "    \"points\": %d,\n", m_n3Dpts);
			fprintf(fp, "    \"observations\": %d\n", nobs);
			fprintf(fp, "  },\n");
			fprintf(fp, "  \"result\": {\n");
			fprintf(fp, "    \"success\": %s,\n",
				m_last_metrics.success ? "true" : "false");
			fprintf(fp, "    \"initial_cost\": %.17g,\n", summary.initial_cost);
			fprintf(fp, "    \"final_cost\": %.17g,\n", summary.final_cost);
			fprintf(fp, "    \"initial_rmse_px\": %.17g,\n", initial_rmse);
			fprintf(fp, "    \"final_rmse_px\": %.17g,\n", final_rmse);
			fprintf(fp, "    \"iterations\": %d,\n", iteration_count);
			fprintf(fp, "    \"accepted_steps\": %d,\n", accepted_steps);
			fprintf(fp, "    \"rejected_steps\": %d,\n", rejected_steps);
			fprintf(fp, "    \"solver_time_sec\": %.17g,\n",
				summary.total_time_in_seconds);
			fprintf(fp, "    \"linear_solver_time_sec\": %.17g,\n",
				summary.linear_solver_time_in_seconds);
			fprintf(fp, "    \"initial_gradient_max_norm\": %.17g,\n",
				strict_metrics.initial_gradient_max_norm);
			fprintf(fp, "    \"final_gradient_max_norm\": %.17g,\n",
				final_gradient_max_norm);
			fprintf(fp, "    \"final_gradient_norm\": %.17g,\n",
				final_gradient_norm);
			fprintf(fp, "    \"gradient_reduction_ratio_final\": %.17g,\n",
				strict_metrics.final_gradient_reduction_ratio);
			fprintf(fp, "    \"reached_gradient_tolerance\": %s,\n",
				strict_metrics.reached_gradient_tolerance ? "true" : "false");
			fprintf(fp, "    \"iterations_to_gradient_tolerance\": %d,\n",
				strict_metrics.iterations_to_gradient_tolerance);
			fprintf(fp, "    \"final_relative_function_decrease\": %.17g,\n",
				strict_metrics.final_relative_function_decrease);
			fprintf(fp, "    \"final_relative_step_size\": %.17g,\n",
				strict_metrics.final_relative_step_size);
			fprintf(fp, "    \"final_lm_gain_ratio\": %.17g,\n",
				strict_metrics.final_lm_gain_ratio);
			fprintf(fp, "    \"final_gradient_lipschitz_estimate\": %.17g,\n",
				strict_metrics.final_gradient_lipschitz_estimate);
			fprintf(fp, "    \"final_direction_quality\": %.17g,\n",
				strict_metrics.final_direction_quality);
			fprintf(fp, "    \"termination_type\": %d\n",
				static_cast<int>(summary.termination_type));
			fprintf(fp, "  },\n");
			fprintf(fp, "  \"diagnostics\": {\n");
			fprintf(fp, "    \"strict_vector_diagnostics\": %s,\n",
				strict_metrics.vector_metrics_available ? "true" : "false");
			fprintf(fp, "    \"point_condition_sample\": %d,\n",
				std::max(0, point_condition_sample));
			fprintf(fp, "    \"schur_sample\": %d\n", std::max(0, schur_sample));
			fprintf(fp, "  }\n");
			fprintf(fp, "}\n");
			fclose(fp);
		}
	}

	return m_last_metrics.success;
}

bool PBA::ba_initialize( char* szCamera, char* szFeature,  char* szCalib, char* szXYZ )
{
	// printf("BA: Bundle Adjustment Version 1.0\n");
	FILE* fp = nullptr;

	free(m_archor); m_archor = nullptr;
	free(m_photo); m_photo = nullptr;
	free(m_feature); m_feature = nullptr;
	free(m_motstruct); m_motstruct = nullptr;
	free(m_imgpts); m_imgpts = nullptr;
	free(m_XYZ); m_XYZ = nullptr;
	free(m_K); m_K = nullptr;
	free(m_V); m_V = nullptr;
	free(m_vmask); m_vmask = nullptr;
	free(m_umask); m_umask = nullptr;
	free(m_smask); m_smask = nullptr;
	free(m_imgptsSum); m_imgptsSum = nullptr;
	free(m_struct); m_struct = nullptr;
	free(m_pnt2main); m_pnt2main = nullptr;
	free(m_archorSort); m_archorSort = nullptr;
	free(m_KR); m_KR = nullptr;
	free(m_KdA); m_KdA = nullptr;
	free(m_KdB); m_KdB = nullptr;
	free(m_KdG); m_KdG = nullptr;
	free(m_P); m_P = nullptr;
	points.clear();
	cams.clear();
	tracks.clear();
	intrs.clear();
	m_bProvideXYZ = false;
	m_bFocal = false;

	//must input initial initial camera pose file and projection image points file
	fopen_s(&fp, szCamera, "r" );
	if ( fp == NULL )
	{
		fprintf( stderr, "BA: Missing initial camera poses file! \n");
		exit(1);
	}
	else
		fclose(fp);

	fopen_s(&fp, szFeature, "r" );
	if ( fp == NULL )
	{	
		fprintf( stderr, "BA: Missing feature projection points file! \n");
		exit(1); 
	}
	else
		fclose(fp);

	if (szCalib != NULL)
	{
		FILE* fpc = nullptr;
		fopen_s(&fpc, szCalib, "r");
		nc_ = findNcameras(fpc) / 3;
		fclose(fpc);
		fpc = nullptr;
		// printf("%d\n", nc_);
		m_bFocal = false;
		m_K = (double*)malloc(9 * nc_ * sizeof(double));
		ba_readCameraPoseration(szCalib, m_K);
	}

	if ( szXYZ != NULL )
		m_bProvideXYZ = true;	
	//read camera pose & features images projs, and initialize features points( three kinds of angle )
	pba_readAndInitialize( szCamera, szFeature,szCalib, &m_ncams, &m_n3Dpts, &m_n2Dprojs,&m_motstruct,//number of camera, 3D points, 2D projection points,6 camera pose and 3 feature parameters
		&m_imgpts, &m_archor, &m_vmask, &m_umask, &m_photo, &m_feature, &m_archorSort);

	return true;
}


void PBA::pba_readProjectionAndInitilizeFeature(FILE *fp,
	double *params, double *projs, char *vmask, int ncams, 
	int *archor,char* umask,int* nphoto, int* nfeature, int* archorSort )
{
	int n;
	int nframes;
	int ptno = 0, cur;

	int nproj2D = 0;
	
	int count = 0;

	int frameno;
	int feastart = 0;

	double* ptr1 = projs;

	int i;
	int  sum, cnp = 6;

	int nFlag;
	
	int *ptr2;
	bool bAdjust;	

	bool bM, bN;
	//read all projection point, initialize three feature angle at the same time
	while(!feof(fp))
	{
		nFlag = 0;
		n = readNInts( fp, &nframes, 1 );  
		if( n!=1 )
			break;

		Track track;
		track.nview = nframes;
		track.obss.reserve(nframes);

		archor[ptno*3] = nframes;
		cur = 0;
		bM = bN = false;
		for( i=0, sum = 0; i<nframes; ++i )
		{
			n = readNInts( fp, &frameno, 1 );
			nphoto[nproj2D] = frameno;
			nfeature[nproj2D] = ptno;
			nproj2D++;

			if(frameno>=ncams)
			{
				fprintf(stderr, "ParallaxBA: the image No. of projection point is out of max image No.\n");
				return;
			}

			n += readNDoubles( fp, ptr1, 2 ); 

			Observation obs;
			obs.view_idx = frameno;
			obs.u = ptr1[0];
			obs.v = ptr1[1];
			track.obss.push_back(obs);

			ptr1+=2;
			if(n!=3)
			{
				fprintf(stderr, "ParallaxBA:reading image projections wrong!\n");
				return;
			}

			if ( bM && bN )
			{
				ptr2 = archorSort+ptno*2;
				//bAdjust = pba_initializeOtheArchors_Mindw( //_Mindw
				bAdjust = pba_initializeOtheArchors( //Maxdw
					projs+feastart*2,
					nphoto+feastart,
					m_motstruct,
					m_K,
					m_motstruct + m_ncams*cnp + ptno * 3, 
					ptr2,
					sum,
					i,
					ptno );
				if ( bAdjust )
				{
					archor[ptno*3+1] = *(nphoto+feastart+ptr2[0]);
					archor[ptno*3+2] = *(nphoto+feastart+ptr2[1]);
				}
				sum++;
			}

			if ( bM && !bN )
			{	
				bool bT = pba_initializeAssoArchor( 
					projs+feastart*2,	
					nphoto+feastart,	
					m_motstruct,
					m_K,
					m_motstruct+m_ncams*cnp+ptno*3,
					0,
					1,
					ptno );

				if (bT)
				{
					archorSort[ptno*2+1] = i;
					archor[ptno*3+2] = nphoto[count];
					sum++;

					bN = true;
				}
			}

			if ( !bM )
			{
				bool bT = pba_initializeMainArchor( 
					projs+feastart*2,	
					m_motstruct,		
					m_K,				
					m_motstruct+m_ncams*cnp+ptno*3,
					nphoto[count],		
					ptno,				
					m_KR );				

				archorSort[ptno*2] = i;
				archor[ptno*3+1] = nphoto[count];
				sum++;
				bM = true;
			}	
			count++;	
		}

		tracks.push_back(track);
		feastart += nframes;
		ptno++;
	}
}

void PBA::pba_readAndInitialize(char *camsfname, char *ptsfname,char *calibfname, int *ncams,
	int *n3Dpts, int *n2Dprojs,
	double **motstruct, double **imgpts,
	int **archor, char **vmask,
	char **umask, int **nphoto,
	int** nfeature, int** archorSort)
{
	FILE *fpc = nullptr, *fpp = nullptr, *fpXYZ = nullptr;
	int i, tmp1, tmp2;
	double ptMain[3], ptA[3];
	double dW1, dW2;	

	//calculate number of cameras, 3D points and projection points
	fopen_s(&fpc, camsfname, "r" );
	*ncams	=	findNcameras( fpc );
	m_ncams =	*ncams;
	cams.reserve(m_ncams);
	m_V = (int*)malloc(sizeof(int) * m_ncams);

	fopen_s(&fpp, ptsfname, "r" );
	readNpointsAndNprojections( fpp, n3Dpts, 3, n2Dprojs, 2 );
	points.reserve(*n3Dpts);
	tracks.reserve(*n3Dpts);

	*motstruct = (double*)malloc((*ncams * 6 + *n3Dpts * 3) * sizeof(double));
	if(	*motstruct==NULL )
	{
		fprintf(stderr, "ParallaxBA error: Memory allocation for 'motstruct' failed \n");
		exit(1);
	}

	*imgpts = (double*)malloc(*n2Dprojs * 2 * sizeof(double));
	if(	*imgpts==NULL )
	{
		fprintf(stderr, "ParallaxBA error: Memory allocation for 'imgpts' failed\n");
		exit(1);
	}

	rewind(fpc);
	rewind(fpp);

	*vmask = nullptr;
	*umask = nullptr;

	//allocate main and associate anchors
	*archor = (int*)malloc(*n3Dpts*3*sizeof(int));//
	memset(*archor, -1, *n3Dpts * 3 * sizeof(int)); 

	*nphoto		= (int*)malloc(*n2Dprojs*sizeof(int));//
	*nfeature	= (int*)malloc(*n2Dprojs*sizeof(int));//
	*archorSort = (int*)malloc(*n3Dpts*2*sizeof(int));
	memset(*archorSort, -1, *n3Dpts * 2 * sizeof(int));

	ba_readCameraPose(fpc, *motstruct, m_V);


	fclose(fpc);
	fpc = NULL;
	
	//Update KR
	m_KR  = (double*)malloc(m_ncams*9*sizeof(double));
	
	ba_updateKR(m_KR, nullptr, nullptr, nullptr, m_K, *motstruct);

	//if XYZ are provided, we can use them as feature initialization.
	// fprintf(stdout, "%s\n", m_bProvideXYZ ? "true" : "false");  
	if (m_bProvideXYZ)
	{
		// printf("%s\n",m_szXYZ);
		fopen_s(&fpXYZ, m_szXYZ, "r");
		m_XYZ = (double*)malloc(m_n3Dpts*3*sizeof(double));

		for( i = 0; i < m_n3Dpts; i++){
			if (fscanf_s(fpXYZ, "%lf  %lf  %lf", m_XYZ + i * 3, m_XYZ + i * 3 + 1, m_XYZ + i * 3 + 2) != 3) {
				fprintf(stderr, "BA error: Format of XYZ initialization file is wrong\n");
				exit(1);
			}

			Point3D p3d;
			p3d.xyz[0] = m_XYZ[i * 3];
			p3d.xyz[1] = m_XYZ[i * 3 + 1];
			p3d.xyz[2] = m_XYZ[i * 3 + 2];
			points.push_back(p3d);

		}
		// for (int i = 0; i < m_n3Dpts;i++){
		// 	printf("%f %f %f\n", points[i].xyz[0], points[i].xyz[1], points[i].xyz[2]);
		// }

		fclose(fpXYZ);
	}


	pba_readProjectionAndInitilizeFeature(fpp,
		*motstruct + *ncams * 6,
		*imgpts,
		*vmask,
		*ncams,
		*archor,
		*umask,
		*nphoto,
		*nfeature,
		*archorSort);

	fclose(fpp);

	
	double pti2k[3];
	int cur = 0;

	if (m_bProvideXYZ)
	{
		for (i = 0; i < m_n3Dpts; i++)
		{
			int nM = m_archor[i*3+1];
			int nN = m_archor[i*3+2];
			//printf("%d %d\n", nM, nN);


			ptMain[0] = *(*motstruct + nM*6 + 3);
			ptMain[1] = *(*motstruct + nM*6 + 4);
			ptMain[2] = *(*motstruct + nM*6 + 5);

			ptA[0] = *(*motstruct + nN*6 + 3);
			ptA[1] = *(*motstruct + nN*6 + 4);
			ptA[2] = *(*motstruct + nN*6 + 5);

			pti2k[0] = ptA[0] - ptMain[0];
			pti2k[1] = ptA[1] - ptMain[1];
			pti2k[2] = ptA[2] - ptMain[2];

			double dispti2k;
			dispti2k = sqrt(pti2k[0] * pti2k[0] + pti2k[1] * pti2k[1] + pti2k[2] * pti2k[2]);

			ptMain[0] = m_XYZ[i * 3 + 0] - ptMain[0];
			ptMain[1] = m_XYZ[i * 3 + 1] - ptMain[1];
			ptMain[2] = m_XYZ[i * 3 + 2] - ptMain[2]; 

			ptA[0] = m_XYZ[i * 3 + 0] - ptA[0];
			ptA[1] = m_XYZ[i * 3 + 1] - ptA[1];
			ptA[2] = m_XYZ[i * 3 + 2] - ptA[2];

			dW1 = ptMain[0] * ptMain[0] + ptMain[1] * ptMain[1] + ptMain[2] * ptMain[2];

			dW2 = ptA[0] * ptA[0] + ptA[1] * ptA[1] + ptA[2] * ptA[2];

			double* pKR = m_KR + nM * 9;
			double n[2], n2[2], ptXj[3];

			ptXj[0] = ptMain[0];	ptXj[1] = ptMain[1];	ptXj[2] = ptMain[2];

			n[0] = (pKR[0]*ptXj[0] + pKR[1]*ptXj[1] + pKR[2]*ptXj[2])/
				(pKR[6] * ptXj[0] + pKR[7] * ptXj[1] + pKR[8] * ptXj[2]);

			n[1] = (pKR[3]*ptXj[0] + pKR[4]*ptXj[1] + pKR[5]*ptXj[2])/
				(pKR[6] * ptXj[0] + pKR[7] * ptXj[1] + pKR[8] * ptXj[2]);

			pKR = m_KR + nN*9;

			ptXj[0] = ptA[0];	ptXj[1] = ptA[1];	ptXj[2] = ptA[2];
			n2[0] = (pKR[0]*ptXj[0] + pKR[1]*ptXj[1] + pKR[2]*ptXj[2])/
				(pKR[6] * ptXj[0] + pKR[7] * ptXj[1] + pKR[8] * ptXj[2]);

			n2[1] = (pKR[3]*ptXj[0] + pKR[4]*ptXj[1] + pKR[5]*ptXj[2])/
				(pKR[6] * ptXj[0] + pKR[7] * ptXj[1] + pKR[8] * ptXj[2]);

			//printf("%d %d %d\n", m_archorSort[i * 2], m_archorSort[i * 2 + 1], m_archor[i * 3]);
			
			int id1 = cur + m_archorSort[i * 2];
			int id2 = cur + m_archorSort[i * 2 + 1];
			//printf("%f %f\n", m_imgpts[id1 * 2], m_imgpts[id1 * 2 + 1]);
			//printf("%f %f\n", m_imgpts[id2 * 2], m_imgpts[id2 * 2 + 1]);
			double err1 = (m_imgpts[id1 * 2] - n[0]) * (m_imgpts[id1 * 2] - n[0]) + (m_imgpts[id1 * 2 + 1] - n[1]) * (m_imgpts[id1 * 2 + 1] - n[1]);;
			double err2 = (m_imgpts[id2 * 2] - n2[0]) * (m_imgpts[id2 * 2] - n2[0]) + (m_imgpts[id2 * 2 + 1] - n2[1]) * (m_imgpts[id2 * 2 + 1] - n2[1]);

			//printf("%f %f\n", err1, err2);
			cur += m_archor[i * 3];

			if ((dW1 > 900.0 * dW2) || (err1 > err2) && (m_archor[3 * i] == 2))
			{
				m_archor[i*3+1] = nN;
				m_archor[i*3+2] = nM;

				tmp1 = m_archorSort[i*2] ;
				tmp2 = m_archorSort[i*2+1] ;

				m_archorSort[i*2] = tmp2;
				m_archorSort[i*2+1] = tmp1;

				double dDAngle = atan2(ptA[0], ptA[2]);
				double dHAngle = atan2(ptA[1], sqrt(ptA[0] * ptA[0] + ptA[2] * ptA[2]));

				(*motstruct)[m_ncams * 6 + i * 3] = dDAngle;
				(*motstruct)[m_ncams * 6 + i * 3 + 1] = dHAngle;

				//double dwwDot = ptMain[0]*ptA[0] + ptMain[1]*ptA[1] + ptMain[2]*ptA[2];				
			}	
		}
	}
	pba_applyAnchorPolicyAblation(*motstruct + *ncams * 6);
	if (!m_bProvideXYZ) {
		points.resize(m_n3Dpts);
		for (i = 0; i < m_n3Dpts; ++i) {
			Point3D& point = points[i];
			point.nM = m_archor[i * 3 + 1];
			point.nA = m_archor[i * 3 + 2];
			point.parallax_world[0] = (*motstruct)[m_ncams * 6 + i * 3];
			point.parallax_world[1] = (*motstruct)[m_ncams * 6 + i * 3 + 1];
			point.parallax_world[2] = (*motstruct)[m_ncams * 6 + i * 3 + 2];

			const int nM = point.nM;
			const int nA = point.nA;
			double bearing[3] = {
				sin(point.parallax_world[0]) * cos(point.parallax_world[1]),
				sin(point.parallax_world[1]),
				cos(point.parallax_world[0]) * cos(point.parallax_world[1])
			};
			double range = 1.0;
			if (nM >= 0 && nM < m_ncams && nA >= 0 && nA < m_ncams && nA != nM) {
				double baseline[3] = {
					cams[nA].camera_center[0] - cams[nM].camera_center[0],
					cams[nA].camera_center[1] - cams[nM].camera_center[1],
					cams[nA].camera_center[2] - cams[nM].camera_center[2]
				};
				const double baseline_norm = sqrt(
					baseline[0] * baseline[0] +
					baseline[1] * baseline[1] +
					baseline[2] * baseline[2]);
				const double omega = point.parallax_world[2];
				const double sin_omega = sin(omega);
				if (baseline_norm > 1e-12 && std::abs(sin_omega) > 1e-12) {
					const double beta = acos(clamp_acos_arg(
						(bearing[0] * baseline[0] +
						 bearing[1] * baseline[1] +
						 bearing[2] * baseline[2]) / baseline_norm));
					const double candidate_range =
						baseline_norm * sin(beta + omega) / sin_omega;
					if (std::isfinite(candidate_range) && candidate_range > 1e-9) {
						range = candidate_range;
					}
				}
			}
			point.xyz[0] = cams[nM].camera_center[0] + range * bearing[0];
			point.xyz[1] = cams[nM].camera_center[1] + range * bearing[1];
			point.xyz[2] = cams[nM].camera_center[2] + range * bearing[2];
		}
	}
	double x,y,z;
	for (int i = 0; i < points.size();i++){
		x=points[i].xyz[0];
		y=points[i].xyz[1];
		z=points[i].xyz[2];
		
		points[i].world_depth[0]=atan2(x,z);
		points[i].world_depth[1]=atan2(y,sqrt(x*x+z*z));
		points[i].world_depth[2]=sqrt(x*x+y*y+z*z);

		points[i].world_inverse_depth[0]=points[i].world_depth[0];
		points[i].world_inverse_depth[1]=points[i].world_depth[1];
		points[i].world_inverse_depth[2]=1/points[i].world_depth[2];


		points[i].nM = m_archor[i * 3 + 1];
		points[i].nA = m_archor[i * 3 + 2];
		const double initial_parallax =
			(*motstruct)[m_ncams * 6 + i * 3 + 2];

		int nM = points[i].nM;
		double dx = points[i].xyz[0] - cams[nM].camera_center[0];
		double dy = points[i].xyz[1] - cams[nM].camera_center[1];
		double dz = points[i].xyz[2] - cams[nM].camera_center[2];
		double d = sqrt(dx * dx + dy * dy + dz * dz);
		const double world_azimuth = atan2(dx, dz);
		const double world_elevation =
			atan2(dy, sqrt(dx * dx + dz * dz));
		double safe_dz = std::abs(dz) < 1e-12
			? std::copysign(1e-12, dz == 0.0 ? 1.0 : dz)
			: dz;
		double safe_z = std::abs(points[i].xyz[2]) < 1e-12
			? std::copysign(1e-12, points[i].xyz[2] == 0.0 ? 1.0 : points[i].xyz[2])
			: points[i].xyz[2];
		points[i].parallax_world[0] = world_azimuth;
		points[i].parallax_world[1] = world_elevation;
		points[i].parallax_world[2] = initial_parallax;
		points[i].archored_spherical_range[0] = world_azimuth;
		points[i].archored_spherical_range[1] = world_elevation;
		points[i].archored_spherical_range[2] = d;
		points[i].archored_spherical_inverse_range[0] = world_azimuth;
		points[i].archored_spherical_inverse_range[1] = world_elevation;
		points[i].archored_spherical_inverse_range[2] = 1 / d;
		points[i].archored_xyz[0] = dx;
		points[i].archored_xyz[1] = dy;
		points[i].archored_xyz[2] = dz;

		points[i].archored_xy_inverse_z[0] = dx / safe_dz;
		points[i].archored_xy_inverse_z[1] = dy / safe_dz;
		points[i].archored_xy_inverse_z[2] = 1 / safe_dz;

		points[i].xy_inverse_z[0] = points[i].xyz[0] / safe_z;
		points[i].xy_inverse_z[1] = points[i].xyz[1] / safe_z;
		points[i].xy_inverse_z[2] = 1 / safe_z;

		double rotation_main[9];
		double point_world_relative[3] = {dx, dy, dz};
		double point_camera[3];
		cep::cost::EulerToWorldToCamera(cams[nM].euler_angle, rotation_main);
		cep::cost::MatVec(rotation_main, point_world_relative, point_camera);
		const double camera_range = sqrt(
			point_camera[0] * point_camera[0] +
			point_camera[1] * point_camera[1] +
			point_camera[2] * point_camera[2]);
		const double safe_camera_z = std::abs(point_camera[2]) < 1e-12
			? std::copysign(1e-12, point_camera[2] == 0.0 ? 1.0 : point_camera[2])
			: point_camera[2];
		const double camera_azimuth = atan2(point_camera[0], point_camera[2]);
		const double camera_elevation = atan2(
			point_camera[1],
			sqrt(point_camera[0] * point_camera[0] + point_camera[2] * point_camera[2]));

		points[i].anchor_camera_xyz[0] = point_camera[0];
		points[i].anchor_camera_xyz[1] = point_camera[1];
		points[i].anchor_camera_xyz[2] = point_camera[2];
		points[i].anchor_camera_xy_inverse_z[0] = point_camera[0] / safe_camera_z;
		points[i].anchor_camera_xy_inverse_z[1] = point_camera[1] / safe_camera_z;
		points[i].anchor_camera_xy_inverse_z[2] = 1 / safe_camera_z;
		points[i].anchor_camera_spherical_range[0] = camera_azimuth;
		points[i].anchor_camera_spherical_range[1] = camera_elevation;
		points[i].anchor_camera_spherical_range[2] = camera_range;
		points[i].anchor_camera_spherical_inverse_range[0] = camera_azimuth;
		points[i].anchor_camera_spherical_inverse_range[1] = camera_elevation;
		points[i].anchor_camera_spherical_inverse_range[2] = 1 / camera_range;
		points[i].parallax_camera[0] = camera_azimuth;
		points[i].parallax_camera[1] = camera_elevation;
		points[i].parallax_camera[2] = initial_parallax;
	}

	if (m_bProvideXYZ) {
		free(m_XYZ);
		m_XYZ = nullptr;
	}

	fpXYZ = NULL;
}

void PBA::pba_applyAnchorPolicyAblation(double* feature_params)
{
	if (!m_bProvideXYZ || feature_params == nullptr || m_XYZ == nullptr ||
		m_archor == nullptr || m_archorSort == nullptr || m_photo == nullptr ||
		m_imgpts == nullptr || m_motstruct == nullptr || m_KR == nullptr) {
		return;
	}

	const AnchorPolicy policy = anchor_policy_from_environment();
	if (policy == AnchorPolicy::kCurrent) {
		return;
	}

	struct Candidate {
		int main_local = -1;
		int assoc_local = -1;
		int main_camera = -1;
		int assoc_camera = -1;
		double omega = 0.0;
		double sin_omega = 0.0;
		double baseline = 0.0;
		double err_main = 0.0;
		double err_assoc = 0.0;
		double err_sum = 0.0;
	};

	auto camera_center = [this](int camera_id) -> const double* {
		return m_motstruct + camera_id * 6 + 3;
	};

	auto projection_error = [this](int camera_id, const double ray[3],
		int observation_id, double* error) -> bool {
		const double* pKR = m_KR + camera_id * 9;
		const double denom =
			pKR[6] * ray[0] + pKR[7] * ray[1] + pKR[8] * ray[2];
		if (std::abs(denom) < 1e-12) {
			return false;
		}
		const double u =
			(pKR[0] * ray[0] + pKR[1] * ray[1] + pKR[2] * ray[2]) / denom;
		const double v =
			(pKR[3] * ray[0] + pKR[4] * ray[1] + pKR[5] * ray[2]) / denom;
		if (!std::isfinite(u) || !std::isfinite(v)) {
			return false;
		}
		const double du = m_imgpts[observation_id * 2] - u;
		const double dv = m_imgpts[observation_id * 2 + 1] - v;
		*error = du * du + dv * dv;
		return std::isfinite(*error);
	};

	auto apply_selected_pair = [this, feature_params, &camera_center](
		int point_id, int nviews, const double* xyz, Candidate best,
		bool apply_refinement) {
		const double* center_main = camera_center(best.main_camera);
		const double* center_assoc = camera_center(best.assoc_camera);
		double ray_main[3] = {
			xyz[0] - center_main[0],
			xyz[1] - center_main[1],
			xyz[2] - center_main[2]
		};
		double ray_assoc[3] = {
			xyz[0] - center_assoc[0],
			xyz[1] - center_assoc[1],
			xyz[2] - center_assoc[2]
		};
		const double main_dist2 =
			ray_main[0] * ray_main[0] +
			ray_main[1] * ray_main[1] +
			ray_main[2] * ray_main[2];
		const double assoc_dist2 =
			ray_assoc[0] * ray_assoc[0] +
			ray_assoc[1] * ray_assoc[1] +
			ray_assoc[2] * ray_assoc[2];

		if (apply_refinement &&
			((main_dist2 > 900.0 * assoc_dist2) ||
			(best.err_main > best.err_assoc && nviews == 2))) {
			std::swap(best.main_local, best.assoc_local);
			std::swap(best.main_camera, best.assoc_camera);
			std::swap(best.err_main, best.err_assoc);
			std::swap(ray_main[0], ray_assoc[0]);
			std::swap(ray_main[1], ray_assoc[1]);
			std::swap(ray_main[2], ray_assoc[2]);
		}

		m_archor[point_id * 3 + 1] = best.main_camera;
		m_archor[point_id * 3 + 2] = best.assoc_camera;
		m_archorSort[point_id * 2] = best.main_local;
		m_archorSort[point_id * 2 + 1] = best.assoc_local;
		feature_params[point_id * 3] = std::atan2(ray_main[0], ray_main[2]);
		feature_params[point_id * 3 + 1] =
			std::atan2(ray_main[1], std::sqrt(ray_main[0] * ray_main[0] + ray_main[2] * ray_main[2]));
		feature_params[point_id * 3 + 2] = best.omega;
	};

	int observation_start = 0;
	for (int point_id = 0; point_id < m_n3Dpts; ++point_id) {
		const int nviews = m_archor[point_id * 3];
		if (nviews < 2) {
			observation_start += std::max(0, nviews);
			continue;
		}

		const double* xyz = m_XYZ + point_id * 3;
		if (policy == AnchorPolicy::kCurrentPairPaperOrder ||
			policy == AnchorPolicy::kCurrentPairPaperOrderRefine) {
			int main_local = m_archorSort[point_id * 2];
			int assoc_local = m_archorSort[point_id * 2 + 1];
			if (main_local < 0 || assoc_local < 0 ||
				main_local >= nviews || assoc_local >= nviews ||
				main_local == assoc_local) {
				observation_start += nviews;
				continue;
			}
			if (main_local > assoc_local) {
				std::swap(main_local, assoc_local);
			}

			const int main_camera = m_photo[observation_start + main_local];
			const int assoc_camera = m_photo[observation_start + assoc_local];
			if (main_camera < 0 || main_camera >= m_ncams ||
				assoc_camera < 0 || assoc_camera >= m_ncams ||
				main_camera == assoc_camera) {
				observation_start += nviews;
				continue;
			}

			const double* center_main = camera_center(main_camera);
			const double* center_assoc = camera_center(assoc_camera);
			double ray_main[3] = {
				xyz[0] - center_main[0],
				xyz[1] - center_main[1],
				xyz[2] - center_main[2]
			};
			double ray_assoc[3] = {
				xyz[0] - center_assoc[0],
				xyz[1] - center_assoc[1],
				xyz[2] - center_assoc[2]
			};
			const double main_norm2 =
				ray_main[0] * ray_main[0] +
				ray_main[1] * ray_main[1] +
				ray_main[2] * ray_main[2];
			const double assoc_norm2 =
				ray_assoc[0] * ray_assoc[0] +
				ray_assoc[1] * ray_assoc[1] +
				ray_assoc[2] * ray_assoc[2];
			if (main_norm2 <= 1e-24 || assoc_norm2 <= 1e-24) {
				observation_start += nviews;
				continue;
			}

			double err_main = std::numeric_limits<double>::infinity();
			double err_assoc = std::numeric_limits<double>::infinity();
			projection_error(
				main_camera, ray_main, observation_start + main_local, &err_main);
			projection_error(
				assoc_camera, ray_assoc, observation_start + assoc_local, &err_assoc);
			const double dot =
				ray_main[0] * ray_assoc[0] +
				ray_main[1] * ray_assoc[1] +
				ray_main[2] * ray_assoc[2];
			Candidate best;
			best.main_local = main_local;
			best.assoc_local = assoc_local;
			best.main_camera = main_camera;
			best.assoc_camera = assoc_camera;
			best.omega = std::acos(
				clamp_acos_arg(dot / std::sqrt(main_norm2 * assoc_norm2)));
			best.sin_omega = std::sin(best.omega);
			best.err_main = err_main;
			best.err_assoc = err_assoc;
			best.err_sum = err_main + err_assoc;
			apply_selected_pair(
				point_id, nviews, xyz, best,
				policy == AnchorPolicy::kCurrentPairPaperOrderRefine);
			observation_start += nviews;
			continue;
		}

		std::vector<Candidate> candidates;
		candidates.reserve((nviews * (nviews - 1)) / 2);
		double min_err_sum = std::numeric_limits<double>::infinity();

		for (int a = 0; a < nviews; ++a) {
			const int camera_a = m_photo[observation_start + a];
			if (camera_a < 0 || camera_a >= m_ncams) {
				continue;
			}
			const double* center_a = camera_center(camera_a);
			double ray_a[3] = {
				xyz[0] - center_a[0],
				xyz[1] - center_a[1],
				xyz[2] - center_a[2]
			};
			const double ray_a_norm2 =
				ray_a[0] * ray_a[0] + ray_a[1] * ray_a[1] + ray_a[2] * ray_a[2];
			if (ray_a_norm2 <= 1e-24) {
				continue;
			}

			for (int b = a + 1; b < nviews; ++b) {
				const int camera_b = m_photo[observation_start + b];
				if (camera_b < 0 || camera_b >= m_ncams || camera_b == camera_a) {
					continue;
				}
				const double* center_b = camera_center(camera_b);
				double ray_b[3] = {
					xyz[0] - center_b[0],
					xyz[1] - center_b[1],
					xyz[2] - center_b[2]
				};
				const double ray_b_norm2 =
					ray_b[0] * ray_b[0] + ray_b[1] * ray_b[1] + ray_b[2] * ray_b[2];
				if (ray_b_norm2 <= 1e-24) {
					continue;
				}

				double err_a = 0.0;
				double err_b = 0.0;
				if (!projection_error(camera_a, ray_a, observation_start + a, &err_a) ||
					!projection_error(camera_b, ray_b, observation_start + b, &err_b)) {
					continue;
				}

				const double dot =
					ray_a[0] * ray_b[0] + ray_a[1] * ray_b[1] + ray_a[2] * ray_b[2];
				const double omega = std::acos(
					clamp_acos_arg(dot / std::sqrt(ray_a_norm2 * ray_b_norm2)));
				if (!std::isfinite(omega)) {
					continue;
				}

				const double baseline_x = center_b[0] - center_a[0];
				const double baseline_y = center_b[1] - center_a[1];
				const double baseline_z = center_b[2] - center_a[2];
				Candidate candidate;
				candidate.main_local = a;
				candidate.assoc_local = b;
				candidate.main_camera = camera_a;
				candidate.assoc_camera = camera_b;
				candidate.omega = omega;
				candidate.sin_omega = std::sin(omega);
				candidate.baseline = std::sqrt(
					baseline_x * baseline_x +
					baseline_y * baseline_y +
					baseline_z * baseline_z);
				candidate.err_main = err_a;
				candidate.err_assoc = err_b;
				candidate.err_sum = err_a + err_b;
				min_err_sum = std::min(min_err_sum, candidate.err_sum);
				candidates.push_back(candidate);
			}
		}

		if (candidates.empty()) {
			observation_start += nviews;
			continue;
		}

		const double reproj_gate = std::max(
			min_err_sum * 4.0,
			min_err_sum + 100.0);
		int best_index = -1;
		double best_score = -std::numeric_limits<double>::infinity();
		for (int candidate_id = 0; candidate_id < static_cast<int>(candidates.size());
			++candidate_id) {
			const Candidate& candidate = candidates[candidate_id];
			double score = -std::numeric_limits<double>::infinity();
			if (policy == AnchorPolicy::kTrueMaxParallax) {
				score = candidate.omega;
			} else if (policy == AnchorPolicy::kMaxParallaxReproj) {
				if (candidate.err_sum > reproj_gate) {
					continue;
				}
				score = candidate.omega;
			} else if (policy == AnchorPolicy::kScoreReproj) {
				score = candidate.sin_omega /
					(1.0 + std::sqrt(candidate.err_main) + std::sqrt(candidate.err_assoc));
			} else if (policy == AnchorPolicy::kScoreBaselineReproj) {
				score = candidate.sin_omega * candidate.baseline /
					(1.0 + std::sqrt(candidate.err_main) + std::sqrt(candidate.err_assoc));
			}

			if (best_index < 0 ||
				score > best_score + 1e-15 ||
				(std::abs(score - best_score) <= 1e-15 &&
					candidate.err_sum < candidates[best_index].err_sum)) {
				best_index = candidate_id;
				best_score = score;
			}
		}

		if (best_index < 0) {
			observation_start += nviews;
			continue;
		}

		Candidate best = candidates[best_index];
		const double* center_main = camera_center(best.main_camera);
		const double* center_assoc = camera_center(best.assoc_camera);
		double ray_main[3] = {
			xyz[0] - center_main[0],
			xyz[1] - center_main[1],
			xyz[2] - center_main[2]
		};
		double ray_assoc[3] = {
			xyz[0] - center_assoc[0],
			xyz[1] - center_assoc[1],
			xyz[2] - center_assoc[2]
		};
		const double main_dist2 =
			ray_main[0] * ray_main[0] +
			ray_main[1] * ray_main[1] +
			ray_main[2] * ray_main[2];
		const double assoc_dist2 =
			ray_assoc[0] * ray_assoc[0] +
			ray_assoc[1] * ray_assoc[1] +
			ray_assoc[2] * ray_assoc[2];

		if ((main_dist2 > 900.0 * assoc_dist2) ||
			(best.err_main > best.err_assoc && nviews == 2)) {
			std::swap(best.main_local, best.assoc_local);
			std::swap(best.main_camera, best.assoc_camera);
			std::swap(best.err_main, best.err_assoc);
			std::swap(center_main, center_assoc);
			std::swap(ray_main[0], ray_assoc[0]);
			std::swap(ray_main[1], ray_assoc[1]);
			std::swap(ray_main[2], ray_assoc[2]);
		}

		m_archor[point_id * 3 + 1] = best.main_camera;
		m_archor[point_id * 3 + 2] = best.assoc_camera;
		m_archorSort[point_id * 2] = best.main_local;
		m_archorSort[point_id * 2 + 1] = best.assoc_local;
		feature_params[point_id * 3] = std::atan2(ray_main[0], ray_main[2]);
		feature_params[point_id * 3 + 1] =
			std::atan2(ray_main[1], std::sqrt(ray_main[0] * ray_main[0] + ray_main[2] * ray_main[2]));
		feature_params[point_id * 3 + 2] = best.omega;

		observation_start += nviews;
	}
}


bool PBA::pba_initializeMainArchor( 
	double* imgpts,	
	double* camera,	
	double* K,		
	double* feature,
	int nP,			
	int FID,		
	double* KR )	
{
	//solve  KRX = x
	Vector3d x;
	if (m_bProvideXYZ)
	{
		x(0) = m_XYZ[FID*3] - *(camera + nP*6+3);
		x(1) = m_XYZ[FID*3+1] - *(camera + nP*6+4);
		x(2) = m_XYZ[FID*3+2] - *(camera + nP*6+5);
	}
	
	else
	{
		double *ptr = m_KR + nP*9;	
		Matrix3d  A;
		A << ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8];
		
		double matx[3];				
		matx[0] = imgpts[0];
		matx[1] = imgpts[1];
		matx[2] = 1;
		
		Vector3d  b(matx);
		
		x = A.colPivHouseholderQr().solve(b);
	}

	double* pKR = KR + nP*9;
	double t = pKR[6]*x(0) + pKR[7]*x(1) + pKR[8]*x(2);	

	//compute azimuth and elevation angle
	double dDAngle = atan2( x(0), x(2) );				
	double dHAngle = atan2( x(1), sqrt(x(0)*x(0)+ x(2)*x(2)) );

	feature[0] = dDAngle;
	feature[1] = dHAngle;
	feature[2] = 0;

	if ( t < 0 )
		return true;
	else
		return false;
}

bool PBA::pba_initializeAssoArchor( 
	double* imgpts,
	int* photo,
	double* camera,
	double* K,
	double* feature,
	int nMI,
	int nAI,
	int FID )
{
	int nM = photo[nMI];                           
	int nA = photo[nAI];                         

	Vector3d  xM, xA;

	if (m_bProvideXYZ)
	{
		xM[0] = m_XYZ[FID*3]   - *(camera + nM*6+3);
		xM[1] = m_XYZ[FID*3+1] - *(camera + nM*6+4);
		xM[2] = m_XYZ[FID*3+2] - *(camera + nM*6+5);

		xA[0] = m_XYZ[FID*3]   - *(camera + nA*6+3);
		xA[1] = m_XYZ[FID*3+1] - *(camera + nA*6+4);
		xA[2] = m_XYZ[FID*3+2] - *(camera + nA*6+5);
	}
	else
	{
		//Main anchor ray
		double *ptr1 = m_KR + nM*9;
		Matrix3d  AM;	
		AM << ptr1[0], ptr1[1], ptr1[2], ptr1[3], ptr1[4], ptr1[5], ptr1[6], ptr1[7], ptr1[8];

		double matxM[3];
		matxM[0] = *(imgpts+2*nMI);
		matxM[1] = *(imgpts+2*nMI+1);
		matxM[2] = 1;

		Vector3d  bM(matxM);
		xM = AM.colPivHouseholderQr().solve(bM);			

		//Associate archor ray
		double *ptr2 = m_KR + nA*9;
		Matrix3d  AA;	
		AA << ptr2[0], ptr2[1], ptr2[2], ptr2[3], ptr2[4], ptr2[5], ptr2[6], ptr2[7], ptr2[8];

		double matxA[3];
		matxA[0] = *(imgpts+2*nAI);
		matxA[1] = *(imgpts+2*nAI+1);
		matxA[2] = 1;

		Vector3d  bA(matxA);
		xA = AA.colPivHouseholderQr().solve(bA);			
	}
		
	//Parallax Angle
	double dDot = xM(0)*xA(0) + xM(1)*xA(1) + xM(2)*xA(2);			

	double dDisM = sqrt( xM(0)*xM(0)+xM(1)*xM(1)+xM(2)*xM(2) );		
	double dDisA = sqrt( xA(0)*xA(0)+xA(1)*xA(1)+xA(2)*xA(2) );		

	if (dDot/(dDisM*dDisA)>1)			
		feature[2] = 0;
	else if (dDot/(dDisM*dDisA)<-1)		
		feature[2] = PI;
	else
	{
		double dw = acos(clamp_acos_arg(dDot / (dDisM * dDisA)));
		feature[2] = dw;
	}

	
	double pti2k[3];
	pti2k[0] = *(camera + nA*6+3) - *(camera + nM*6+3);    
	pti2k[1] = *(camera + nA*6+4) - *(camera + nM*6+4);    
	pti2k[2] = *(camera + nA*6+5) - *(camera + nM*6+5);    

	double dDot1 = xM[0]*pti2k[0] + xM[1]*pti2k[1] + xM[2]*pti2k[2];
	double dDisi2k = sqrt( pti2k[0]*pti2k[0] + pti2k[1]*pti2k[1] + pti2k[2]*pti2k[2] );
	double tmp = dDot1/(dDisM*dDisi2k);
	tmp = clamp_acos_arg(tmp);
	double dW2 = acos(tmp);

	return true;
}

bool PBA::pba_initializeOtheArchors( 
	double* imgpts,
	int* photo,
	double* camera,
	double* K,
	double* feature,
	int* archorSort,
	int nfeacout,
	int nOI,
	int FID )
{
	double dw = feature[2];                   
	double dwNew;
	double dmaxw = dw;
	int   nNewI = 0;
	bool bAdjust = false;
	double dDot,dDisM,dDisA;

	if ( dw < MAXARCHOR   )
	{
		//current archor vector 
		int nO = photo[nOI];

		Vector3d  xO;
		if ( m_bProvideXYZ)
		{
			xO(0) = m_XYZ[FID*3] - *(camera + nO*6 + 3);
			xO(1) = m_XYZ[FID*3+1]-*(camera + nO*6 + 4);
			xO(2) = m_XYZ[FID*3+2]-*(camera + nO*6 + 5);
		}
		else
		{
			double *ptr1 = m_KR + nO*9;
			Matrix3d  AO;	
			AO << ptr1[0], ptr1[1], ptr1[2], ptr1[3], ptr1[4], ptr1[5], ptr1[6], ptr1[7], ptr1[8];

			double matxO[3];
			matxO[0] = *(imgpts+nOI*2);
			matxO[1] = *(imgpts+nOI*2+1);
			matxO[2] = 1;

			Vector3d  bO(matxO);
			xO = AO.colPivHouseholderQr().solve(bO);
		}

		double dDAngle = atan2( xO(0), xO(2) );
		double dHAngle = atan2( xO(1), sqrt(xO(0)*xO(0)+ xO(2)*xO(2)) );

		for (int i = 0; i < nfeacout; i++ )
		{
			//Main Archor Vector
			int nM = photo[i];                            
			Vector3d  xM;

			if (m_bProvideXYZ)
			{
				xM(0) = m_XYZ[FID*3]  -*(camera + nM*6+3);
				xM(1) = m_XYZ[FID*3+1]-*(camera + nM*6+4);
				xM(2) = m_XYZ[FID*3+2]-*(camera + nM*6+5);
			}
			else
			{
				double *ptr2 = m_KR + nM*9;
				Matrix3d  AM;	
				AM << ptr2[0], ptr2[1], ptr2[2], ptr2[3], ptr2[4], ptr2[5], ptr2[6], ptr2[7], ptr2[8];

				double matxM[3];
				matxM[0] = *(imgpts+i*2);
				matxM[1] = *(imgpts+i*2+1);
				matxM[2] = 1;

				Vector3d  bM(matxM);
				xM = AM.colPivHouseholderQr().solve(bM);
			}

			//Parallax angle between current archor and main archor
			dDot = xM(0)*xO(0) + xM(1)*xO(1) + xM(2)*xO(2);
			dDisM = sqrt( xM(0)*xM(0)+xM(1)*xM(1)+xM(2)*xM(2) );
			dDisA = sqrt( xO(0)*xO(0)+xO(1)*xO(1)+xO(2)*xO(2) );

			if( dDot/(dDisM*dDisA) > 1 )
				dwNew = 0;
			else if(dDot/(dDisM*dDisA)<-1)
				dwNew = PI;
			else
				dwNew = acos(clamp_acos_arg(dDot / (dDisM * dDisA)));

			if ( dwNew > dmaxw )
			{
				dmaxw = dwNew;
				archorSort[0] = nOI;   
				archorSort[1] = i;     
				feature[0] = dDAngle;
				feature[1] = dHAngle;
				feature[2] = dmaxw;
				bAdjust = true;
			}
		}
	}
	return bAdjust;
}
