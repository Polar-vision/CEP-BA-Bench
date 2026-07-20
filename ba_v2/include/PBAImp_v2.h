#pragma once
#include "BAExporter_v2.h"
#include "IBA_v2.h"
#include <map>
#include <vector>
#include <algorithm>
#include <thread>
#include "ceres/ceres.h"
#include "CameraFrameParameterizations.h"

using namespace std;

class PBA : public IBA
{
public:
	PBA(void);
	~PBA(void);
	virtual bool ba_run(char* szCam,
		                char* szFea, 
						char* szXYZ, 
						char* szCalib, 
						char* szReport,
						char* szPose, 
						char* sz3D,
						MethodId method,
						BenchmarkOutputMode output_mode,
						int point_condition_sample,
						int schur_sample) override;


	virtual bool ba_initialize( char* szCamera, char* szFeature, char* szCalib = NULL, char* szXYZ = NULL );

	void	pba_readAndInitialize( char *camsfname, char *ptsfname, char *calibfname, int *ncams, int *n3Dpts, int *n2Dprojs,double **motstruct, 
				double **imgpts, int **archor, char **vmask, char **umask,int **nphoto, int** nfeature, int** archorSort );
	void	pba_readProjectionAndInitilizeFeature(	FILE *fp, double *params, double *projs, char *vmask, int ncams, 
				int *archor,char* umask,int* nphoto, int* nfeature, int* archorSort );
	//initialize feature. You can provide xyz or system also provide them by itself;
	bool    pba_initializeMainArchor( double* imgpts, double* camera,double* K,double* feature, int nP, int FID, double* KR );
	bool    pba_initializeAssoArchor( double* imgpts, int* photo, double* camera,double* K,double* feature,int nMI, int nAI, int FID );
	bool	pba_initializeOtheArchors( double* imgpts, int* photo, double* camera,double* K,double* feature,int* archorSort,int nfeacout, int nOI, int FID );
	void	pba_applyAnchorPolicyAblation(double* feature_params);
	void parallax2xyz();
	void xy_inverse_z2xyz();
	void depth2xyz();
	void inverse_depth2xyz();
	void archored_inverse_depth2xyz();
	void archored_depth2xyz();
	void archored_xy_inverse_z2xyz();
	void archored_xyz2xyz();
	void anchor_camera_xyz2xyz();
	void anchor_camera_xy_inverse_z2xyz();
	void anchor_camera_spherical_range2xyz();
	void anchor_camera_spherical_inverse_range2xyz();
	void parallax_camera2xyz();
	//Parameterization of object point
	//zero archor
	struct xyz_euler_angle_uv {
	xyz_euler_angle_uv(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center,
					const T* const point3D,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		// T c0, c1, c2, s0, s1, s2;
		// c0 = cos(euler_angles[0]);
		// c1 = cos(euler_angles[1]);
		// c2 = cos(euler_angles[2]);
		// s0 = sin(euler_angles[0]);
		// s1 = sin(euler_angles[1]);
		// s2 = sin(euler_angles[2]);
		// R[0] = c1 * c0;
		// R[1] = c1 * s0;
		// R[2] = -s1;
		// R[3] = s2 * s1 * c0 - c2 * s0;
		// R[4] = s2 * s1 * s0 + c2 * c0;
		// R[5] = s2 * c1;
		// R[6] = c2 * s1 * c0 + s2 * s0;
		// R[7] = c2 * s1 * s0 - s2 * c0;
		// R[8] = c2 * c1;

		T Xc[3];
		Xc[0] = point3D[0] - camera_center[0];
		Xc[1] = point3D[1] - camera_center[1];
		Xc[2] = point3D[2] - camera_center[2];

		// P = R * (X - C)
		T p[3];
		p[0] = R[0] * Xc[0] + R[1] * Xc[1] + R[2] * Xc[2];//X
		p[1] = R[3] * Xc[0] + R[4] * Xc[1] + R[5] * Xc[2];//Y
		p[2] = R[6] * Xc[0] + R[7] * Xc[1] + R[8] * Xc[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<xyz_euler_angle_uv, 2, 3, 3, 3>(
			new xyz_euler_angle_uv(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct xy_inverse_z_euler_angle_uv {
	xy_inverse_z_euler_angle_uv(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center,
					const T* const point3D_inverse_z,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T Xc[3];
		// Scale the ray by rho to avoid division:
		// X^W = [xi, eta, 1]^T / rho.
		Xc[0] = point3D_inverse_z[0] - point3D_inverse_z[2] * camera_center[0];
		Xc[1] = point3D_inverse_z[1] - point3D_inverse_z[2] * camera_center[1];
		Xc[2] = T(1) - point3D_inverse_z[2] * camera_center[2];

		// P = R * (X - C)
		T p[3];
		p[0] = R[0] * Xc[0] + R[1] * Xc[1] + R[2] * Xc[2];//X
		p[1] = R[3] * Xc[0] + R[4] * Xc[1] + R[5] * Xc[2];//Y
		p[2] = R[6] * Xc[0] + R[7] * Xc[1] + R[8] * Xc[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<xy_inverse_z_euler_angle_uv, 2, 3, 3, 3>(
			new xy_inverse_z_euler_angle_uv(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct depth_euler_angle_uv {
	depth_euler_angle_uv(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center,
					const T* const point3D,//direction+depth
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T ptXj[3];//take world system as reference
		ptXj[0] = sin(point3D[0]) * cos(point3D[1]);
		ptXj[1] = sin(point3D[1]);
		ptXj[2] = cos(point3D[0]) * cos(point3D[1]);

		T Xc[3];
		Xc[0] = ptXj[0]*point3D[2] - camera_center[0];
		Xc[1] = ptXj[1]*point3D[2] - camera_center[1];
		Xc[2] = ptXj[2]*point3D[2] - camera_center[2];

		// P = R * (X - C)
		T p[3];
		p[0] = R[0] * Xc[0] + R[1] * Xc[1] + R[2] * Xc[2];//X
		p[1] = R[3] * Xc[0] + R[4] * Xc[1] + R[5] * Xc[2];//Y
		p[2] = R[6] * Xc[0] + R[7] * Xc[1] + R[8] * Xc[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<depth_euler_angle_uv, 2, 3, 3, 3>(
			new depth_euler_angle_uv(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct inverse_depth_euler_angle_uv {
	inverse_depth_euler_angle_uv(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center,
					const T* const point3D,//direction+inverse depth
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T ptXj[3];//take world system as reference
		ptXj[0] = sin(point3D[0]) * cos(point3D[1]);
		ptXj[1] = sin(point3D[1]);
		ptXj[2] = cos(point3D[0]) * cos(point3D[1]);

		T Xc[3];
		Xc[0] = ptXj[0]/point3D[2] - camera_center[0];
		Xc[1] = ptXj[1]/point3D[2] - camera_center[1];
		Xc[2] = ptXj[2]/point3D[2] - camera_center[2];

		// P = R * (X - C)
		T p[3];
		p[0] = R[0] * Xc[0] + R[1] * Xc[1] + R[2] * Xc[2];//X
		p[1] = R[3] * Xc[0] + R[4] * Xc[1] + R[5] * Xc[2];//Y
		p[2] = R[6] * Xc[0] + R[7] * Xc[1] + R[8] * Xc[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<inverse_depth_euler_angle_uv, 2, 3, 3, 3>(
			new inverse_depth_euler_angle_uv(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	
	//one archor
	struct archored_xyz_euler_angle_uv_nM {
	archored_xyz_euler_angle_uv_nM(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const archored_point3D,//鐩歌緝浜庝富閿氱偣鐨勭瑳鍗″皵鍧愭爣
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T p[3];
		p[0] = R[0] * archored_point3D[0] + R[1] * archored_point3D[1] + R[2] * archored_point3D[2];//X
		p[1] = R[3] * archored_point3D[0] + R[4] * archored_point3D[1] + R[5] * archored_point3D[2];//Y
		p[2] = R[6] * archored_point3D[0] + R[7] * archored_point3D[1] + R[8] * archored_point3D[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_xyz_euler_angle_uv_nM, 2, 3, 3>(
			new archored_xyz_euler_angle_uv_nM(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_xyz_euler_angle_uv_nP {
	archored_xyz_euler_angle_uv_nP(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center_nP,
					const T* const camera_center_nM,
					const T* const archored_point3D,//鐩歌緝浜庝富閿氱偣鐨勭瑳鍗″皵鍧愭爣
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T nM2nP[3];
		nM2nP[0] = camera_center_nP[0] - camera_center_nM[0];
		nM2nP[1] = camera_center_nP[1] - camera_center_nM[1];
		nM2nP[2] = camera_center_nP[2] - camera_center_nM[2];
		T ray[3];
		ray[0] = archored_point3D[0] - nM2nP[0];
		ray[1] = archored_point3D[1] - nM2nP[1];
		ray[2] = archored_point3D[2] - nM2nP[2];

		T p[3];
		p[0] = R[0] * ray[0] + R[1] * ray[1] + R[2] * ray[2];
		p[1] = R[3] * ray[0] + R[4] * ray[1] + R[5] * ray[2];
		p[2] = R[6] * ray[0] + R[7] * ray[1] + R[8] * ray[2];

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_xyz_euler_angle_uv_nP, 2, 3, 3, 3, 3>(
			new archored_xyz_euler_angle_uv_nP(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_xy_inverse_z_euler_angle_uv_nM {
	archored_xy_inverse_z_euler_angle_uv_nM(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const archored_point3D_idt,//鐩歌緝浜庝富閿氱偣鐨勭瑳鍗″皵鍧愭爣
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T p[3];
		// The common 1/rho scale cancels in perspective projection.
		p[0] = R[0] * archored_point3D_idt[0] + R[1] * archored_point3D_idt[1] + R[2];
		p[1] = R[3] * archored_point3D_idt[0] + R[4] * archored_point3D_idt[1] + R[5];
		p[2] = R[6] * archored_point3D_idt[0] + R[7] * archored_point3D_idt[1] + R[8];

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_xy_inverse_z_euler_angle_uv_nM, 2, 3, 3>(
			new archored_xy_inverse_z_euler_angle_uv_nM(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_xy_inverse_z_euler_angle_uv_nP {
	archored_xy_inverse_z_euler_angle_uv_nP(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center_nP,
					const T* const camera_center_nM,
					const T* const archored_point3D_idt,//鐩歌緝浜庝富閿氱偣鐨勭瑳鍗″皵鍧愭爣
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T nM2nP[3];
		nM2nP[0] = camera_center_nP[0] - camera_center_nM[0];
		nM2nP[1] = camera_center_nP[1] - camera_center_nM[1];
		nM2nP[2] = camera_center_nP[2] - camera_center_nM[2];
		T ray[3];
		// Scale the current-camera ray by rho.
		ray[0] = archored_point3D_idt[0] - archored_point3D_idt[2] * nM2nP[0];
		ray[1] = archored_point3D_idt[1] - archored_point3D_idt[2] * nM2nP[1];
		ray[2] = T(1) - archored_point3D_idt[2] * nM2nP[2];

		T p[3];
		p[0] = R[0] * ray[0] + R[1] * ray[1] + R[2] * ray[2];
		p[1] = R[3] * ray[0] + R[4] * ray[1] + R[5] * ray[2];
		p[2] = R[6] * ray[0] + R[7] * ray[1] + R[8] * ray[2];

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_xy_inverse_z_euler_angle_uv_nP, 2, 3, 3, 3, 3>(
			new archored_xy_inverse_z_euler_angle_uv_nP(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_depth_euler_angle_uv_nM {
	archored_depth_euler_angle_uv_nM(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const point3D_parameters,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T ptXj[3];
		ptXj[0] = sin(point3D_parameters[0]) * cos(point3D_parameters[1]);
		ptXj[1] = sin(point3D_parameters[1]);
		ptXj[2] = cos(point3D_parameters[0]) * cos(point3D_parameters[1]);

		T p[3];
		p[0] = R[0] * ptXj[0] + R[1] * ptXj[1] + R[2] * ptXj[2];//X
		p[1] = R[3] * ptXj[0] + R[4] * ptXj[1] + R[5] * ptXj[2];//Y
		p[2] = R[6] * ptXj[0] + R[7] * ptXj[1] + R[8] * ptXj[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_depth_euler_angle_uv_nM, 2, 3, 3>(
			new archored_depth_euler_angle_uv_nM(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_depth_euler_angle_uv_nP {
	archored_depth_euler_angle_uv_nP(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center_nP,
					const T* const camera_center_nM,
					const T* const point3D_parameters,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T ptXj[3];
		ptXj[0] = sin(point3D_parameters[0]) * cos(point3D_parameters[1]) * point3D_parameters[2];
		ptXj[1] = sin(point3D_parameters[1]) * point3D_parameters[2];
		ptXj[2] = cos(point3D_parameters[0]) * cos(point3D_parameters[1]) * point3D_parameters[2];
		T ptXm[3];
		ptXm[0] = camera_center_nP[0] - camera_center_nM[0];
		ptXm[1] = camera_center_nP[1] - camera_center_nM[1];
		ptXm[2] = camera_center_nP[2] - camera_center_nM[2];

		T Xc[3];
		Xc[0] = ptXj[0] - ptXm[0];
		Xc[1] = ptXj[1] - ptXm[1];
		Xc[2] = ptXj[2] - ptXm[2];

		// P = R * (X - C)
		T p[3];
		p[0] = R[0] * Xc[0] + R[1] * Xc[1] + R[2] * Xc[2];//X
		p[1] = R[3] * Xc[0] + R[4] * Xc[1] + R[5] * Xc[2];//Y
		p[2] = R[6] * Xc[0] + R[7] * Xc[1] + R[8] * Xc[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_depth_euler_angle_uv_nP, 2, 3, 3, 3, 3>(
			new archored_depth_euler_angle_uv_nP(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_inverse_depth_euler_angle_uv_nM {
	archored_inverse_depth_euler_angle_uv_nM(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const point3D_parameters,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T ptXj[3];
		ptXj[0] = sin(point3D_parameters[0]) * cos(point3D_parameters[1]);
		ptXj[1] = sin(point3D_parameters[1]);
		ptXj[2] = cos(point3D_parameters[0]) * cos(point3D_parameters[1]);

		T p[3];
		p[0] = R[0] * ptXj[0] + R[1] * ptXj[1] + R[2] * ptXj[2];//X
		p[1] = R[3] * ptXj[0] + R[4] * ptXj[1] + R[5] * ptXj[2];//Y
		p[2] = R[6] * ptXj[0] + R[7] * ptXj[1] + R[8] * ptXj[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_inverse_depth_euler_angle_uv_nM, 2, 3, 3>(
			new archored_inverse_depth_euler_angle_uv_nM(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct archored_inverse_depth_euler_angle_uv_nP {
	archored_inverse_depth_euler_angle_uv_nP(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center_nP,
					const T* const camera_center_nM,
					const T* const point3D_parameters,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T ptXj[3];
		ptXj[0] = sin(point3D_parameters[0]) * cos(point3D_parameters[1]) / point3D_parameters[2];
		ptXj[1] = sin(point3D_parameters[1]) / point3D_parameters[2];
		ptXj[2] = cos(point3D_parameters[0]) * cos(point3D_parameters[1]) / point3D_parameters[2];
		T ptXm[3];
		ptXm[0] = camera_center_nP[0] - camera_center_nM[0];
		ptXm[1] = camera_center_nP[1] - camera_center_nM[1];
		ptXm[2] = camera_center_nP[2] - camera_center_nM[2];

		T Xc[3];
		Xc[0] = ptXj[0] - ptXm[0];
		Xc[1] = ptXj[1] - ptXm[1];
		Xc[2] = ptXj[2] - ptXm[2];

		// P = R * (X - C)
		T p[3];
		p[0] = R[0] * Xc[0] + R[1] * Xc[1] + R[2] * Xc[2];//X
		p[1] = R[3] * Xc[0] + R[4] * Xc[1] + R[5] * Xc[2];//Y
		p[2] = R[6] * Xc[0] + R[7] * Xc[1] + R[8] * Xc[2];//Z

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<archored_inverse_depth_euler_angle_uv_nP, 2, 3, 3, 3, 3>(
			new archored_inverse_depth_euler_angle_uv_nP(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};

	//two archors
	struct parallax_euler_angle_uv_nM {
	parallax_euler_angle_uv_nM(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const point3D_direction,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T p[3];
		T ptXj[3];
		ptXj[0] = sin(point3D_direction[0]) * cos(point3D_direction[1]);
		ptXj[1] = sin(point3D_direction[1]);
		ptXj[2] = cos(point3D_direction[0]) * cos(point3D_direction[1]);
		p[0] = R[0] * ptXj[0] + R[1] * ptXj[1] + R[2] * ptXj[2];
		p[1] = R[3] * ptXj[0] + R[4] * ptXj[1] + R[5] * ptXj[2];
		p[2] = R[6] * ptXj[0] + R[7] * ptXj[1] + R[8] * ptXj[2];

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);

		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {		
		return new ceres::AutoDiffCostFunction<parallax_euler_angle_uv_nM, 2, 3, 3>(
			new parallax_euler_angle_uv_nM(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct parallax_euler_angle_uv_nA {
	parallax_euler_angle_uv_nA(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center_nP,
					const T* const camera_center_nM,
					const T* const point3D_parameters,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;

		T p[3], pti2k[3], ptXUnit[3], ptXk[3];
		pti2k[0] = camera_center_nP[0] - camera_center_nM[0];	
		pti2k[1] = camera_center_nP[1] - camera_center_nM[1];	
		pti2k[2] = camera_center_nP[2] - camera_center_nM[2];	

		ptXUnit[0] = sin(point3D_parameters[0]) * cos(point3D_parameters[1]);
		ptXUnit[1] = sin(point3D_parameters[1]);
		ptXUnit[2] = cos(point3D_parameters[0]) * cos(point3D_parameters[1]);

		// sin(w2 + parallax) is evaluated without acos; this keeps the same ray
		// direction while avoiding an expensive inverse-trig call per residual.
		T dDot = ptXUnit[0]*pti2k[0] + ptXUnit[1]*pti2k[1] + ptXUnit[2]*pti2k[2];
		T dDisi2k = sqrt(
			pti2k[0]*pti2k[0] + pti2k[1]*pti2k[1] +
			pti2k[2]*pti2k[2] + T(1e-24));
		T cos_w2 = dDot / dDisi2k;
		if (cos_w2 > T(1))
			cos_w2 = T(1);
		else if (cos_w2 < T(-1))
			cos_w2 = T(-1);
		T sin_w2_squared = T(1) - cos_w2 * cos_w2;
		if (sin_w2_squared < T(0))
			sin_w2_squared = T(0);
		T sin_w2 = sqrt(sin_w2_squared + T(1e-24));
		T sin_parallax = sin(point3D_parameters[2]);
		T sin_w2_plus_parallax =
			sin_w2 * cos(point3D_parameters[2]) + cos_w2 * sin_parallax;

		//compute Xk vector according sin theory
		ptXk[0] = dDisi2k * sin_w2_plus_parallax * ptXUnit[0] - sin_parallax * pti2k[0];
		ptXk[1] = dDisi2k * sin_w2_plus_parallax * ptXUnit[1] - sin_parallax * pti2k[1];
		ptXk[2] = dDisi2k * sin_w2_plus_parallax * ptXUnit[2] - sin_parallax * pti2k[2];

		p[0] = R[0] * ptXk[0] + R[1] * ptXk[1] + R[2] * ptXk[2];
		p[1] = R[3] * ptXk[0] + R[4] * ptXk[1] + R[5] * ptXk[2];
		p[2] = R[6] * ptXk[0] + R[7] * ptXk[1] + R[8] * ptXk[2];

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);
		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {	
		// printf("%s\n","ReprojectionError entered");	
		return new ceres::AutoDiffCostFunction<parallax_euler_angle_uv_nA, 2, 3, 3, 3, 3>(
			new parallax_euler_angle_uv_nA(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	struct parallax_euler_angle_uv_nP {
	parallax_euler_angle_uv_nP(double observed_u, double observed_v,
					double fx, double fy, double cx, double cy)
		: u(observed_u), v(observed_v),
		fx(fx), fy(fy), cx(cx), cy(cy) {}

	template <typename T>
	bool operator()(const T* const euler_angles,
					const T* const camera_center_nP,
					const T* const camera_center_nM,
					const T* const camera_center_nA,
					const T* const point3D_parameters,
					T* residuals) const {
		T R[9];
		T ey = euler_angles[0];
		T ex = euler_angles[1];
		T ez = euler_angles[2];
		T c1 = cos(ey);   T c2 = cos(ex);   T c3 = cos(ez);
		T s1 = sin(ey);   T s2 = sin(ex);   T s3 = sin(ez);
		R[0]=c1*c3-s1*s2*s3;     R[1]=c2*s3;     R[2]=s1*c3+c1*s2*s3;
		R[3]=-c1*s3-s1*s2*c3;    R[4]=c2*c3;     R[5]=-s1*s3+c1*s2*c3;
		R[6]=-s1*c2;             R[7]=-s2;       R[8]=c1*c2;
	
		T p[3], pti2k[3], pti2l[3], ptXUnit[3], ptXk[3];

		pti2k[0] = camera_center_nA[0] - camera_center_nM[0];		
		pti2k[1] = camera_center_nA[1] - camera_center_nM[1];		
		pti2k[2] = camera_center_nA[2] - camera_center_nM[2];
		pti2l[0] = camera_center_nP[0] - camera_center_nM[0];
		pti2l[1] = camera_center_nP[1] - camera_center_nM[1];		
		pti2l[2] = camera_center_nP[2] - camera_center_nM[2];
		
		ptXUnit[0] = sin(point3D_parameters[0]) * cos(point3D_parameters[1]);
		ptXUnit[1] = sin(point3D_parameters[1]);
		ptXUnit[2] = cos(point3D_parameters[0]) * cos(point3D_parameters[1]);

		// sin(w2 + parallax) is evaluated without acos; this keeps the same ray
		// direction while avoiding an expensive inverse-trig call per residual.
		T dDot = ptXUnit[0]*pti2k[0] + ptXUnit[1]*pti2k[1]+ ptXUnit[2]*pti2k[2];
		T dDisi2k = sqrt(
			pti2k[0]*pti2k[0] + pti2k[1]*pti2k[1] +
			pti2k[2]*pti2k[2] + T(1e-24));
		T cos_w2 = dDot / dDisi2k;
		if (cos_w2 > T(1))
			cos_w2 = T(1);
		else if (cos_w2 < T(-1))
			cos_w2 = T(-1);
		T sin_w2_squared = T(1) - cos_w2 * cos_w2;
		if (sin_w2_squared < T(0))
			sin_w2_squared = T(0);
		T sin_w2 = sqrt(sin_w2_squared + T(1e-24));
		T sin_parallax = sin(point3D_parameters[2]);
		T sin_w2_plus_parallax =
			sin_w2 * cos(point3D_parameters[2]) + cos_w2 * sin_parallax;

		//compute Xl vector according sin theory
		ptXk[0] = dDisi2k * sin_w2_plus_parallax * ptXUnit[0] - sin_parallax * pti2l[0];
		ptXk[1] = dDisi2k * sin_w2_plus_parallax * ptXUnit[1] - sin_parallax * pti2l[1];
		ptXk[2] = dDisi2k * sin_w2_plus_parallax * ptXUnit[2] - sin_parallax * pti2l[2];
		
		p[0] = R[0] * ptXk[0] + R[1] * ptXk[1] + R[2] * ptXk[2];
		p[1] = R[3] * ptXk[0] + R[4] * ptXk[1] + R[5] * ptXk[2];
		p[2] = R[6] * ptXk[0] + R[7] * ptXk[1] + R[8] * ptXk[2];

		// Normalize
		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		// Project
		T predicted_u = fx * xp + cx;
		T predicted_v = fy * yp + cy;

		// Residual
		residuals[0] = predicted_u - T(u);
		residuals[1] = predicted_v - T(v);
		
		return true;
	}

	static ceres::CostFunction* Create(double u, double v,
									double fx, double fy, double cx, double cy) {	
		// printf("%s\n","ReprojectionError entered");	
		return new ceres::AutoDiffCostFunction<parallax_euler_angle_uv_nP, 2, 3, 3, 3, 3, 3>(
			new parallax_euler_angle_uv_nP(u, v, fx, fy, cx, cy));
	}

	double u, v;
	double fx, fy, cx, cy;
	};
	

};
