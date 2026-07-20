
#include "IBA_v2.h"
#include <stdio.h>

int IBA::findNcameras(FILE* fp)
{
	int lineno, ncams, ch;

	lineno = ncams = 0;
	while (!feof(fp))
	{
		if ((ch = fgetc(fp)) == '#') { /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}

		if (feof(fp)) break;

		ungetc(ch, fp);

		SKIP_LINE(fp);
		++lineno;
		if (ferror(fp))
		{
			fprintf(stderr, "findNcameras(): error reading input file, line %d\n", lineno);
			exit(1);
		}
		++ncams;
	}
	return ncams;
}

IBA::IBA(void)
{
}

IBA::~IBA(void)
{
}

int IBA::countNDoubles(FILE* fp)
{
	int lineno, ch, np, i;
	char buf[MAXSTRLEN], * s;
	double dummy;

	lineno = 0;
	while (!feof(fp))
	{
		if ((ch = fgetc(fp)) == '#')
		{ /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}

		if (feof(fp)) return 0;

		ungetc(ch, fp);
		++lineno;
		if (!fgets(buf, MAXSTRLEN - 1, fp)) { /* read the line found... */
			fprintf(stderr, "countNDoubles(): error reading input file, line %d\n", lineno);
			exit(1);
		}
		/* ...and count the number of doubles it has */
		for (np = i = 0, s = buf; 1; ++np, s += i) {
			ch = sscanf_s(s, "%lf%n", &dummy, &i);
			if (ch == 0 || ch == EOF) break;
		}

		rewind(fp);
		return np;
	}
	return 0; // should not reach this point
}

int IBA::skipNDoubles(FILE* fp, int nvals)
{
	int i;
	int j;

	for (i = 0; i < nvals; ++i)
	{
		j = fscanf_s(fp, "%*f");
		if (j == EOF) return EOF;

		if (ferror(fp)) return EOF - 1;
	}

	return nvals;
}

void IBA::readNpointsAndNprojections(FILE* fp, int* n3Dpts, int pnp, int* nprojs, int mnp)
{
	int nfirst, lineno, npts, nframes, ch, n;

	/* #parameters for the first line */
	nfirst = countNDoubles(fp);

	*n3Dpts = *nprojs = lineno = npts = 0;
	while (!feof(fp))
	{
		if ((ch = fgetc(fp)) == '#')
		{ /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}

		if (feof(fp)) break;

		ungetc(ch, fp);
		++lineno;
		//skipNDoubles(fp, pnp);
		n = readNInts(fp, &nframes, 1);
		
		if (n != 1)
			exit(1);

		//printf("%d ", nframes);

		SKIP_LINE(fp);
		*nprojs += nframes;
		++npts;
	}

	*n3Dpts = npts;
}

void IBA::ba_readCablibration(FILE* fp, double* K)
{
	int n = fscanf_s(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf\n", &K[0], &K[3], &K[6], &K[1], &K[4], &K[7], &K[2], &K[5], &K[8]);

	if (n != 9)
	{
		fprintf(stderr, "BA error: Format of Calibaration is wrong\n");
		exit(1);
	}
}

void IBA::ba_readCameraPose(FILE* fp, double* params, int* m_v)
{
	int n, num, lineno = 0;
	double* tofilter;
	double* pPrams = params;
	int* pm_C = m_v;
	int jjg = 0;
	//the number of element per line is 8, it represents that focal length vary, or it is constant
	num = countNDoubles(fp);
	if (num == 8)
	{
		m_bFocal = true;
		m_K = (double*)malloc(m_ncams * 2 * sizeof(double));
		tofilter = (double*)malloc(8 * sizeof(double));
	}
	else if (num == 7) {
		tofilter = (double*)malloc(7 * sizeof(double));
	}
	else
		tofilter = (double*)malloc(6 * sizeof(double));

	while (!feof(fp))
	{
		if (num == 6) {
			n = readNDoubles(fp, tofilter, 6);
			if (n == -1) {
				//printf("%d %d\n", lineno, jjg);
				break;
			}
			Camera cam;
			cam.euler_angle[0] = tofilter[0];
			cam.euler_angle[1] = tofilter[1];
			cam.euler_angle[2] = tofilter[2];
			cam.camera_center[0] = tofilter[3];
			cam.camera_center[1] = tofilter[4];
			cam.camera_center[2] = tofilter[5];
			cam.camidx = 1;
			cams.push_back(cam);

			pPrams[0] = tofilter[0];	pPrams[1] = tofilter[1];	pPrams[2] = tofilter[2];
			pPrams[3] = tofilter[3];	pPrams[4] = tofilter[4];	pPrams[5] = tofilter[5];
			*pm_C = 1;
			//m_v[lineno++] = 1;
		}
		if (num == 7){
			n = readNDoubles(fp, tofilter, 7);
			if (n == -1) {
				//printf("%d %d\n", lineno, jjg);
				break;
			}

			Camera cam;
			cam.euler_angle[0] = tofilter[0];
			cam.euler_angle[1] = tofilter[1];
			cam.euler_angle[2] = tofilter[2];
			cam.camera_center[0] = tofilter[3];
			cam.camera_center[1] = tofilter[4];
			cam.camera_center[2] = tofilter[5];
			cam.camidx = static_cast<int>(tofilter[6]);
			cams.push_back(cam);

			pPrams[0] = tofilter[0];	pPrams[1] = tofilter[1];	pPrams[2] = tofilter[2];
			pPrams[3] = tofilter[3];	pPrams[4] = tofilter[4];	pPrams[5] = tofilter[5];
			*pm_C = static_cast<int>(tofilter[6]);
			++jjg;
		}
		if (num == 8){
			n = readNDoubles(fp, tofilter, 8);
			if (n == -1) {
				//printf("%d %d\n", lineno, jjg);
				break;
			}
			pPrams[0] = tofilter[0];	pPrams[1] = tofilter[1];	pPrams[2] = tofilter[2];
			pPrams[3] = tofilter[3];	pPrams[4] = tofilter[4];	pPrams[5] = tofilter[5];

			m_K[lineno * 2] = tofilter[6];
			m_K[lineno * 2 + 1] = tofilter[7];
		}
			
		pPrams += 6;
		if (num == 7 || num == 6) {
			pm_C += 1;
		}
		++lineno;
	}
	if (tofilter != NULL) {
		free(tofilter);
		tofilter = NULL;
	}
}

int IBA::readNInts(FILE* fp, int* vals, int nvals)
{
	int i;
	int n, j;

	for (i = n = 0; i < nvals; ++i) {
		j = fscanf_s(fp, "%d", vals + i);
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}
	return n;
}

int IBA::readNDoubles(FILE* fp, double* vals, int nvals)
{
	int i;
	int n, j;

	for (i = n = 0; i < nvals; ++i)
	{
		j = fscanf_s(fp, "%lf", vals + i);
		if (j == EOF) return EOF;

		if (j != 1 || ferror(fp)) return EOF - 1;

		n += j;
	}

	return n;
}

void IBA::ba_readCameraPoseration(char* fname, double* ical)
{
	FILE* fp = nullptr;
	int  ch = EOF;
	fopen_s(&fp, fname, "r");
	if (fp == nullptr)
	{
		fprintf(stderr, "BA: Cannot open calbration file %s, exiting\n", fname);
		return;
	}

	int s = 0;
	for (int i = 0; i < nc_; i++) {
		int num = fscanf_s(fp, "%lf %lf %lf %lf %lf %lf %lf %lf %lf", \
			& ical[9 * i + 0], &ical[9 * i + 3], &ical[9 * i + 6], \
			& ical[9 * i + 1], &ical[9 * i + 4], &ical[9 * i + 7], \
			& ical[9 * i + 2], &ical[9 * i + 5], &ical[9 * i + 8]);
		Intrinsic intr;
		intr.fx = ical[9 * i + 0];
		intr.fy = ical[9 * i + 4];
		intr.cx = ical[9 * i + 6];
		intr.cy = ical[9 * i + 7];
		intrs.push_back(intr);
		s += num;
	}
	if (s != 9 * nc_)
	{
		fprintf(stderr, "BA error: Format of Calibration file is wrong");
		return;
	}

	fclose(fp);
	fp = NULL;
}

void IBA::ba_updateKR(double* KR, double* KdA, double* KdB, double* KdG, double* K, double* p)
{
	if (KdA == nullptr || KdB == nullptr || KdG == nullptr)
	{
		if (!m_bFocal)
		{
			for (int i = 0; i < m_ncams; i++)
			{
				const double* ptAngle = p + i * 6;
				double* pKR = KR + i * 9;
				const double* pK = K + (m_V[i] - 1) * 9;

				const double ey = ptAngle[0];
				const double ex = ptAngle[1];
				const double ez = ptAngle[2];
				const double c1 = cos(ey);
				const double c2 = cos(ex);
				const double c3 = cos(ez);
				const double s1 = sin(ey);
				const double s2 = sin(ex);
				const double s3 = sin(ez);

				double matR[9];
				matR[0] = c1*c3-s1*s2*s3;     matR[1] = c2*s3;     matR[2] = s1*c3+c1*s2*s3;
				matR[3] = -c1*s3-s1*s2*c3;    matR[4] = c2*c3;     matR[5] = -s1*s3+c1*s2*c3;
				matR[6] = -s1*c2;             matR[7] = -s2;       matR[8] = c1*c2;

				pKR[0] = pK[0] * matR[0] + pK[3] * matR[3] + pK[6] * matR[6];
				pKR[1] = pK[0] * matR[1] + pK[3] * matR[4] + pK[6] * matR[7];
				pKR[2] = pK[0] * matR[2] + pK[3] * matR[5] + pK[6] * matR[8];
				pKR[3] = pK[1] * matR[0] + pK[4] * matR[3] + pK[7] * matR[6];
				pKR[4] = pK[1] * matR[1] + pK[4] * matR[4] + pK[7] * matR[7];
				pKR[5] = pK[1] * matR[2] + pK[4] * matR[5] + pK[7] * matR[8];
				pKR[6] = pK[2] * matR[0] + pK[5] * matR[3] + pK[8] * matR[6];
				pKR[7] = pK[2] * matR[1] + pK[5] * matR[4] + pK[8] * matR[7];
				pKR[8] = pK[2] * matR[2] + pK[5] * matR[5] + pK[8] * matR[8];
			}
		}
		else
		{
			double localK[9];
			memset(localK, 0, 9 * sizeof(double));
			localK[8] = 1;
			for (int i = 0; i < m_ncams; i++)
			{
				const double* ptAngle = p + i * 6;
				double* pKR = KR + i * 9;

				const double c0 = cos(ptAngle[0]);
				const double s0 = sin(ptAngle[0]);
				const double c1 = cos(ptAngle[1]);
				const double s1 = sin(ptAngle[1]);
				const double c2 = cos(ptAngle[2]);
				const double s2 = sin(ptAngle[2]);

				double matR[9];
				matR[0] = c1 * c0;
				matR[1] = c1 * s0;
				matR[2] = -s1;
				matR[3] = s2 * s1 * c0 - c2 * s0;
				matR[4] = s2 * s1 * s0 + c2 * c0;
				matR[5] = s2 * c1;
				matR[6] = c2 * s1 * c0 + s2 * s0;
				matR[7] = c2 * s1 * s0 - s2 * c0;
				matR[8] = c2 * c1;

				localK[0] = m_K[i * 3];
				localK[4] = m_K[i * 3];

				pKR[0] = localK[0] * matR[0] + localK[3] * matR[3] + localK[6] * matR[6];
				pKR[1] = localK[0] * matR[1] + localK[3] * matR[4] + localK[6] * matR[7];
				pKR[2] = localK[0] * matR[2] + localK[3] * matR[5] + localK[6] * matR[8];
				pKR[3] = localK[1] * matR[0] + localK[4] * matR[3] + localK[7] * matR[6];
				pKR[4] = localK[1] * matR[1] + localK[4] * matR[4] + localK[7] * matR[7];
				pKR[5] = localK[1] * matR[2] + localK[4] * matR[5] + localK[7] * matR[8];
				pKR[6] = localK[2] * matR[0] + localK[5] * matR[3] + localK[8] * matR[6];
				pKR[7] = localK[2] * matR[1] + localK[5] * matR[4] + localK[8] * matR[7];
				pKR[8] = localK[2] * matR[2] + localK[5] * matR[5] + localK[8] * matR[8];
			}
		}
		return;
	}

	if (!m_bFocal)
	{
		int i = 0;
		double* ptAngle;
		double* pKR, * pKdA, * pKdB, * pKdG, * pK;
		double matR[9];
		double matRG[9], matRB[9], matRA[9];
		double matDRG[9], matDRB[9], matDRA[9];
		double tmp1[9], tmp2[9];
		//for (i = 0; i < m_ncams; i++) {
		//	printf("%d\n", m_V[i]);
		//}
		for (i = 0; i < m_ncams; i++)
		{
			ptAngle = p + i * 6;
			/*phi omega kappa系统*/
			//matR=matRG*matRB*matRA
			//ptAngle=[kappa,phi,omega]
			double ey = ptAngle[0];
			double ex = ptAngle[1];
			double ez = ptAngle[2];
			double c1 = cos(ey);   double c2 = cos(ex);   double c3 = cos(ez);
			double s1 = sin(ey);   double s2 = sin(ex);   double s3 = sin(ez);
			matR[0]=c1*c3-s1*s2*s3;     matR[1]=c2*s3;     matR[2]=s1*c3+c1*s2*s3;
			matR[3]=-c1*s3-s1*s2*c3;    matR[4]=c2*c3;     matR[5]=-s1*s3+c1*s2*c3;
			matR[6]=-s1*c2;             matR[7]=-s2;       matR[8]=c1*c2;
			// matR[0] = cos(ptAngle[1]) * cos(ptAngle[0]);
			// matR[1] = cos(ptAngle[1]) * sin(ptAngle[0]);
			// matR[2] = -sin(ptAngle[1]);
			// matR[3] = sin(ptAngle[2]) * sin(ptAngle[1]) * cos(ptAngle[0]) - cos(ptAngle[2]) * sin(ptAngle[0]);
			// matR[4] = sin(ptAngle[2]) * sin(ptAngle[1]) * sin(ptAngle[0]) + cos(ptAngle[2]) * cos(ptAngle[0]);
			// matR[5] = sin(ptAngle[2]) * cos(ptAngle[1]);
			// matR[6] = cos(ptAngle[2]) * sin(ptAngle[1]) * cos(ptAngle[0]) + sin(ptAngle[2]) * sin(ptAngle[0]);
			// matR[7] = cos(ptAngle[2]) * sin(ptAngle[1]) * sin(ptAngle[0]) - sin(ptAngle[2]) * cos(ptAngle[0]);
			// matR[8] = cos(ptAngle[2]) * cos(ptAngle[1]);

			matRG[0] = 1;		matRG[1] = 0;				matRG[2] = 0;
			matRG[3] = 0;		matRG[4] = cos(ptAngle[2]);	matRG[5] = sin(ptAngle[2]);
			matRG[6] = 0;		matRG[7] = -sin(ptAngle[2]);	matRG[8] = cos(ptAngle[2]);

			matRB[0] = cos(ptAngle[1]);		matRB[1] = 0;		matRB[2] = -sin(ptAngle[1]);
			matRB[3] = 0;					matRB[4] = 1;		matRB[5] = 0;
			matRB[6] = sin(ptAngle[1]);		matRB[7] = 0;		matRB[8] = cos(ptAngle[1]);

			matRA[0] = cos(ptAngle[0]);		matRA[1] = sin(ptAngle[0]);			matRA[2] = 0;
			matRA[3] = -sin(ptAngle[0]);	matRA[4] = cos(ptAngle[0]);			matRA[5] = 0;
			matRA[6] = 0;					matRA[7] = 0;						matRA[8] = 1;

			matDRG[0] = 0;		matDRG[1] = 0;			matDRG[2] = 0;
			matDRG[3] = 0;		matDRG[4] = -sin(ptAngle[2]);	matDRG[5] = cos(ptAngle[2]);
			matDRG[6] = 0;		matDRG[7] = -cos(ptAngle[2]);	matDRG[8] = -sin(ptAngle[2]);

			matDRB[0] = -sin(ptAngle[1]);		matDRB[1] = 0;		matDRB[2] = -cos(ptAngle[1]);
			matDRB[3] = 0;						matDRB[4] = 0;		matDRB[5] = 0;
			matDRB[6] = cos(ptAngle[1]);		matDRB[7] = 0;		matDRB[8] = -sin(ptAngle[1]);

			matDRA[0] = -sin(ptAngle[0]);		matDRA[1] = cos(ptAngle[0]);		matDRA[2] = 0;
			matDRA[3] = -cos(ptAngle[0]);		matDRA[4] = -sin(ptAngle[0]);		matDRA[5] = 0;
			matDRA[6] = 0;						matDRA[7] = 0;						matDRA[8] = 0;

			//pKR=KR*matR
			pKR = KR + i * 9;
			pK = K + (m_V[i] - 1) * 9;
			//printf("%d\n", m_V[i]);
			//printf("%f %f %f\n", pK[0], pK[3], pK[6]);
			//printf("%f %f %f\n", pK[1], pK[4], pK[7]);
			//printf("%f %f %f\n", pK[2], pK[5], pK[8]);
			pKR[0] = pK[0] * matR[0] + pK[3] * matR[3] + pK[6] * matR[6];
			pKR[1] = pK[0] * matR[1] + pK[3] * matR[4] + pK[6] * matR[7];
			pKR[2] = pK[0] * matR[2] + pK[3] * matR[5] + pK[6] * matR[8];
			pKR[3] = pK[1] * matR[0] + pK[4] * matR[3] + pK[7] * matR[6];
			pKR[4] = pK[1] * matR[1] + pK[4] * matR[4] + pK[7] * matR[7];
			pKR[5] = pK[1] * matR[2] + pK[4] * matR[5] + pK[7] * matR[8];
			pKR[6] = pK[2] * matR[0] + pK[5] * matR[3] + pK[8] * matR[6];
			pKR[7] = pK[2] * matR[1] + pK[5] * matR[4] + pK[8] * matR[7];
			pKR[8] = pK[2] * matR[2] + pK[5] * matR[5] + pK[8] * matR[8];

			pKdG = KdG + i * 9;
			tmp1[0] = pK[0] * matDRG[0] + pK[3] * matDRG[3] + pK[6] * matDRG[6];
			tmp1[1] = pK[1] * matDRG[0] + pK[4] * matDRG[3] + pK[7] * matDRG[6];
			tmp1[2] = pK[2] * matDRG[0] + pK[5] * matDRG[3] + pK[8] * matDRG[6];
			tmp1[3] = pK[0] * matDRG[1] + pK[3] * matDRG[4] + pK[6] * matDRG[7];
			tmp1[4] = pK[1] * matDRG[1] + pK[4] * matDRG[4] + pK[7] * matDRG[7];
			tmp1[5] = pK[2] * matDRG[1] + pK[5] * matDRG[4] + pK[8] * matDRG[7];
			tmp1[6] = pK[0] * matDRG[2] + pK[3] * matDRG[5] + pK[6] * matDRG[8];
			tmp1[7] = pK[1] * matDRG[2] + pK[4] * matDRG[5] + pK[7] * matDRG[8];
			tmp1[8] = pK[2] * matDRG[2] + pK[5] * matDRG[5] + pK[8] * matDRG[8];

			tmp2[0] = tmp1[0] * matRB[0] + tmp1[3] * matRB[3] + tmp1[6] * matRB[6];
			tmp2[1] = tmp1[1] * matRB[0] + tmp1[4] * matRB[3] + tmp1[7] * matRB[6];
			tmp2[2] = tmp1[2] * matRB[0] + tmp1[5] * matRB[3] + tmp1[8] * matRB[6];
			tmp2[3] = tmp1[0] * matRB[1] + tmp1[3] * matRB[4] + tmp1[6] * matRB[7];
			tmp2[4] = tmp1[1] * matRB[1] + tmp1[4] * matRB[4] + tmp1[7] * matRB[7];
			tmp2[5] = tmp1[2] * matRB[1] + tmp1[5] * matRB[4] + tmp1[8] * matRB[7];
			tmp2[6] = tmp1[0] * matRB[2] + tmp1[3] * matRB[5] + tmp1[6] * matRB[8];
			tmp2[7] = tmp1[1] * matRB[2] + tmp1[4] * matRB[5] + tmp1[7] * matRB[8];
			tmp2[8] = tmp1[2] * matRB[2] + tmp1[5] * matRB[5] + tmp1[8] * matRB[8];

			pKdG[0] = tmp2[0] * matRA[0] + tmp2[3] * matRA[3] + tmp2[6] * matRA[6];
			pKdG[3] = tmp2[1] * matRA[0] + tmp2[4] * matRA[3] + tmp2[7] * matRA[6];
			pKdG[6] = tmp2[2] * matRA[0] + tmp2[5] * matRA[3] + tmp2[8] * matRA[6];
			pKdG[1] = tmp2[0] * matRA[1] + tmp2[3] * matRA[4] + tmp2[6] * matRA[7];
			pKdG[4] = tmp2[1] * matRA[1] + tmp2[4] * matRA[4] + tmp2[7] * matRA[7];
			pKdG[7] = tmp2[2] * matRA[1] + tmp2[5] * matRA[4] + tmp2[8] * matRA[7];
			pKdG[2] = tmp2[0] * matRA[2] + tmp2[3] * matRA[5] + tmp2[6] * matRA[8];
			pKdG[5] = tmp2[1] * matRA[2] + tmp2[4] * matRA[5] + tmp2[7] * matRA[8];
			pKdG[8] = tmp2[2] * matRA[2] + tmp2[5] * matRA[5] + tmp2[8] * matRA[8];

			pKdB = KdB + i * 9;
			tmp1[0] = pK[0] * matRG[0] + pK[3] * matRG[3] + pK[6] * matRG[6];
			tmp1[1] = pK[1] * matRG[0] + pK[4] * matRG[3] + pK[7] * matRG[6];
			tmp1[2] = pK[2] * matRG[0] + pK[5] * matRG[3] + pK[8] * matRG[6];
			tmp1[3] = pK[0] * matRG[1] + pK[3] * matRG[4] + pK[6] * matRG[7];
			tmp1[4] = pK[1] * matRG[1] + pK[4] * matRG[4] + pK[7] * matRG[7];
			tmp1[5] = pK[2] * matRG[1] + pK[5] * matRG[4] + pK[8] * matRG[7];
			tmp1[6] = pK[0] * matRG[2] + pK[3] * matRG[5] + pK[6] * matRG[8];
			tmp1[7] = pK[1] * matRG[2] + pK[4] * matRG[5] + pK[7] * matRG[8];
			tmp1[8] = pK[2] * matRG[2] + pK[5] * matRG[5] + pK[8] * matRG[8];

			tmp2[0] = tmp1[0] * matDRB[0] + tmp1[3] * matDRB[3] + tmp1[6] * matDRB[6];
			tmp2[1] = tmp1[1] * matDRB[0] + tmp1[4] * matDRB[3] + tmp1[7] * matDRB[6];
			tmp2[2] = tmp1[2] * matDRB[0] + tmp1[5] * matDRB[3] + tmp1[8] * matDRB[6];
			tmp2[3] = tmp1[0] * matDRB[1] + tmp1[3] * matDRB[4] + tmp1[6] * matDRB[7];
			tmp2[4] = tmp1[1] * matDRB[1] + tmp1[4] * matDRB[4] + tmp1[7] * matDRB[7];
			tmp2[5] = tmp1[2] * matDRB[1] + tmp1[5] * matDRB[4] + tmp1[8] * matDRB[7];
			tmp2[6] = tmp1[0] * matDRB[2] + tmp1[3] * matDRB[5] + tmp1[6] * matDRB[8];
			tmp2[7] = tmp1[1] * matDRB[2] + tmp1[4] * matDRB[5] + tmp1[7] * matDRB[8];
			tmp2[8] = tmp1[2] * matDRB[2] + tmp1[5] * matDRB[5] + tmp1[8] * matDRB[8];

			pKdB[0] = tmp2[0] * matRA[0] + tmp2[3] * matRA[3] + tmp2[6] * matRA[6];
			pKdB[3] = tmp2[1] * matRA[0] + tmp2[4] * matRA[3] + tmp2[7] * matRA[6];
			pKdB[6] = tmp2[2] * matRA[0] + tmp2[5] * matRA[3] + tmp2[8] * matRA[6];
			pKdB[1] = tmp2[0] * matRA[1] + tmp2[3] * matRA[4] + tmp2[6] * matRA[7];
			pKdB[4] = tmp2[1] * matRA[1] + tmp2[4] * matRA[4] + tmp2[7] * matRA[7];
			pKdB[7] = tmp2[2] * matRA[1] + tmp2[5] * matRA[4] + tmp2[8] * matRA[7];
			pKdB[2] = tmp2[0] * matRA[2] + tmp2[3] * matRA[5] + tmp2[6] * matRA[8];
			pKdB[5] = tmp2[1] * matRA[2] + tmp2[4] * matRA[5] + tmp2[7] * matRA[8];
			pKdB[8] = tmp2[2] * matRA[2] + tmp2[5] * matRA[5] + tmp2[8] * matRA[8];

			pKdA = KdA + i * 9;
			tmp2[0] = tmp1[0] * matRB[0] + tmp1[3] * matRB[3] + tmp1[6] * matRB[6];
			tmp2[1] = tmp1[1] * matRB[0] + tmp1[4] * matRB[3] + tmp1[7] * matRB[6];
			tmp2[2] = tmp1[2] * matRB[0] + tmp1[5] * matRB[3] + tmp1[8] * matRB[6];
			tmp2[3] = tmp1[0] * matRB[1] + tmp1[3] * matRB[4] + tmp1[6] * matRB[7];
			tmp2[4] = tmp1[1] * matRB[1] + tmp1[4] * matRB[4] + tmp1[7] * matRB[7];
			tmp2[5] = tmp1[2] * matRB[1] + tmp1[5] * matRB[4] + tmp1[8] * matRB[7];
			tmp2[6] = tmp1[0] * matRB[2] + tmp1[3] * matRB[5] + tmp1[6] * matRB[8];
			tmp2[7] = tmp1[1] * matRB[2] + tmp1[4] * matRB[5] + tmp1[7] * matRB[8];
			tmp2[8] = tmp1[2] * matRB[2] + tmp1[5] * matRB[5] + tmp1[8] * matRB[8];

			pKdA[0] = tmp2[0] * matDRA[0] + tmp2[3] * matDRA[3] + tmp2[6] * matDRA[6];
			pKdA[3] = tmp2[1] * matDRA[0] + tmp2[4] * matDRA[3] + tmp2[7] * matDRA[6];
			pKdA[6] = tmp2[2] * matDRA[0] + tmp2[5] * matDRA[3] + tmp2[8] * matDRA[6];
			pKdA[1] = tmp2[0] * matDRA[1] + tmp2[3] * matDRA[4] + tmp2[6] * matDRA[7];
			pKdA[4] = tmp2[1] * matDRA[1] + tmp2[4] * matDRA[4] + tmp2[7] * matDRA[7];
			pKdA[7] = tmp2[2] * matDRA[1] + tmp2[5] * matDRA[4] + tmp2[8] * matDRA[7];
			pKdA[2] = tmp2[0] * matDRA[2] + tmp2[3] * matDRA[5] + tmp2[6] * matDRA[8];
			pKdA[5] = tmp2[1] * matDRA[2] + tmp2[4] * matDRA[5] + tmp2[7] * matDRA[8];
			pKdA[8] = tmp2[2] * matDRA[2] + tmp2[5] * matDRA[5] + tmp2[8] * matDRA[8];

		}
	}
	else
	{
		int i = 0;
		double* ptAngle;
		double* pKR, * pKdA, * pKdB, * pKdG;
		double matR[9];
		double matRG[9], matRB[9], matRA[9];
		double matDRG[9], matDRB[9], matDRA[9];
		double tmp1[9], tmp2[9];
		double K[9];
		memset(K, 0, 9 * sizeof(double));
		K[8] = 1;
		for (i = 0; i < m_ncams; i++)
		{
			ptAngle = p + i * 6;

			matR[0] = cos(ptAngle[1]) * cos(ptAngle[0]);
			matR[1] = cos(ptAngle[1]) * sin(ptAngle[0]);
			matR[2] = -sin(ptAngle[1]);
			matR[3] = sin(ptAngle[2]) * sin(ptAngle[1]) * cos(ptAngle[0]) - cos(ptAngle[2]) * sin(ptAngle[0]);
			matR[4] = sin(ptAngle[2]) * sin(ptAngle[1]) * sin(ptAngle[0]) + cos(ptAngle[2]) * cos(ptAngle[0]);
			matR[5] = sin(ptAngle[2]) * cos(ptAngle[1]);
			matR[6] = cos(ptAngle[2]) * sin(ptAngle[1]) * cos(ptAngle[0]) + sin(ptAngle[2]) * sin(ptAngle[0]);
			matR[7] = cos(ptAngle[2]) * sin(ptAngle[1]) * sin(ptAngle[0]) - sin(ptAngle[2]) * cos(ptAngle[0]);
			matR[8] = cos(ptAngle[2]) * cos(ptAngle[1]);

			matRG[0] = 1;		matRG[1] = 0;				matRG[2] = 0;
			matRG[3] = 0;		matRG[4] = cos(ptAngle[2]);	matRG[5] = sin(ptAngle[2]);
			matRG[6] = 0;		matRG[7] = -sin(ptAngle[2]);	matRG[8] = cos(ptAngle[2]);
			matRB[0] = cos(ptAngle[1]);		matRB[1] = 0;		matRB[2] = -sin(ptAngle[1]);
			matRB[3] = 0;					matRB[4] = 1;		matRB[5] = 0;
			matRB[6] = sin(ptAngle[1]);		matRB[7] = 0;		matRB[8] = cos(ptAngle[1]);
			matRA[0] = cos(ptAngle[0]);		matRA[1] = sin(ptAngle[0]);			matRA[2] = 0;
			matRA[3] = -sin(ptAngle[0]);	matRA[4] = cos(ptAngle[0]);			matRA[5] = 0;
			matRA[6] = 0;					matRA[7] = 0;						matRA[8] = 1;
			matDRG[0] = 0;		matDRG[1] = 0;			matDRG[2] = 0;
			matDRG[3] = 0;		matDRG[4] = -sin(ptAngle[2]);	matDRG[5] = cos(ptAngle[2]);
			matDRG[6] = 0;		matDRG[7] = -cos(ptAngle[2]);	matDRG[8] = -sin(ptAngle[2]);
			matDRB[0] = -sin(ptAngle[1]);		matDRB[1] = 0;		matDRB[2] = -cos(ptAngle[1]);
			matDRB[3] = 0;						matDRB[4] = 0;		matDRB[5] = 0;
			matDRB[6] = cos(ptAngle[1]);		matDRB[7] = 0;		matDRB[8] = -sin(ptAngle[1]);
			matDRA[0] = -sin(ptAngle[0]);		matDRA[1] = cos(ptAngle[0]);		matDRA[2] = 0;
			matDRA[3] = -cos(ptAngle[0]);		matDRA[4] = -sin(ptAngle[0]);		matDRA[5] = 0;
			matDRA[6] = 0;						matDRA[7] = 0;						matDRA[8] = 0;

			//KR

			K[0] = m_K[i * 3];
			K[4] = m_K[i * 3];

			pKR = KR + i * 9;
			pKR[0] = K[0] * matR[0] + K[3] * matR[3] + K[6] * matR[6];
			pKR[1] = K[0] * matR[1] + K[3] * matR[4] + K[6] * matR[7];
			pKR[2] = K[0] * matR[2] + K[3] * matR[5] + K[6] * matR[8];
			pKR[3] = K[1] * matR[0] + K[4] * matR[3] + K[7] * matR[6];
			pKR[4] = K[1] * matR[1] + K[4] * matR[4] + K[7] * matR[7];
			pKR[5] = K[1] * matR[2] + K[4] * matR[5] + K[7] * matR[8];
			pKR[6] = K[2] * matR[0] + K[5] * matR[3] + K[8] * matR[6];
			pKR[7] = K[2] * matR[1] + K[5] * matR[4] + K[8] * matR[7];
			pKR[8] = K[2] * matR[2] + K[5] * matR[5] + K[8] * matR[8];

			//KdG
			pKdG = KdG + i * 9;
			tmp1[0] = K[0] * matDRG[0] + K[3] * matDRG[3] + K[6] * matDRG[6];
			tmp1[1] = K[1] * matDRG[0] + K[4] * matDRG[3] + K[7] * matDRG[6];
			tmp1[2] = K[2] * matDRG[0] + K[5] * matDRG[3] + K[8] * matDRG[6];
			tmp1[3] = K[0] * matDRG[1] + K[3] * matDRG[4] + K[6] * matDRG[7];
			tmp1[4] = K[1] * matDRG[1] + K[4] * matDRG[4] + K[7] * matDRG[7];
			tmp1[5] = K[2] * matDRG[1] + K[5] * matDRG[4] + K[8] * matDRG[7];
			tmp1[6] = K[0] * matDRG[2] + K[3] * matDRG[5] + K[6] * matDRG[8];
			tmp1[7] = K[1] * matDRG[2] + K[4] * matDRG[5] + K[7] * matDRG[8];
			tmp1[8] = K[2] * matDRG[2] + K[5] * matDRG[5] + K[8] * matDRG[8];

			tmp2[0] = tmp1[0] * matRB[0] + tmp1[3] * matRB[3] + tmp1[6] * matRB[6];
			tmp2[1] = tmp1[1] * matRB[0] + tmp1[4] * matRB[3] + tmp1[7] * matRB[6];
			tmp2[2] = tmp1[2] * matRB[0] + tmp1[5] * matRB[3] + tmp1[8] * matRB[6];
			tmp2[3] = tmp1[0] * matRB[1] + tmp1[3] * matRB[4] + tmp1[6] * matRB[7];
			tmp2[4] = tmp1[1] * matRB[1] + tmp1[4] * matRB[4] + tmp1[7] * matRB[7];
			tmp2[5] = tmp1[2] * matRB[1] + tmp1[5] * matRB[4] + tmp1[8] * matRB[7];
			tmp2[6] = tmp1[0] * matRB[2] + tmp1[3] * matRB[5] + tmp1[6] * matRB[8];
			tmp2[7] = tmp1[1] * matRB[2] + tmp1[4] * matRB[5] + tmp1[7] * matRB[8];
			tmp2[8] = tmp1[2] * matRB[2] + tmp1[5] * matRB[5] + tmp1[8] * matRB[8];

			pKdG[0] = tmp2[0] * matRA[0] + tmp2[3] * matRA[3] + tmp2[6] * matRA[6];
			pKdG[3] = tmp2[1] * matRA[0] + tmp2[4] * matRA[3] + tmp2[7] * matRA[6];
			pKdG[6] = tmp2[2] * matRA[0] + tmp2[5] * matRA[3] + tmp2[8] * matRA[6];
			pKdG[1] = tmp2[0] * matRA[1] + tmp2[3] * matRA[4] + tmp2[6] * matRA[7];
			pKdG[4] = tmp2[1] * matRA[1] + tmp2[4] * matRA[4] + tmp2[7] * matRA[7];
			pKdG[7] = tmp2[2] * matRA[1] + tmp2[5] * matRA[4] + tmp2[8] * matRA[7];
			pKdG[2] = tmp2[0] * matRA[2] + tmp2[3] * matRA[5] + tmp2[6] * matRA[8];
			pKdG[5] = tmp2[1] * matRA[2] + tmp2[4] * matRA[5] + tmp2[7] * matRA[8];
			pKdG[8] = tmp2[2] * matRA[2] + tmp2[5] * matRA[5] + tmp2[8] * matRA[8];

			//KdB
			pKdB = KdB + i * 9;
			tmp1[0] = K[0] * matRG[0] + K[3] * matRG[3] + K[6] * matRG[6];
			tmp1[1] = K[1] * matRG[0] + K[4] * matRG[3] + K[7] * matRG[6];
			tmp1[2] = K[2] * matRG[0] + K[5] * matRG[3] + K[8] * matRG[6];
			tmp1[3] = K[0] * matRG[1] + K[3] * matRG[4] + K[6] * matRG[7];
			tmp1[4] = K[1] * matRG[1] + K[4] * matRG[4] + K[7] * matRG[7];
			tmp1[5] = K[2] * matRG[1] + K[5] * matRG[4] + K[8] * matRG[7];
			tmp1[6] = K[0] * matRG[2] + K[3] * matRG[5] + K[6] * matRG[8];
			tmp1[7] = K[1] * matRG[2] + K[4] * matRG[5] + K[7] * matRG[8];
			tmp1[8] = K[2] * matRG[2] + K[5] * matRG[5] + K[8] * matRG[8];

			tmp2[0] = tmp1[0] * matDRB[0] + tmp1[3] * matDRB[3] + tmp1[6] * matDRB[6];
			tmp2[1] = tmp1[1] * matDRB[0] + tmp1[4] * matDRB[3] + tmp1[7] * matDRB[6];
			tmp2[2] = tmp1[2] * matDRB[0] + tmp1[5] * matDRB[3] + tmp1[8] * matDRB[6];
			tmp2[3] = tmp1[0] * matDRB[1] + tmp1[3] * matDRB[4] + tmp1[6] * matDRB[7];
			tmp2[4] = tmp1[1] * matDRB[1] + tmp1[4] * matDRB[4] + tmp1[7] * matDRB[7];
			tmp2[5] = tmp1[2] * matDRB[1] + tmp1[5] * matDRB[4] + tmp1[8] * matDRB[7];
			tmp2[6] = tmp1[0] * matDRB[2] + tmp1[3] * matDRB[5] + tmp1[6] * matDRB[8];
			tmp2[7] = tmp1[1] * matDRB[2] + tmp1[4] * matDRB[5] + tmp1[7] * matDRB[8];
			tmp2[8] = tmp1[2] * matDRB[2] + tmp1[5] * matDRB[5] + tmp1[8] * matDRB[8];

			pKdB[0] = tmp2[0] * matRA[0] + tmp2[3] * matRA[3] + tmp2[6] * matRA[6];
			pKdB[3] = tmp2[1] * matRA[0] + tmp2[4] * matRA[3] + tmp2[7] * matRA[6];
			pKdB[6] = tmp2[2] * matRA[0] + tmp2[5] * matRA[3] + tmp2[8] * matRA[6];
			pKdB[1] = tmp2[0] * matRA[1] + tmp2[3] * matRA[4] + tmp2[6] * matRA[7];
			pKdB[4] = tmp2[1] * matRA[1] + tmp2[4] * matRA[4] + tmp2[7] * matRA[7];
			pKdB[7] = tmp2[2] * matRA[1] + tmp2[5] * matRA[4] + tmp2[8] * matRA[7];
			pKdB[2] = tmp2[0] * matRA[2] + tmp2[3] * matRA[5] + tmp2[6] * matRA[8];
			pKdB[5] = tmp2[1] * matRA[2] + tmp2[4] * matRA[5] + tmp2[7] * matRA[8];
			pKdB[8] = tmp2[2] * matRA[2] + tmp2[5] * matRA[5] + tmp2[8] * matRA[8];

			//KdA
			pKdA = KdA + i * 9;
			tmp2[0] = tmp1[0] * matRB[0] + tmp1[3] * matRB[3] + tmp1[6] * matRB[6];
			tmp2[1] = tmp1[1] * matRB[0] + tmp1[4] * matRB[3] + tmp1[7] * matRB[6];
			tmp2[2] = tmp1[2] * matRB[0] + tmp1[5] * matRB[3] + tmp1[8] * matRB[6];
			tmp2[3] = tmp1[0] * matRB[1] + tmp1[3] * matRB[4] + tmp1[6] * matRB[7];
			tmp2[4] = tmp1[1] * matRB[1] + tmp1[4] * matRB[4] + tmp1[7] * matRB[7];
			tmp2[5] = tmp1[2] * matRB[1] + tmp1[5] * matRB[4] + tmp1[8] * matRB[7];
			tmp2[6] = tmp1[0] * matRB[2] + tmp1[3] * matRB[5] + tmp1[6] * matRB[8];
			tmp2[7] = tmp1[1] * matRB[2] + tmp1[4] * matRB[5] + tmp1[7] * matRB[8];
			tmp2[8] = tmp1[2] * matRB[2] + tmp1[5] * matRB[5] + tmp1[8] * matRB[8];

			pKdA[0] = tmp2[0] * matDRA[0] + tmp2[3] * matDRA[3] + tmp2[6] * matDRA[6];
			pKdA[3] = tmp2[1] * matDRA[0] + tmp2[4] * matDRA[3] + tmp2[7] * matDRA[6];
			pKdA[6] = tmp2[2] * matDRA[0] + tmp2[5] * matDRA[3] + tmp2[8] * matDRA[6];
			pKdA[1] = tmp2[0] * matDRA[1] + tmp2[3] * matDRA[4] + tmp2[6] * matDRA[7];
			pKdA[4] = tmp2[1] * matDRA[1] + tmp2[4] * matDRA[4] + tmp2[7] * matDRA[7];
			pKdA[7] = tmp2[2] * matDRA[1] + tmp2[5] * matDRA[4] + tmp2[8] * matDRA[7];
			pKdA[2] = tmp2[0] * matDRA[2] + tmp2[3] * matDRA[5] + tmp2[6] * matDRA[8];
			pKdA[5] = tmp2[1] * matDRA[2] + tmp2[4] * matDRA[5] + tmp2[7] * matDRA[8];
			pKdA[8] = tmp2[2] * matDRA[2] + tmp2[5] * matDRA[5] + tmp2[8] * matDRA[8];

		}
	}
}

void IBA::readNpointsAndNprojectionsFromProj(FILE* fp, int& n3Dpts, int& nprojs)
{
	int nfirst, lineno, npts, nframes, ch, n;
	nprojs = 0;
	n3Dpts = 0;
	npts = 0;

	/* #parameters for the first line */
	nfirst = countNDoubles(fp);

	//*n3Dpts=*nprojs=lineno=npts=0;
	while (!feof(fp))
	{
		if ((ch = fgetc(fp)) == '#')
		{ /* skip comments */
			SKIP_LINE(fp);
			++lineno;
			continue;
		}

		if (feof(fp)) break;

		ungetc(ch, fp);
		++lineno;
		//skipNDoubles(fp, pnp);
		n = readNInts(fp, &nframes, 1);
		if (n != 1)
		{
			fprintf(stderr, "readNpointsAndNprojections(): error reading input file, line %d: "
				"expecting number of frames for 3D point\n", lineno);
			exit(1);
		}

		SKIP_LINE(fp);
		nprojs += nframes;
		++npts;
	}

	n3Dpts = npts;
}

void IBA::readPointProjections(FILE* fp, double* imgpts, int* photo, int* imgptsSum, int n3Dpts, int n2Dprojs)
{
	int nframes, ch, lineno, ptno, frameno, n;
	int i;
	int nproj2D = 0;

	lineno = ptno = 0;
	while (!feof(fp))
	{
		if ((ch = fgetc(fp)) == '#')
		{ /* skip comments */
			SKIP_LINE(fp);
			lineno++;

			continue;
		}

		if (feof(fp)) break;

		ungetc(ch, fp);

		n = readNInts(fp, &nframes, 1);  /* read in number of image projections */
		if (n != 1)
		{
			fprintf(stderr, "sba_readProjectionAndInitilizeFeature(): error reading input file, line %d:\n"
				"expecting number of frames for 3D point\n", lineno);
			exit(1);
		}

		imgptsSum[ptno] = nframes;

		for (i = 0; i < nframes; ++i)
		{
			n = readNInts(fp, &frameno, 1); /* read in frame number... */

			photo[nproj2D] = frameno;

			n += readNDoubles(fp, imgpts + nproj2D * 2, 2); /* ...and image projection */

			nproj2D++;
		}
		fscanf_s(fp, "\n"); // consume trailing newline

		lineno++;
		ptno++;
	}
}
void IBA::readImagePts(const char* szProj, double** imgpts, int** photo, int** imgptsSum, int& n3Dpts, int& n2Dprojs)
{
	FILE* fpp = nullptr;
	fopen_s(&fpp, szProj, "r");
	if (fpp == nullptr) {
		fprintf(stderr, "cannot open file %s, exiting\n", szProj);
		exit(1);
	}
	readNpointsAndNprojectionsFromProj(fpp, n3Dpts, n2Dprojs);

	*imgpts = (double*)malloc(n2Dprojs * 2 * sizeof(double));
	if (*imgpts == NULL) {
		fprintf(stderr, "memory allocation for 'imgpts' failed in readInitialSBAEstimate()\n");
		exit(1);
	}

	*photo = (int*)malloc(n2Dprojs * sizeof(int));
	if (*photo == NULL)
	{
		fprintf(stderr, "memory allocation for 'struct' failed in readInitialSBAEstimate()\n");
		exit(1);
	}

	*imgptsSum = (int*)malloc(n3Dpts * sizeof(int));
	if (*imgptsSum == NULL)
	{
		fprintf(stderr, "memory allocation for 'struct' failed in readInitialSBAEstimate()\n");
		exit(1);
	}

	rewind(fpp);
	readPointProjections(fpp, *imgpts, *photo, *imgptsSum, n3Dpts, n2Dprojs);

	fclose(fpp);
}


