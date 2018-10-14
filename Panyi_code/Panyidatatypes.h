// $Id: Panyidatatypes.h,v 1.1.1.1 2016/12/30 06:03:09 zjcao Exp $
#ifndef _DATETYPES_H
#define _DATETYPES_H


#include <complex>
#include <stdint.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_odeiv.h>


/* Integer types */
typedef unsigned int UINT4;		/**< Four-byte unsigned integer. */

/* Real types */
typedef float REAL4;    /**< Single precision real floating-point number (4 bytes). */
typedef double REAL8;   /**< Double precision real floating-point number (8 bytes). */
typedef std::complex<double> COMPLEX16;
//typedef struct tagCOMPLEX16 { REAL8 re; REAL8 im; } COMPLEX16;

typedef int16_t  INT2;		/**< Two-byte signed integer */
typedef int32_t  INT4;		/**< Four-byte signed integer. */
typedef int64_t  INT8;		/**< Eight-byte signed integer; on some platforms this is equivalent to <tt>long int</tt> instead. */
typedef uint16_t UINT2;		/**< Two-byte unsigned integer. */
typedef uint64_t UINT8;		/**< Eight-byte unsigned integer; on some platforms this is equivalent to <tt>unsigned long int</tt> instead. */



//#define LAL_INT8_C INT64_C
#define LAL_INT8_C long int

#define I (XLALCOMPLEX16Rect(0.0,1.0))
#define XLAL_BILLION_REAL8 1e9
#define XLAL_BILLION_INT8 LAL_INT8_C( 1000000000 )
#define EOB_RD_EFOLDS 10.0
#define LAL_MTSUN_SI  4.92549095e-6   /**< Geometrized solar mass, s */
#define LAL_PI_2      1.5707963267948966192313216916397514  /**< pi/2 */
#define LAL_1_PI      0.3183098861837906715377675267450287  /**< 1/pi */
#define LAL_MSUN_SI   1.98892e30      /**< Solar mass, kg */
#define LAL_GAMMA     0.5772156649015328606065120900824024  /**< gamma */
#define LAL_E         2.7182818284590452353602874713526625  /**< e */
#define LAL_MRSUN_SI  1.47662504e3    /**< Geometrized solar mass, m */
#define LAL_TWOPI     6.2831853071795864769252867665590058  /**< 2*pi */

#define LALFree(p) free(p)


typedef struct tagGSParams {
    int ampO;                 /**< twice PN order of the amplitude */
    REAL8 phiRef;             /**< phase at fRef */
    REAL8 deltaT;             /**< sampling interval */
    REAL8 m1;                 /**< mass of companion 1 */
    REAL8 m2;                 /**< mass of companion 2 */
    REAL8 f_min;              /**< start frequency */
    REAL8 e0;                 /**< eccentricity at start frequency */
    REAL8 f_max;              /**< end frequency */
    REAL8 distance;           /**< distance of source */
    REAL8 inclination;        /**< inclination of L relative to line of sight */
    REAL8 s1x;                /**< (x,y,z) components of spin of m1 body */
    REAL8 s1y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s1z;                /**< dimensionless spin, Kerr bound: |s1| <= 1 */
    REAL8 s2x;                /**< (x,y,z) component ofs spin of m2 body */
    REAL8 s2y;                /**< z-axis along line of sight, L in x-z plane */
    REAL8 s2z;                /**< dimensionless spin, Kerr bound: |s2| <= 1 */
    char outname[256];        /**< file to which output should be written */
    int ampPhase;
    int verbose;
} GSParams;

typedef struct
tagLALUnit
{
  int  unitNumerator[7]; /**< Array of unit power numerators. */
  int  unitDenominatorMinusOne[7]; /**< Array of unit power denominators-minus-one. */
}
LALUnit;

/** Vector of type REAL8, see \ref ss_Vector for more details. */
typedef struct tagREAL8Vector
{
  UINT4  length; /**< Number of elements in array. */
  REAL8 *data; /**< Pointer to the data array. */
}
REAL8Vector;

typedef REAL8Vector     REAL8Sequence;	/**< See \ref ss_Sequence for documentation */

typedef struct
tagLIGOTimeGPS
{
  int gpsSeconds; /**< Seconds since 0h UTC 6 Jan 1980. */
  int gpsNanoSeconds; /**< Residual nanoseconds. */
}
LIGOTimeGPS;


/** Time series of REAL8 data, see \ref ss_TimeSeries for more details. */
typedef struct
tagREAL8TimeSeries
{
  char           name[64]; /**< The name of the time series. */
  LIGOTimeGPS    epoch; /**< The start time of the time series. */
  REAL8          deltaT; /**< The time step between samples of the time series in seconds. */
  REAL8          f0; /**< The heterodyning frequency, in Hertz (zero if not heterodyned). */
  LALUnit        sampleUnits; /**< The physical units of the quantity being sampled. */
  //REAL8Sequence *data; /**< The sequence of sampled data. */
  REAL8Vector *data; /**< The sequence of sampled data. */
}
REAL8TimeSeries;

typedef struct tagCOMPLEX16Vector
{
  UINT4      length; /**< Number of elements in array. */
  COMPLEX16 *data; /**< Pointer to the data array. */
}
COMPLEX16Vector;

typedef struct
tagEOBNonQCCoeffs
{
  REAL8 a1;
  REAL8 a2;
  REAL8 a3;
  REAL8 a3S;
  REAL8 a4;
  REAL8 a5;
  REAL8 b1;
  REAL8 b2;
  REAL8 b3;
  REAL8 b4;
} EOBNonQCCoeffs;

typedef struct
tagark4GSLIntegrator
{
  gsl_odeiv_step    *step;
  gsl_odeiv_control *control;
  gsl_odeiv_evolve  *evolve;

  gsl_odeiv_system  *sys;

  int (* dydt) (double t, const double y[], double dydt[], void * params);
  int (* stop) (double t, const double y[], double dydt[], void * params);

  int retries;		/* retries with smaller step when derivatives encounter singularity */
  int stopontestonly;	/* stop only on test, use tend to size buffers only */

  int returncode;
} ark4GSLIntegrator;

typedef struct
tagUINT4Vector
{
  UINT4  length; /**< Number of elements in array. */
  UINT4  *data; /**< Pointer to the data array. */
}
UINT4Vector;

/** Multidimentional array of REAL8, see \ref ss_Array for more details. */
typedef struct
tagREAL8Array
{
  UINT4Vector *dimLength; /**< Vector of array dimensions. */
  REAL8       *data; /**< Pointer to the data array. */
}
REAL8Array;

typedef struct
tagEOBACoefficients
{
  REAL8 n4;
  REAL8 n5;
  REAL8 d0;
  REAL8 d1;
  REAL8 d2;
  REAL8 d3;
  REAL8 d4;
  REAL8 d5;
}
EOBACoefficients;

typedef struct
tagFacWaveformCoeffs
{
  REAL8 delta22vh3;
  REAL8 delta22vh6;
  REAL8 delta22v8;
  REAL8 delta22vh9;
  REAL8 delta22v5;

  REAL8 rho22v2;
  REAL8 rho22v3;
  REAL8 rho22v4;
  REAL8 rho22v5;
  REAL8 rho22v6;
  REAL8 rho22v6l;
  REAL8 rho22v7;
  REAL8 rho22v8;
  REAL8 rho22v8l;
  REAL8 rho22v10;
  REAL8 rho22v10l;

  REAL8 delta21vh3;
  REAL8 delta21vh6;
  REAL8 delta21vh7;
  REAL8 delta21vh9;
  REAL8 delta21v5;
  REAL8 delta21v7;

  REAL8 rho21v1;
  REAL8 rho21v2;
  REAL8 rho21v3;
  REAL8 rho21v4;
  REAL8 rho21v5;
  REAL8 rho21v6;
  REAL8 rho21v6l;
  REAL8 rho21v7;
  REAL8 rho21v7l;
  REAL8 rho21v8;
  REAL8 rho21v8l;
  REAL8 rho21v10;
  REAL8 rho21v10l;

  REAL8 f21v1;

  REAL8 delta33vh3;
  REAL8 delta33vh6;
  REAL8 delta33vh9;
  REAL8 delta33v5;
  REAL8 delta33v7;

  REAL8 rho33v2;
  REAL8 rho33v3;
  REAL8 rho33v4;
  REAL8 rho33v5;
  REAL8 rho33v6;
  REAL8 rho33v6l;
  REAL8 rho33v7;
  REAL8 rho33v8;
  REAL8 rho33v8l;

  REAL8 f33v3;

  REAL8 delta32vh3;
  REAL8 delta32vh4;
  REAL8 delta32vh6;
  REAL8 delta32vh9;

  REAL8 rho32v;
  REAL8 rho32v2;
  REAL8 rho32v3;
  REAL8 rho32v4;
  REAL8 rho32v5;
  REAL8 rho32v6;
  REAL8 rho32v6l;
  REAL8 rho32v8;
  REAL8 rho32v8l;

  REAL8 delta31vh3;
  REAL8 delta31vh6;
  REAL8 delta31vh7;
  REAL8 delta31vh9;
  REAL8 delta31v5;

  REAL8 rho31v2;
  REAL8 rho31v3;
  REAL8 rho31v4;
  REAL8 rho31v5;
  REAL8 rho31v6;
  REAL8 rho31v6l;
  REAL8 rho31v7;
  REAL8 rho31v8;
  REAL8 rho31v8l;

  REAL8 f31v3;

  REAL8 delta44vh3;
  REAL8 delta44vh6;
  REAL8 delta44v5;

  REAL8 rho44v2;
  REAL8 rho44v3;
  REAL8 rho44v4;
  REAL8 rho44v5;
  REAL8 rho44v6;
  REAL8 rho44v6l;

  REAL8 delta43vh3;
  REAL8 delta43vh4;
  REAL8 delta43vh6;

  REAL8 rho43v;
  REAL8 rho43v2;
  REAL8 rho43v4;
  REAL8 rho43v5;
  REAL8 rho43v6;
  REAL8 rho43v6l;

  REAL8 f43v;

  REAL8 delta42vh3;
  REAL8 delta42vh6;

  REAL8 rho42v2;
  REAL8 rho42v3;
  REAL8 rho42v4;
  REAL8 rho42v5;
  REAL8 rho42v6;
  REAL8 rho42v6l;

  REAL8 delta41vh3;
  REAL8 delta41vh4;
  REAL8 delta41vh6;

  REAL8 rho41v;
  REAL8 rho41v2;
  REAL8 rho41v4;
  REAL8 rho41v5;
  REAL8 rho41v6;
  REAL8 rho41v6l;

  REAL8 f41v;

  REAL8 delta55vh3;
  REAL8 delta55v5;
  REAL8 rho55v2;
  REAL8 rho55v3;
  REAL8 rho55v4;
  REAL8 rho55v5;
  REAL8 rho55v6;

  REAL8 delta54vh3;
  REAL8 delta54vh4;
  REAL8 rho54v2;
  REAL8 rho54v3;
  REAL8 rho54v4;

  REAL8 delta53vh3;
  REAL8 rho53v2;
  REAL8 rho53v3;
  REAL8 rho53v4;
  REAL8 rho53v5;

  REAL8 delta52vh3;
  REAL8 delta52vh4;
  REAL8 rho52v2;
  REAL8 rho52v3;
  REAL8 rho52v4;

  REAL8 delta51vh3;
  REAL8 rho51v2;
  REAL8 rho51v3;
  REAL8 rho51v4;
  REAL8 rho51v5;

  REAL8 delta66vh3;
  REAL8 rho66v2;
  REAL8 rho66v3;
  REAL8 rho66v4;

  REAL8 delta65vh3;
  REAL8 rho65v2;
  REAL8 rho65v3;

  REAL8 delta64vh3;
  REAL8 rho64v2;
  REAL8 rho64v3;
  REAL8 rho64v4;

  REAL8 delta63vh3;
  REAL8 rho63v2;
  REAL8 rho63v3;

  REAL8 delta62vh3;
  REAL8 rho62v2;
  REAL8 rho62v3;
  REAL8 rho62v4;

  REAL8 delta61vh3;
  REAL8 rho61v2;
  REAL8 rho61v3;

  REAL8 delta77vh3;
  REAL8 rho77v2;
  REAL8 rho77v3;

  REAL8 rho76v2;

  REAL8 delta75vh3;
  REAL8 rho75v2;
  REAL8 rho75v3;

  REAL8 rho74v2;

  REAL8 delta73vh3;
  REAL8 rho73v2;
  REAL8 rho73v3;

  REAL8 rho72v2;

  REAL8 delta71vh3;
  REAL8 rho71v2;
  REAL8 rho71v3;

  REAL8 rho88v2;
  REAL8 rho87v2;
  REAL8 rho86v2;
  REAL8 rho85v2;
  REAL8 rho84v2;
  REAL8 rho83v2;
  REAL8 rho82v2;
  REAL8 rho81v2;
}
FacWaveformCoeffs;

typedef
struct tagNewtonMultipolePrefixes
{
  COMPLEX16 values[8+1][8+1];
}
NewtonMultipolePrefixes;

typedef
struct tagEOBParams
{
  REAL8 eta;
  REAL8 omega;
  REAL8 m1;
  REAL8 m2;
  EOBACoefficients        *aCoeffs;
  FacWaveformCoeffs       *hCoeffs;
  EOBNonQCCoeffs          *nqcCoeffs;
  NewtonMultipolePrefixes *prefixes;
}
EOBParams;

typedef struct
tagSpinEOBHCoeffs
{
  double KK;
  double k0;
  double k1;
  double k2;
  double k3;
  double k4;
  double b3;
  double bb3;
}
SpinEOBHCoeffs;

typedef struct
tagSpinEOBParams
{
  EOBParams               *eobParams;
  SpinEOBHCoeffs          *seobCoeffs;
  REAL8Vector             *sigmaStar;
  REAL8Vector             *sigmaKerr;
  REAL8                   a;
  int                     alignedSpins;
  int                     tortoise;
  
  // we assume m1>m2
  REAL8Vector             *Spin1;
  REAL8Vector             *Spin2;
}
SpinEOBParams;

typedef
struct tagSEOBRootParams
{
  REAL8          values[12]; /**<< Dynamical variables, x, y, z, px, py, pz, S1x, S1y, S1z, S2x, S2y and S2z */
  SpinEOBParams *params;     /**<< Spin EOB parameters -- physical, pre-computed, etc. */
  REAL8          omega;      /**<< Orbital frequency */
}
SEOBRootParams;


/* We need to encapsulate the data for the GSL derivative function */
typedef
struct tagHcapDerivParams
{
   const REAL8   *values;
   SpinEOBParams *params;
   UINT4         varyParam;
}
HcapDerivParams;

#define XLAL_REAL8_FAIL_NAN_INT LAL_INT8_C(0x7ff80000000001a1) /**< Hexadecimal representation of <tt>REAL8</tt> NaN failure bit pattern */ 
#define XLAL_IS_REAL8_FAIL_NAN(val) XLALIsREAL8FailNaN(val) /**< Tests if <tt>val</tt> is a XLAL <tt>REAL8</tt> failure NaN. */

/* We need to encapsulate the data for calculating spherical 2nd derivatives */
typedef
struct tagHcapSphDeriv2Params
{
  const REAL8     *sphValues;
  SpinEOBParams   *params;
  UINT4           varyParam1;
  UINT4           varyParam2;
}
HcapSphDeriv2Params;


/** XLAL error numbers and return values. */
enum XLALErrorValue {
	XLAL_SUCCESS =  0, /**< Success return value (not an error number) */
	XLAL_FAILURE = -1, /**< Failure return value (not an error number) */

	/* these are standard error numbers */
	XLAL_EIO     =  5,  /**< I/O error */
	XLAL_ENOMEM  = 12,  /**< Memory allocation error */
	XLAL_EFAULT  = 14,  /**< Invalid pointer */
	XLAL_EINVAL  = 22,  /**< Invalid argument */
	XLAL_EDOM    = 33,  /**< Input domain error */
	XLAL_ERANGE  = 34,  /**< Output range error */

	/* extended error numbers start at 128 ...
	 * should be beyond normal errnos */

	/* these are common errors for XLAL functions */
	XLAL_EFAILED = 128, /**< Generic failure */
	XLAL_EBADLEN = 129, /**< Inconsistent or invalid length */
	XLAL_ESIZE   = 130, /**< Wrong size */
	XLAL_EDIMS   = 131, /**< Wrong dimensions */
	XLAL_ETYPE   = 132, /**< Wrong or unknown type */
	XLAL_ETIME   = 133, /**< Invalid time */
	XLAL_EFREQ   = 134, /**< Invalid freqency */
	XLAL_EUNIT   = 135, /**< Invalid units */
	XLAL_ENAME   = 136, /**< Wrong name */
	XLAL_EDATA   = 137, /**< Invalid data */

	/* user-defined errors */
	XLAL_EUSR0   = 200, /**< User-defined error 0 */
	XLAL_EUSR1   = 201, /**< User-defined error 1 */
	XLAL_EUSR2   = 202, /**< User-defined error 2 */
	XLAL_EUSR3   = 203, /**< User-defined error 3 */
	XLAL_EUSR4   = 204, /**< User-defined error 4 */
	XLAL_EUSR5   = 205, /**< User-defined error 5 */
	XLAL_EUSR6   = 206, /**< User-defined error 6 */
	XLAL_EUSR7   = 207, /**< User-defined error 7 */
	XLAL_EUSR8   = 208, /**< User-defined error 8 */
	XLAL_EUSR9   = 209, /**< User-defined error 9 */

	/* external or internal errors */
	XLAL_ESYS    = 254, /**< System error */
	XLAL_EERR    = 255, /**< Internal error */

	/* specific mathematical and numerical errors start at 256 */

	/* IEEE floating point errors */
	XLAL_EFPINVAL  = 256, /**< IEEE Invalid floating point operation, eg sqrt(-1), 0/0 */
	XLAL_EFPDIV0   = 257, /**< IEEE Division by zero floating point error */
	XLAL_EFPOVRFLW = 258, /**< IEEE Floating point overflow error */
	XLAL_EFPUNDFLW = 259, /**< IEEE Floating point underflow error */
	XLAL_EFPINEXCT = 260, /**< IEEE Floating point inexact error */

	/* numerical algorithm errors */
	XLAL_EMAXITER  = 261, /**< Exceeded maximum number of iterations */
	XLAL_EDIVERGE  = 262, /**< Series is diverging */
	XLAL_ESING     = 263, /**< Apparent singularity detected */
	XLAL_ETOL      = 264, /**< Failed to reach specified tolerance */
	XLAL_ELOSS     = 265, /**< Loss of accuracy */

	/* failure from within a function call: "or" error number with this */
	XLAL_EFUNC     = 1024 /**< Internal function call failed bit: "or" this with existing error number */
};

typedef struct
tagREAL8VectorSequence
{
  UINT4  length; /**< The number \a l of vectors. */
  UINT4  vectorLength; /**< The length \a n of each vector. */
  REAL8 *data; /**< Pointer to the data array.  Element \a i of vector \a j is \c data[ \a jn + \a i \c ]. */
}
REAL8VectorSequence;

enum
{
  LALUnitIndexMeter, 	/**< The meter index. */
  LALUnitIndexKiloGram, /**< The kilogram index. */
  LALUnitIndexSecond, 	/**< The second index. */
  LALUnitIndexAmpere, 	/**< The ampere index. */
  LALUnitIndexKelvin, 	/**< The kelvin index. */
  LALUnitIndexStrain, 	/**< The strain index. */
  LALUnitIndexADCCount, /**< The ADC counts index. */
  LALNumUnits 		/**< The number of units. */
};

#define LAL_PI        3.1415926535897932384626433832795029  /**< pi */

enum enumLALNameLength { LALNameLength = 64 };

#endif /* _DATE_H */


