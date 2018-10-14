
#include <complex.h>
#include <stdint.h>
#include <stdio.h>
#include "MYconstants.h"

#if __STDC_VERSION__ >= 199901L
# define _LAL_RESTRICT_ restrict
# define _LAL_INLINE_ inline
#elif defined __GNUC__
# define _LAL_RESTRICT_ __restrict__
# define _LAL_INLINE_ __inline__
#else
# define _LAL_RESTRICT_
# define _LAL_INLINE_
#endif


/* Integer types */
typedef unsigned int UINT4;        /**< Four-byte unsigned integer. */

/* Real types */
typedef float REAL4;    /**< Single precision real floating-point number (4 bytes). */
typedef double REAL8;   /**< Double precision real floating-point number (8 bytes). */
typedef double complex COMPLEX16;
//typedef struct tagCOMPLEX16 { REAL8 re; REAL8 im; } COMPLEX16;

typedef int16_t  INT2;        /**< Two-byte signed integer */
typedef int32_t  INT4;        /**< Four-byte signed integer. */
typedef int64_t  INT8;        /**< Eight-byte signed integer; on some platforms this is equivalent to <tt>long int</tt> instead. */
typedef uint16_t UINT2;        /**< Two-byte unsigned integer. */
typedef uint64_t UINT8;        /**< Eight-byte unsigned integer; on some platforms this is equivalent to <tt>unsigned long int</tt> instead. */
typedef char CHAR;        /**< One-byte signed integer, see \ref LALDatatypes for more details */
typedef unsigned char UCHAR;    /**< One-byte unsigned integer, see \ref LALDatatypes for more details */
typedef unsigned char BOOLEAN;    /**< Boolean logical type, see \ref LALDatatypes for more details */

#undef no_argument
#undef required_argument
#undef optional_argument

#define no_argument        0
#define required_argument    1
#define optional_argument    2
#define LAL_INT8_C INT64_C
#define LAL_UINT8_C UINT64_C
#define XLAL_BILLION_REAL8 1e9
#define XLAL_BILLION_INT8 LAL_INT8_C( 1000000000 )

#define LAL_UINT8_MAX   LAL_UINT8_C(18446744073709551615)
#define LAL_UINT4_MAX   LAL_UINT8_C(4294967295)
#define LAL_UINT2_MAX   LAL_UINT8_C(65535)
#define LAL_INT8_MAX    LAL_UINT8_C(9223372036854775807)
#define LAL_INT4_MAX    LAL_UINT8_C(2147483647)
#define LAL_INT2_MAX    LAL_UINT8_C(32767)

#define LALFree(p) free(p)
#define XLAL_EPOCH_UNIX_GPS 315964800

#define LALFree(p) free(p)
#define XLALFree(p) free(p)

#define XLAL_REAL8_FAIL_NAN_INT LAL_INT8_C(0x7ff80000000001a1)
#define XLAL_IS_REAL8_FAIL_NAN(val) XLALIsREAL8FailNaN(val)
#define XLAL_REAL8_FAIL_NAN ( XLALREAL8FailNaN() )
static _LAL_INLINE_ REAL8 XLALREAL8FailNaN(void)
{
    volatile const union {
        INT8 i;
        REAL8 x;
    } val = {
        XLAL_REAL8_FAIL_NAN_INT};
    return val.x;
}

static _LAL_INLINE_ int XLALIsREAL8FailNaN(REAL8 val)
{
    volatile const union {
        INT8 i;
        unsigned char s[8];
    } a = {
        XLAL_REAL8_FAIL_NAN_INT};
    volatile union {
        REAL8 x;
        unsigned char s[8];
    } b;
    size_t n;
    b.x = val;
    for (n = 0; n < sizeof(val); ++n)
        if (a.s[n] != b.s[n])
            return 0;
    return 1;
}


typedef struct tagLALStatus {
    INT4 statusCode;                            /**< A numerical code identifying the type of error, or 0 for nominal status; Negative values are reserved for certain standard error types */
    const CHAR *statusDescription;              /**< An explanatory string corresponding to the numerical status code */
    volatile const CHAR *Id;                    /**< A character string identifying the source file and version number of the function being reported on */
    const CHAR *function;                       /**< The name of the function */
    const CHAR *file;                           /**< The name of the source file containing the function code */
    INT4 line;                                  /**< The line number in the source file where the current \c statusCode was set */
    struct tagLALStatus *statusPtr;             /**< Pointer to the next node in the list; \c NULL if this function is not reporting a subroutine error */
    INT4 level;                                 /**< The nested-function level where any error was reported */
} LALStatus;

/** lalDebugLevel bit field values */
enum {
    LALERRORBIT = 0001,   /**< enable error messages */
    LALWARNINGBIT = 0002, /**< enable warning messages */
    LALINFOBIT = 0004,    /**< enable info messages */
    LALTRACEBIT = 0010,   /**< enable tracing messages */
    LALMEMDBGBIT = 0020,  /**< enable memory debugging routines */
    LALMEMPADBIT = 0040,  /**< enable memory padding */
    LALMEMTRKBIT = 0100,  /**< enable memory tracking */
    LALMEMINFOBIT = 0200  /**< enable memory info messages */
};

/** composite lalDebugLevel values */
enum {
    LALNDEBUG = 0,      /**< no debug */
    LALERROR = LALERRORBIT,             /**< enable error messages */
    LALWARNING = LALWARNINGBIT,         /**< enable warning messages */
    LALINFO = LALINFOBIT,               /**< enable info messages */
    LALTRACE = LALTRACEBIT,             /**< enable tracing messages */
    LALMSGLVL1 = LALERRORBIT,           /**< enable error messages */
    LALMSGLVL2 = LALERRORBIT | LALWARNINGBIT,   /**< enable error and warning messages */
    LALMSGLVL3 = LALERRORBIT | LALWARNINGBIT | LALINFOBIT,      /**< enable error, warning, and info messages */
    LALMEMDBG = LALMEMDBGBIT | LALMEMPADBIT | LALMEMTRKBIT,     /**< enable memory debugging tools */
    LALMEMTRACE = LALTRACEBIT | LALMEMDBG | LALMEMINFOBIT,      /**< enable memory tracing tools */
    LALALLDBG = ~LALNDEBUG      /**< enable all debugging */
};

enum XLALErrorValue {
    XLAL_SUCCESS = 0,      /**< Success return value (not an error number) */
    XLAL_FAILURE = -1,     /**< Failure return value (not an error number) */
    
    /* these are standard error numbers */
    XLAL_ENOENT = 2,        /**< No such file or directory */
    XLAL_EIO = 5,           /**< I/O error */
    XLAL_ENOMEM = 12,       /**< Memory allocation error */
    XLAL_EFAULT = 14,       /**< Invalid pointer */
    XLAL_EINVAL = 22,       /**< Invalid argument */
    XLAL_EDOM = 33,         /**< Input domain error */
    XLAL_ERANGE = 34,       /**< Output range error */
    XLAL_ENOSYS = 38,       /**< Function not implemented */
    
    /* extended error numbers start at 128 ...
     * should be beyond normal errnos */
    
    /* these are common errors for XLAL functions */
    XLAL_EFAILED = 128,     /**< Generic failure */
    XLAL_EBADLEN = 129,     /**< Inconsistent or invalid length */
    XLAL_ESIZE = 130,       /**< Wrong size */
    XLAL_EDIMS = 131,       /**< Wrong dimensions */
    XLAL_ETYPE = 132,       /**< Wrong or unknown type */
    XLAL_ETIME = 133,       /**< Invalid time */
    XLAL_EFREQ = 134,       /**< Invalid freqency */
    XLAL_EUNIT = 135,       /**< Invalid units */
    XLAL_ENAME = 136,       /**< Wrong name */
    XLAL_EDATA = 137,       /**< Invalid data */
    
    /* user-defined errors */
    XLAL_EUSR0 = 200,       /**< User-defined error 0 */
    XLAL_EUSR1 = 201,       /**< User-defined error 1 */
    XLAL_EUSR2 = 202,       /**< User-defined error 2 */
    XLAL_EUSR3 = 203,       /**< User-defined error 3 */
    XLAL_EUSR4 = 204,       /**< User-defined error 4 */
    XLAL_EUSR5 = 205,       /**< User-defined error 5 */
    XLAL_EUSR6 = 206,       /**< User-defined error 6 */
    XLAL_EUSR7 = 207,       /**< User-defined error 7 */
    XLAL_EUSR8 = 208,       /**< User-defined error 8 */
    XLAL_EUSR9 = 209,       /**< User-defined error 9 */
    
    /* external or internal errors */
    XLAL_ESYS = 254,        /**< System error */
    XLAL_EERR = 255,        /**< Internal error */
    
    /* specific mathematical and numerical errors start at 256 */
    
    /* IEEE floating point errors */
    XLAL_EFPINVAL = 256,      /**< IEEE Invalid floating point operation, eg sqrt(-1), 0/0 */
    XLAL_EFPDIV0 = 257,       /**< IEEE Division by zero floating point error */
    XLAL_EFPOVRFLW = 258,     /**< IEEE Floating point overflow error */
    XLAL_EFPUNDFLW = 259,     /**< IEEE Floating point underflow error */
    XLAL_EFPINEXCT = 260,     /**< IEEE Floating point inexact error */
    
    /* numerical algorithm errors */
    XLAL_EMAXITER = 261,      /**< Exceeded maximum number of iterations */
    XLAL_EDIVERGE = 262,      /**< Series is diverging */
    XLAL_ESING = 263,         /**< Apparent singularity detected */
    XLAL_ETOL = 264,          /**< Failed to reach specified tolerance */
    XLAL_ELOSS = 265,         /**< Loss of accuracy */
    
    /* failure from within a function call: "or" error number with this */
    XLAL_EFUNC = 1024         /**< Internal function call failed bit: "or" this with existing error number */
};


enum {
    LALUnitIndexMeter, /**< The meter index. */
    LALUnitIndexKiloGram, /**< The kilogram index. */
    LALUnitIndexSecond, /**< The second index. */
    LALUnitIndexAmpere, /**< The ampere index. */
    LALUnitIndexKelvin, /**< The kelvin index. */
    LALUnitIndexStrain, /**< The strain index. */
    LALUnitIndexADCCount, /**< The ADC counts index. */
    LALNumUnits         /**< The number of units. */
};

/** Enumeration of Detectors: follows order of DQ bit assignments */
enum {
    LAL_TAMA_300_DETECTOR    =    0,
    LAL_VIRGO_DETECTOR    =    1,
    LAL_GEO_600_DETECTOR    =    2,
    LAL_LHO_2K_DETECTOR    =    3,
    LAL_LHO_4K_DETECTOR    =    4,
    LAL_LLO_4K_DETECTOR    =    5,
    LAL_CIT_40_DETECTOR    =    6,
    LAL_ALLEGRO_DETECTOR    =    7,
    LAL_AURIGA_DETECTOR    =    8,
    LAL_EXPLORER_DETECTOR    =    9,
    LAL_NIOBE_DETECTOR    =    10,
    LAL_NAUTILUS_DETECTOR    =    11,
    LAL_ET1_DETECTOR    =    12,
    LAL_ET2_DETECTOR    =    13,
    LAL_ET3_DETECTOR    =    14,
    LAL_ET0_DETECTOR    =    15,
    LAL_KAGRA_DETECTOR    =    16,
    LAL_LIO_4K_DETECTOR =   17,
    LAL_NUM_DETECTORS    =    18
};


typedef struct tagLALUnit {
    INT2 powerOfTen; /**< Overall power-of-ten scaling is 10^\c powerOfTen. */
    INT2 unitNumerator[LALNumUnits]; /**< Array of unit power numerators. */
    UINT2 unitDenominatorMinusOne[LALNumUnits]; /**< Array of unit power denominators-minus-one. */
} LALUnit;



typedef void XLALErrorHandlerType(const char *func, const char *file,
                                  int line, int errnum);

enum enumLALNameLength { LALNameLength = 64 };

struct LALoption
{
    const char *name;
    /* has_arg can't be an enum because some compilers complain about
     type mismatches in all the code that assumes it is an int.  */
    int has_arg;
    int *flag;
    int val;
};

typedef enum tagLALDetectorType {
    LALDETECTORTYPE_ABSENT,        /**< No FrDetector associated with this detector */
    LALDETECTORTYPE_IFODIFF,    /**< IFO in differential mode */
    LALDETECTORTYPE_IFOXARM,    /**< IFO in one-armed mode (X arm) */
    LALDETECTORTYPE_IFOYARM,    /**< IFO in one-armed mode (Y arm) */
    LALDETECTORTYPE_IFOCOMM,    /**< IFO in common mode */
    LALDETECTORTYPE_CYLBAR        /**< Cylindrical bar */
}
LALDetectorType;

typedef struct tagLALFrDetector
{
    CHAR    name[LALNameLength];    /**< A unique identifying string */
    CHAR    prefix[3];        /**< Two-letter prefix for detector's channel names */
    REAL8    vertexLongitudeRadians;    /**< The geodetic longitude \f$\lambda\f$ of the vertex in radians */
    REAL8    vertexLatitudeRadians;    /**< The geodetic latitude \f$\beta\f$ of the vertex in radians */
    REAL4    vertexElevation;    /**< The height of the vertex above the reference ellipsoid in meters */
    REAL4    xArmAltitudeRadians;    /**< The angle \f${\mathcal{A}}_X\f$ up from the local tangent plane of the reference ellipsoid to the X arm (or bar's cylidrical axis) in radians */
    REAL4    xArmAzimuthRadians;    /**< The angle \f$\zeta_X\f$ clockwise from North to the projection of the X arm (or bar's cylidrical axis) into the local tangent plane of the reference ellipsoid in radians */
    REAL4    yArmAltitudeRadians;    /**< The angle \f${\mathcal{A}}_Y\f$ up from the local tangent plane of the reference ellipsoid to the Y arm in radians (unused for bars: set it to zero) */
    REAL4    yArmAzimuthRadians;    /**< The angle \f$\zeta_Y\f$ clockwise from North to the projection of the Y arm into the local tangent plane of the reference ellipsoid in radians (unused for bars: set it to zero) */
    REAL4    xArmMidpoint;        /**< The distance to the midpoint of the X arm in meters (unused for bars: set it to zero) */
    REAL4    yArmMidpoint;        /**< The distance to the midpoint of the Y arm in meters (unused for bars: set it to zero) */
}
LALFrDetector;

typedef struct tagLALDetector
{
    REAL8        location[3];    /**< The three components, in an Earth-fixed Cartesian coordinate system, of the position vector from the center of the Earth to the detector in meters */
    REAL4        response[3][3];    /**< The Earth-fixed Cartesian components of the detector's response tensor \f$d^{ab}\f$ */
    LALDetectorType    type;        /**< The type of the detector (e.g., IFO in differential mode, cylindrical bar, etc.) */
    LALFrDetector    frDetector;    /**< The original LALFrDetector structure from which this was created */
}
LALDetector;

#define LAL_CAT(x,y) x ## y
#define LAL_XCAT(x,y) LAL_CAT(x,y)

/** expands to constant c of detector d */
#define LAL_DETECTOR_CONSTANT(d,c) LAL_XCAT(LAL_XCAT(LAL_,d),LAL_XCAT(_,c))

/** initializer for detector location vector */
#define LAL_DETECTOR_LOCATION(d) \
{ \
    LAL_DETECTOR_CONSTANT(d,VERTEX_LOCATION_X_SI),\
    LAL_DETECTOR_CONSTANT(d,VERTEX_LOCATION_Y_SI),\
    LAL_DETECTOR_CONSTANT(d,VERTEX_LOCATION_Z_SI) \
}

/** expands to component c (X,Y,Z) of arm X of detector d */
#define LAL_ARM_X(d,c) LAL_DETECTOR_CONSTANT(d,LAL_XCAT(ARM_X_DIRECTION_,c))

/** expands to component c (X,Y,Z) of arm Y of detector d */
#define LAL_ARM_Y(d,c) LAL_DETECTOR_CONSTANT(d,LAL_XCAT(ARM_Y_DIRECTION_,c))

/** expands to component c (X,Y,Z) of axis of detector d */
#define LAL_AXIS(d,c) LAL_DETECTOR_CONSTANT(d,LAL_XCAT(AXIS_DIRECTION_,c))

/** expands to a 3x3 matix initializer for the response for IFODIFF detector d */
#define LAL_DETECTOR_RESPONSE_IFODIFF(d) \
{ \
    { \
        0.5*( LAL_ARM_X(d,X) * LAL_ARM_X(d,X) - LAL_ARM_Y(d,X) * LAL_ARM_Y(d,X) ), \
        0.5*( LAL_ARM_X(d,X) * LAL_ARM_X(d,Y) - LAL_ARM_Y(d,X) * LAL_ARM_Y(d,Y) ), \
        0.5*( LAL_ARM_X(d,X) * LAL_ARM_X(d,Z) - LAL_ARM_Y(d,X) * LAL_ARM_Y(d,Z) )  \
    }, \
    { \
        0.5*( LAL_ARM_X(d,Y) * LAL_ARM_X(d,X) - LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,X) ), \
        0.5*( LAL_ARM_X(d,Y) * LAL_ARM_X(d,Y) - LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,Y) ), \
        0.5*( LAL_ARM_X(d,Y) * LAL_ARM_X(d,Z) - LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,Z) )  \
    }, \
    { \
        0.5*( LAL_ARM_X(d,Z) * LAL_ARM_X(d,X) - LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,X) ), \
        0.5*( LAL_ARM_X(d,Z) * LAL_ARM_X(d,Y) - LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,Y) ), \
        0.5*( LAL_ARM_X(d,Z) * LAL_ARM_X(d,Z) - LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,Z) )  \
    } \
}

/** expands to a 3x3 matix initializer for the response for IFOCOMM detector d */
#define LAL_DETECTOR_RESPONSE_IFOCOMM(d) \
{ \
    { \
        0.5*( LAL_ARM_X(d,X) * LAL_ARM_X(d,X) + LAL_ARM_Y(d,X) * LAL_ARM_Y(d,X) ), \
        0.5*( LAL_ARM_X(d,X) * LAL_ARM_X(d,Y) + LAL_ARM_Y(d,X) * LAL_ARM_Y(d,Y) ), \
        0.5*( LAL_ARM_X(d,X) * LAL_ARM_X(d,Z) + LAL_ARM_Y(d,X) * LAL_ARM_Y(d,Z) )  \
    }, \
    { \
        0.5*( LAL_ARM_X(d,Y) * LAL_ARM_X(d,X) + LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,X) ), \
        0.5*( LAL_ARM_X(d,Y) * LAL_ARM_X(d,Y) + LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,Y) ), \
        0.5*( LAL_ARM_X(d,Y) * LAL_ARM_X(d,Z) + LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,Z) )  \
    }, \
    { \
        0.5*( LAL_ARM_X(d,Z) * LAL_ARM_X(d,X) + LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,X) ), \
        0.5*( LAL_ARM_X(d,Z) * LAL_ARM_X(d,Y) + LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,Y) ), \
        0.5*( LAL_ARM_X(d,Z) * LAL_ARM_X(d,Z) + LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,Z) )  \
    } \
}

/** expands to a 3x3 matix initializer for the response for IFOXARM detector d */
#define LAL_DETECTOR_RESPONSE_IFOXARM(d) \
{ \
    { \
        0.5 * LAL_ARM_X(d,X) * LAL_ARM_X(d,X), \
        0.5 * LAL_ARM_X(d,X) * LAL_ARM_X(d,Y), \
        0.5 * LAL_ARM_X(d,X) * LAL_ARM_X(d,Z)  \
    }, \
    { \
        0.5 * LAL_ARM_X(d,Y) * LAL_ARM_X(d,X), \
        0.5 * LAL_ARM_X(d,Y) * LAL_ARM_X(d,Y), \
        0.5 * LAL_ARM_X(d,Y) * LAL_ARM_X(d,Z)  \
    }, \
    { \
        0.5 * LAL_ARM_X(d,Z) * LAL_ARM_X(d,X), \
        0.5 * LAL_ARM_X(d,Z) * LAL_ARM_X(d,Y), \
        0.5 * LAL_ARM_X(d,Z) * LAL_ARM_X(d,Z)  \
    } \
}

/** expands to a 3x3 matix initializer for the response for IFOYARM detector d */
#define LAL_DETECTOR_RESPONSE_IFOYARM(d) \
{ \
    { \
        0.5 * LAL_ARM_Y(d,X) * LAL_ARM_Y(d,X), \
        0.5 * LAL_ARM_Y(d,X) * LAL_ARM_Y(d,Y), \
        0.5 * LAL_ARM_Y(d,X) * LAL_ARM_Y(d,Z)  \
    }, \
    { \
        0.5 * LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,X), \
        0.5 * LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,Y), \
        0.5 * LAL_ARM_Y(d,Y) * LAL_ARM_Y(d,Z)  \
    }, \
    { \
        0.5 * LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,X), \
        0.5 * LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,Y), \
        0.5 * LAL_ARM_Y(d,Z) * LAL_ARM_Y(d,Z)  \
    } \
}

/** expands to a 3x3 matix initializer for the response for CYLBAR detector d */
#define LAL_DETECTOR_RESPONSE_CYLBAR(d) \
{ \
    { \
        LAL_AXIS(d,X) * LAL_AXIS(d,X), \
        LAL_AXIS(d,X) * LAL_AXIS(d,Y), \
        LAL_AXIS(d,X) * LAL_AXIS(d,Z)  \
    }, \
    { \
        LAL_AXIS(d,Y) * LAL_AXIS(d,X), \
        LAL_AXIS(d,Y) * LAL_AXIS(d,Y), \
        LAL_AXIS(d,Y) * LAL_AXIS(d,Z)  \
    }, \
    { \
        LAL_AXIS(d,Z) * LAL_AXIS(d,X), \
        LAL_AXIS(d,Z) * LAL_AXIS(d,Y), \
        LAL_AXIS(d,Z) * LAL_AXIS(d,Z)  \
    } \
}

#define LAL_FR_STREAM_DETECTOR_STRUCT(d) \
{ \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_NAME), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_PREFIX), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_LONGITUDE_RAD), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_LATITUDE_RAD), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ELEVATION_SI), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ARM_X_ALTITUDE_RAD), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ARM_X_AZIMUTH_RAD), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ARM_Y_ALTITUDE_RAD), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ARM_Y_AZIMUTH_RAD), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ARM_X_MIDPOINT_SI), \
    LAL_DETECTOR_CONSTANT(d,DETECTOR_ARM_Y_MIDPOINT_SI) \
}

#define LAL_DETECTOR_RESPONSE(d,t) \
    LAL_XCAT( LAL_DETECTOR_RESPONSE_, t )(d)

#define LAL_DETECTOR_STRUCT(d,t) \
{ \
    LAL_DETECTOR_LOCATION(d),      \
    LAL_DETECTOR_RESPONSE(d,t),    \
    LAL_XCAT(LALDETECTORTYPE_,t),  \
    LAL_FR_STREAM_DETECTOR_STRUCT(d)      \
}



/** Vector of type REAL8, see \ref ss_Vector for more details. */
typedef struct tagREAL8Vector
{
    UINT4  length; /**< Number of elements in array. */
    REAL8 *data; /**< Pointer to the data array. */
}
REAL8Vector;

typedef REAL8Vector     REAL8Sequence;    /**< See \ref ss_Sequence for documentation */


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

struct tagLALREAL8SequenceInterp {
    const REAL8Sequence *s;
    int kernel_length;
    double *cached_kernel;
    double residual;
    /* samples.  the length of the kernel sets the bandwidth of the
     * interpolator:  the longer the kernel, the closer to an ideal
     * interpolator it becomes.  we tie the interval at which the
     * kernel is regenerated to this in a heuristic way to hide the
     * sub-sample residual quantization in the filter's roll-off. */
    double noop_threshold;
    /* calling-code supplied kernel generator */
    void (*kernel)(double *, int, double, void *);
    void *kernel_data;
};

typedef struct tagLALREAL8SequenceInterp LALREAL8SequenceInterp;

struct tagLALREAL8TimeSeriesInterp {
    const REAL8TimeSeries *series;
    LALREAL8SequenceInterp *seqinterp;
};

typedef struct tagLALREAL8TimeSeriesInterp LALREAL8TimeSeriesInterp;

