

#define XLALErrorHandler ( * XLALGetErrorHandlerPtr() ) /**< Modifiable lvalue containing the XLAL error handler */

static void XLALDestroyErrorHandlerPtr(void *xlalErrorHandlerPtr);
static void XLALCreateErrorHandlerKey(void);
int LALError(LALStatus * status, const char *statement);
void LALAbort(const char *fmt, ...);
void XLALDefaultErrorHandler(const char *func, const char *file, int line,
                             int errnum);
int XLALPrintWarning(const char *fmt, ...);
int XLALVPrintWarning(const char *fmt, va_list ap);

int XLALPrintError(const char *fmt, ...);

const char *XLALErrorString(int code);
int XLALVPrintError(const char *fmt, va_list ap);
void XLALPrintErrorMessage(const char *func, const char *file, int line,
                           const char *fmt, ...);
void XLALError(const char *func, const char *file, int line, int errnum);
void XLALVPrintErrorMessage(const char *func, const char *file, int line,
                            const char *fmt, va_list ap);

void XLALBacktraceErrorHandler(const char *func, const char *file,
                               int line, int errnum);
LIGOTimeGPS * XLALINT8NSToGPS( LIGOTimeGPS *epoch, INT8 ns );
INT8 XLALGPSToINT8NS( const LIGOTimeGPS *epoch );
REAL8Vector* XLALCreateREAL8Vector(UINT4 length);
int XLALSetErrno(int errnum);
XLALErrorHandlerType **XLALGetErrorHandlerPtr(void);
void XLALPrintWarningMessage(const char *func, const char *file, int line,
                             const char *fmt, ...);
void XLALVPrintWarningMessage(const char *func, const char *file, int line,
                              const char *fmt, va_list ap);

int LALPrintError(const char *fmt, ...);

XLALErrorHandlerType *XLALSetErrorHandler(XLALErrorHandlerType *
                                          newHandler);

LIGOTimeGPS * XLALGPSAddGPS( LIGOTimeGPS *epoch, const LIGOTimeGPS *dt );

REAL8TimeSeries *XLALCreateREAL8TimeSeries ( const char *name, const LIGOTimeGPS *epoch, REAL8 f0, REAL8 deltaT, const LALUnit *sampleUnits, int length );

REAL8TimeSeries *XLALSimDetectorStrainREAL8TimeSeries(
                                                      const REAL8TimeSeries *hplus,
                                                      const REAL8TimeSeries *hcross,
                                                      REAL8 right_ascension,
                                                      REAL8 declination,
                                                      REAL8 psi,
                                                      const LALDetector *detector
                                                      );

void XLALPerror(const char *func, const char *file, int line, int code);


char *XLALGPSToStr(char *s, const LIGOTimeGPS *t);

LIGOTimeGPS * XLALGPSAdd( LIGOTimeGPS *epoch, REAL8 dt );

void XLALDestroyREAL8TimeSeries( REAL8TimeSeries * series );

void LALCheckMemoryLeaks(void);
void XLALDestroyREAL8Vector(REAL8Vector* v);

REAL8 XLALGreenwichMeanSiderealTime(
                                    const LIGOTimeGPS *gpstime
                                    );
REAL8 XLALGreenwichSiderealTime(
                                const LIGOTimeGPS *gpstime,
                                REAL8 equation_of_equinoxes
                                );

struct tm * XLALGPSToUTC(
                         struct tm *utc, /**< [Out] Pointer to tm struct where result is stored. */
                         INT4 gpssec /**< [In] Seconds since the GPS epoch. */
);

double XLALTimeDelayFromEarthCenter(
                                    const double detector_earthfixed_xyz_metres[3],
                                    double source_right_ascension_radians,
                                    double source_declination_radians,
                                    const LIGOTimeGPS *gpstime
                                    );

void XLALComputeDetAMResponse(
                              double *fplus,        /**< Returned value of F+ */
                              double *fcross,        /**< Returned value of Fx */
                              const REAL4 D[3][3],    /**< Detector response 3x3 matrix */
                              const double ra,    /**< Right ascention of source (radians) */
                              const double dec,    /**< Declination of source (radians) */
                              const double psi,    /**< Polarization angle of source (radians) */
                              const double gmst    /**< Greenwich mean sidereal time (radians) */
);

REAL8 XLALGreenwichMeanSiderealTime(
                                    const LIGOTimeGPS *gpstime
                                    );

REAL8 XLALGPSGetREAL8( const LIGOTimeGPS *epoch );

int XLALStrToGPS(LIGOTimeGPS *t, const char *nptr, char **endptr);

int
LALgetopt_long_only (int argc, char *const *argv, const char *options,
                     const struct LALoption *long_options, int *opt_index);

int XLALGPSCmp( const LIGOTimeGPS *t0, const LIGOTimeGPS *t1 );

REAL8 XLALGPSDiff( const LIGOTimeGPS *t1, const LIGOTimeGPS *t0 );






