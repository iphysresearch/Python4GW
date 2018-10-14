
/** ERROR **/

#define XLAL_ERROR_NULL(s) return NULL
#define XLAL_ERROR_REAL8(s) return XLAL_REAL8_FAIL_NAN
#define XLAL_ERROR(s) return XLAL_FAILURE
#define XLAL_ERROR_VAL(val, ...) return val

#define LAL_CHECK_VALID_SERIES(s,val) \
do { \
if ( !(s) ) XLAL_ERROR_VAL( val, XLAL_EFAULT ); \
if ( !(s)->data || !(s)->data->data || !(s)->data->length ) XLAL_ERROR_VAL( val, XLAL_EINVAL ); \
} while (0)

#define LAL_CHECK_CONSISTENT_TIME_SERIES(s1,s2,val) \
do { \
if ( XLALGPSCmp( &(s1)->epoch, &(s2)->epoch ) != 0 ) XLAL_ERROR_VAL( val, XLAL_ETIME ); \
if ( fabs( (s1)->deltaT - (s2)->deltaT ) > LAL_REAL8_EPS ) XLAL_ERROR_VAL( val, XLAL_ETIME ); \
if ( fabs( (s1)->f0 - (s2)->f0 ) > LAL_REAL8_EPS ) XLAL_ERROR_VAL( val, XLAL_EFREQ ); \
if ( XLALUnitCompare( &(s1)->sampleUnits, &(s2)->sampleUnits ) ) XLAL_ERROR_VAL( val, XLAL_EUNIT ); \
if ( (s1)->data->length != (s1)->data->length ) XLAL_ERROR_VAL(val, XLAL_EBADLEN ); \
} while (0)

#define LAL_CHECK_COMPATIBLE_TIME_SERIES(s1,s2,val) \
do { \
if ( fabs( (s1)->deltaT - (s2)->deltaT ) > LAL_REAL8_EPS ) XLAL_ERROR_VAL( val, XLAL_ETIME ); \
if ( fabs( (s1)->f0 - (s2)->f0 ) > LAL_REAL8_EPS ) XLAL_ERROR_VAL( val, XLAL_EFREQ ); \
if ( XLALUnitCompare( &(s1)->sampleUnits, &(s2)->sampleUnits ) ) XLAL_ERROR_VAL( val, XLAL_EUNIT ); \
} while (0)

/** END ERROR **/



static double dotprod(const double vec1[3], const double vec2[3]);
static int delta_tai_utc( INT4 gpssec );


void *(XLALMalloc) (size_t n);


INT8 XLALGPSToINT8NS( const LIGOTimeGPS *epoch );
LIGOTimeGPS * XLALINT8NSToGPS( LIGOTimeGPS *epoch, INT8 ns );
LIGOTimeGPS * XLALGPSSet( LIGOTimeGPS *epoch, INT4 gpssec, INT8 gpsnan );
LIGOTimeGPS * XLALGPSSetREAL8( LIGOTimeGPS *epoch, REAL8 t );
REAL8 XLALGPSGetREAL8( const LIGOTimeGPS *epoch );
LIGOTimeGPS * XLALGPSAdd( LIGOTimeGPS *epoch, REAL8 dt );
REAL8 XLALGPSDiff( const LIGOTimeGPS *t1, const LIGOTimeGPS *t0 );
int XLALGPSCmp( const LIGOTimeGPS *t0, const LIGOTimeGPS *t1 );
LIGOTimeGPS * XLALGPSAddGPS( LIGOTimeGPS *epoch, const LIGOTimeGPS *dt );
REAL8 XLALGreenwichMeanSiderealTime(const LIGOTimeGPS *gpstime);
REAL8 XLALGreenwichSiderealTime(const LIGOTimeGPS *gpstime,REAL8 equation_of_equinoxes);
double XLALTimeDelayFromEarthCenter(const double detector_earthfixed_xyz_metres[3],double source_right_ascension_radians,double source_declination_radians,const LIGOTimeGPS *gpstime);
double XLALArrivalTimeDiff( const double detector1_earthfixed_xyz_metres[3], const double detector2_earthfixed_xyz_metres[3], const double source_right_ascension_radians, const double source_declination_radians, const LIGOTimeGPS *gpstime);

/* Returns the leap seconds TAI-UTC at a given GPS second. */
int XLALLeapSeconds( INT4 gpssec ); // Dont Have


REAL8TimeSeries *XLALCreateREAL8TimeSeries ( const char *name, const LIGOTimeGPS *epoch, REAL8 f0, REAL8 deltaT, const LALUnit *sampleUnits, int length );
void XLALDestroyREAL8TimeSeries( REAL8TimeSeries * series );
REAL8Vector* XLALCreateREAL8Vector(UINT4 length);
void XLALDestroyREAL8Vector(REAL8Vector* v);
int XLALUnitCompare( const LALUnit *unit1, const LALUnit *unit2 );
LALREAL8TimeSeriesInterp *XLALREAL8TimeSeriesInterpCreate(const REAL8TimeSeries *series, int kernel_length, void (*kernel)(double *, int, double, void *), void *kernel_data);
int XLALUnitNormalize( LALUnit *unit );
LALREAL8SequenceInterp *XLALREAL8SequenceInterpCreate(const REAL8Sequence *s, int kernel_length, void (*kernel)(double *, int, double, void *), void *kernel_data);
void XLALREAL8SequenceInterpDestroy(LALREAL8SequenceInterp *interp);
void XLALREAL8TimeSeriesInterpDestroy(LALREAL8TimeSeriesInterp *interp);
REAL8 XLALREAL8TimeSeriesInterpEval(LALREAL8TimeSeriesInterp *interp, const LIGOTimeGPS *t, int bounds_check);
REAL8 XLALREAL8SequenceInterpEval(LALREAL8SequenceInterp *interp, double x, int bounds_check);



REAL8 XLALGPSModf( REAL8 *iptr, const LIGOTimeGPS *epoch );
char *XLALGPSToStr(char *s, const LIGOTimeGPS *t);
struct tm * XLALGPSToUTC(struct tm *utc,INT4 gpssec);
REAL8 XLALConvertCivilTimeToJD( const struct tm *civil);
int XLALStrToGPS(LIGOTimeGPS *t, const char *nptr, char **endptr);

