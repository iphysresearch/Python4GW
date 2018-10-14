#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <time.h>
#include "time.h"
#include <locale.h>
#include <ctype.h>
#include <stdarg.h>


#include "MYdatatypes.h"
#include "MYconstants.h"
#include "MYfuns.h"


static void default_kernel(double *cached_kernel, int kernel_length, double residual, void *data);
struct default_kernel_data {
    double welch_factor;
};

static const struct leaps_table { REAL8 jd; INT4 gpssec; int taiutc; } leaps[] =
{
    {2444239.5,    -43200, 19},  /* 1980-Jan-01 */
    {2444786.5,  46828800, 20},  /* 1981-Jul-01 */
    {2445151.5,  78364801, 21},  /* 1982-Jul-01 */
    {2445516.5, 109900802, 22},  /* 1983-Jul-01 */
    {2446247.5, 173059203, 23},  /* 1985-Jul-01 */
#if 0
    /* NOTE: IF THIS WERE A NEGATIVE LEAP SECOND, INSERT AS FOLLOWS */
    {2447161.5, 252028803, 22},  /* 1988-Jan-01 EXAMPLE ONLY! */
#endif
    {2447161.5, 252028804, 24},  /* 1988-Jan-01 */
    {2447892.5, 315187205, 25},  /* 1990-Jan-01 */
    {2448257.5, 346723206, 26},  /* 1991-Jan-01 */
    {2448804.5, 393984007, 27},  /* 1992-Jul-01 */
    {2449169.5, 425520008, 28},  /* 1993-Jul-01 */
    {2449534.5, 457056009, 29},  /* 1994-Jul-01 */
    {2450083.5, 504489610, 30},  /* 1996-Jan-01 */
    {2450630.5, 551750411, 31},  /* 1997-Jul-01 */
    {2451179.5, 599184012, 32},  /* 1999-Jan-01 */
    {2453736.5, 820108813, 33},  /* 2006-Jan-01 */
    {2454832.5, 914803214, 34},  /* 2009-Jan-01 */
    {2456109.5, 1025136015, 35}, /* 2012-Jul-01 */
    {2457204.5, 1119744016, 36}, /* 2015-Jul-01 */
    {2457754.5, 1167264017, 37}, /* 2017-Jan-01 */
};
static const int numleaps = sizeof( leaps ) / sizeof( *leaps );





/** Math **/
static double dotprod(const double vec1[3], const double vec2[3])
{
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2];
}

static int delta_tai_utc( INT4 gpssec )
{
    int leap;
    
    /* assume calling function has already checked this */
    /*
     if ( gpssec <= leaps[0].gpssec )
     {
     fprintf( stderr, "error: don't know leap seconds before gps time %d\n",
     leaps[0].gpssec );
     abort();
     }
     */
    
    for ( leap = 1; leap < numleaps; ++leap )
        if ( gpssec == leaps[leap].gpssec )
            return leaps[leap].taiutc - leaps[leap-1].taiutc;
    
    return 0;
}

int gcd(int m,int n)
{
    int i,rgcd=1;
    for (i=2; i<=(m<n?m:n); )
       if (!(m%i)&&!(n%i))
        {
            m/=i;
            n/=i;
            rgcd*=i;
        }
       else if (!(m%i)) m/=i;
       else if (!(n%i)) n/=i;
       else i++;
    return rgcd;
}
/** END Math **/





/** Malloc **/
void *(XLALMalloc) (size_t n)
{
    void *p;
    p = malloc(n);
    return p;
}

/** END Malloc **/














/** XLALtime **/

INT8 XLALGPSToINT8NS( const LIGOTimeGPS *epoch )
{
    return XLAL_BILLION_INT8 * epoch->gpsSeconds + epoch->gpsNanoSeconds;
}

LIGOTimeGPS * XLALINT8NSToGPS( LIGOTimeGPS *epoch, INT8 ns )
{
    INT8 gpsSeconds = ns / XLAL_BILLION_INT8;
    epoch->gpsSeconds     = gpsSeconds;
    epoch->gpsNanoSeconds = ns % XLAL_BILLION_INT8;
    if( (INT8) epoch->gpsSeconds != gpsSeconds ) {
        fprintf(stderr, "%s(): overflow: %lld", __func__, ns );
        XLAL_ERROR_NULL( XLAL_EDOM );
    }
    return epoch;
}

LIGOTimeGPS * XLALGPSSet( LIGOTimeGPS *epoch, INT4 gpssec, INT8 gpsnan )
{
    return XLALINT8NSToGPS( epoch, XLAL_BILLION_INT8 * gpssec + gpsnan );
}

LIGOTimeGPS * XLALGPSSetREAL8( LIGOTimeGPS *epoch, REAL8 t )
{
    INT4 gpssec = floor(t);
    INT4 gpsnan = nearbyint((t - gpssec) * XLAL_BILLION_REAL8);
    if(isnan(t)) {
        fprintf(stderr,"%s(): NaN", __func__);
        XLAL_ERROR_NULL(XLAL_EFPINVAL);
    }
    if(fabs(t) > 0x7fffffff) {
        fprintf(stderr,"%s(): overflow %.17g", __func__, t);
        XLAL_ERROR_NULL(XLAL_EDOM);
    }
    /* use XLALGPSSet() to normalize the nanoseconds */
    return XLALGPSSet(epoch, gpssec, gpsnan);
}

REAL8 XLALGPSGetREAL8( const LIGOTimeGPS *epoch )
{
    return epoch->gpsSeconds + (epoch->gpsNanoSeconds / XLAL_BILLION_REAL8);
}

LIGOTimeGPS * XLALGPSAddGPS( LIGOTimeGPS *epoch, const LIGOTimeGPS *dt )
{
    /* when GPS times are converted to 8-byte counts of nanoseconds their sum
     * cannot overflow, however it might not be possible to convert the sum
     * back to a LIGOTimeGPS without overflowing.  that is caught by the
     * XLALINT8NSToGPS() function */
    return XLALINT8NSToGPS( epoch, XLALGPSToINT8NS( epoch ) + XLALGPSToINT8NS( dt ) );
}

LIGOTimeGPS * XLALGPSAdd( LIGOTimeGPS *epoch, REAL8 dt )
{
    LIGOTimeGPS dt_gps;
    if(!XLALGPSSetREAL8(&dt_gps, dt))
        XLAL_ERROR_NULL(XLAL_EFUNC);
    return XLALGPSAddGPS(epoch, &dt_gps);
}

REAL8 XLALGPSDiff( const LIGOTimeGPS *t1, const LIGOTimeGPS *t0 )
{
    double hi = t1->gpsSeconds - t0->gpsSeconds;
    double lo = t1->gpsNanoSeconds - t0->gpsNanoSeconds;
    return hi + lo / XLAL_BILLION_REAL8;
}

int XLALGPSCmp( const LIGOTimeGPS *t0, const LIGOTimeGPS *t1 )
{
    if ( t0 == NULL || t1 == NULL ) {
        return ( t1 != NULL ) ? -1 : ( ( t0 != NULL ) ? 1 : 0 );
    }
    else {
        INT8 ns0 = XLALGPSToINT8NS( t0 );
        INT8 ns1 = XLALGPSToINT8NS( t1 );
        return ( ns0 > ns1 ) - ( ns0 < ns1 );
    }
}

REAL8 XLALGreenwichMeanSiderealTime(
                                    const LIGOTimeGPS *gpstime
                                    )
{
    return XLALGreenwichSiderealTime(gpstime, 0.0);
}

REAL8 XLALGreenwichSiderealTime(
                                const LIGOTimeGPS *gpstime,
                                REAL8 equation_of_equinoxes
                                )
{
    struct tm utc;
    double julian_day;
    double t_hi, t_lo;
    double t;
    double sidereal_time;
    
    /*
     * Convert GPS seconds to UTC.  This is where we pick up knowledge
     * of leap seconds which are required for the mapping of atomic
     * time scales to celestial time scales.  We deal only with integer
     * seconds.
     */
    
    if(!XLALGPSToUTC(&utc, gpstime->gpsSeconds))
        XLAL_ERROR_REAL8(XLAL_EFUNC);
    
    /*
     * And now to Julian day number.  Again, only accurate to integer
     * seconds.
     */
    
    julian_day = XLALConvertCivilTimeToJD(&utc);
    if(XLAL_IS_REAL8_FAIL_NAN(julian_day))
        XLAL_ERROR_REAL8(XLAL_EFUNC);
    
    /*
     * Convert Julian day number to the number of centuries since the
     * Julian epoch (1 century = 36525.0 days).  Here, we incorporate
     * the fractional part of the seconds.  For precision, we keep
     * track of the most significant and least significant parts of the
     * time separately.  The original code in NOVAS-C determined t_hi
     * and t_lo from Julian days, with t_hi receiving the integer part
     * and t_lo the fractional part.  Because LAL's Julian day routine
     * is accurate to the second, here the hi/lo split is most
     * naturally done at the integer seconds boundary.  Note that the
     * "hi" and "lo" components have the same units and so the split
     * can be done anywhere.
     */
    
    t_hi = (julian_day - XLAL_EPOCH_J2000_0_JD) / 36525.0;
    t_lo = gpstime->gpsNanoSeconds / (1e9 * 36525.0 * 86400.0);
    
    /*
     * Compute sidereal time in sidereal seconds.  (magic)
     */
    
    t = t_hi + t_lo;
    
    sidereal_time = equation_of_equinoxes + (-6.2e-6 * t + 0.093104) * t * t + 67310.54841;
    sidereal_time += 8640184.812866 * t_lo;
    sidereal_time += 3155760000.0 * t_lo;
    sidereal_time += 8640184.812866 * t_hi;
    sidereal_time += 3155760000.0 * t_hi;
    
    /*
     * Return radians (2 pi radians in 1 sidereal day = 86400 sidereal
     * seconds).
     */
    
    return sidereal_time * LAL_PI / 43200.0;
}

double XLALTimeDelayFromEarthCenter(
                                    const double detector_earthfixed_xyz_metres[3],
                                    double source_right_ascension_radians,
                                    double source_declination_radians,
                                    const LIGOTimeGPS *gpstime
                                    )
{
    static const double earth_center[3] = {0.0, 0.0, 0.0};
    
    /*
     * This is positive when the wavefront arrives at the detector
     * after arriving at the geocentre.
     */
    
    return XLALArrivalTimeDiff(detector_earthfixed_xyz_metres, earth_center, source_right_ascension_radians, source_declination_radians, gpstime);
}

REAL8 XLALConvertCivilTimeToJD( const struct tm *civil /**< [In] civil time in a broken down time structure. */ )
{
    const int sec_per_day = 60 * 60 * 24; /* seconds in a day */
    int year, month, day, sec;
    REAL8 jd;
    
    /* this routine only works for dates after 1900 */
    if ( civil->tm_year <= 0 )
    {
        fprintf(stderr, "XLAL Error - Year must be after 1900\n" );
        XLAL_ERROR_REAL8( XLAL_EDOM );
    }
    
    year  = civil->tm_year + 1900;
    month = civil->tm_mon + 1;     /* month is in range 1-12 */
    day   = civil->tm_mday;        /* day is in range 1-31 */
    sec   = civil->tm_sec + 60*(civil->tm_min + 60*(civil->tm_hour)); /* seconds since midnight */
    
    jd = 367*year - 7*(year + (month + 9)/12)/4 + 275*month/9 + day + 1721014;
    /* note: Julian days start at noon: subtract half a day */
    jd += (REAL8)sec/(REAL8)sec_per_day - 0.5;
    
    return jd;
} // XLALConvertCivilTimeToJD()

double XLALArrivalTimeDiff(
                    const double detector1_earthfixed_xyz_metres[3],
                    const double detector2_earthfixed_xyz_metres[3],
                    const double source_right_ascension_radians,
                    const double source_declination_radians,
                    const LIGOTimeGPS *gpstime
                    )
{
    double delta_xyz[3];
    double ehat_src[3];
    const double greenwich_hour_angle = XLALGreenwichMeanSiderealTime(gpstime) - source_right_ascension_radians;
    
    if(XLAL_IS_REAL8_FAIL_NAN(greenwich_hour_angle))
        XLAL_ERROR_REAL8(XLAL_EFUNC);
    
    /*
     * compute the unit vector pointing from the geocenter to the
     * source
     */
    
    ehat_src[0] = cos(source_declination_radians) * cos(greenwich_hour_angle);
    ehat_src[1] = cos(source_declination_radians) * -sin(greenwich_hour_angle);
    ehat_src[2] = sin(source_declination_radians);
    
    /*
     * position of detector 2 with respect to detector 1
     */
    
    delta_xyz[0] = detector2_earthfixed_xyz_metres[0] - detector1_earthfixed_xyz_metres[0];
    delta_xyz[1] = detector2_earthfixed_xyz_metres[1] - detector1_earthfixed_xyz_metres[1];
    delta_xyz[2] = detector2_earthfixed_xyz_metres[2] - detector1_earthfixed_xyz_metres[2];
    
    /*
     * Arrival time at detector 1 - arrival time at detector 2.  This
     * is positive when the wavefront arrives at detector 1 after
     * detector 2 (and so t at detector 1 is greater than t at detector
     * 2).
     */
    
    return dotprod(ehat_src, delta_xyz) / LAL_C_SI;
}


/** END XLALtime **/













/** Data Fac **/

REAL8TimeSeries *XLALCreateREAL8TimeSeries ( const char *name, const LIGOTimeGPS *epoch, REAL8 f0, REAL8 deltaT, const LALUnit *sampleUnits, int length )
{
    REAL8TimeSeries *r8ts;
    REAL8Vector *r8sequence;
    
    r8ts = (REAL8TimeSeries *)malloc(sizeof(*r8ts));
    r8sequence = XLALCreateREAL8Vector(length);
    if(!r8ts || !r8sequence) {
        free(r8ts);
        XLALDestroyREAL8Vector (r8sequence);
    }
    
    if(name)
        strncpy(r8ts->name, name, LALNameLength);
    else
        r8ts->name[0] = '\0';
    r8ts->epoch = *epoch;
    r8ts->f0 = f0;
    r8ts->deltaT = deltaT;
    r8ts->sampleUnits = *sampleUnits;
    r8ts->data = r8sequence;
    
    return r8ts;
}

void XLALDestroyREAL8TimeSeries( REAL8TimeSeries * series )
{
    if(series)
        XLALDestroyREAL8Vector(series->data);
    free(series);
}

REAL8Vector* XLALCreateREAL8Vector(UINT4 length)
{
    REAL8Vector* vector;
    vector = (REAL8Vector*)malloc(sizeof(*vector));
    vector->length = length;
    if ( ! length ) /* zero length: set data pointer to be NULL */
    {
        vector->data = NULL;
    }
    else /* non-zero length: allocate memory for data */
    {
        vector->data = (REAL8 *)malloc(length * sizeof( *vector->data));
    }
    
    return vector;
}

void XLALDestroyREAL8Vector(REAL8Vector* v)
{
    if(NULL == v)
    {
        return;
    }
    if(v->data)
    {
        v->length = 0;
        free(v->data);
    }
    v->data = NULL;
    free(v);
    
    return;
}

int XLALUnitCompare( const LALUnit *unit1, const LALUnit *unit2 )
{
    LALUnit  unitOne, unitTwo;
    
    if ( ! unit1 || ! unit2 )
        XLAL_ERROR( XLAL_EFAULT );
    
    unitOne = *unit1;
    unitTwo = *unit2;
    
    /* normalize the units */
    if ( XLALUnitNormalize( &unitOne ) )
        XLAL_ERROR( XLAL_EFUNC );
    if ( XLALUnitNormalize( &unitTwo ) )
        XLAL_ERROR( XLAL_EFUNC );
    
    /* factors of 10 disagree? */
    if ( unitOne.powerOfTen != unitTwo.powerOfTen )
        return 1;
    
    /* powers of dimensions disagree? use memcmp() to compare the arrays */
    if ( memcmp( unitOne.unitNumerator, unitTwo.unitNumerator, LALNumUnits * sizeof( *unitOne.unitNumerator ) ) )
        return 1;
    if ( memcmp( unitOne.unitDenominatorMinusOne, unitTwo.unitDenominatorMinusOne, LALNumUnits * sizeof( *unitOne.unitDenominatorMinusOne ) ) )
        return 1;
    
    /* agree in all possible ways */
    return 0;
}

LALREAL8TimeSeriesInterp *XLALREAL8TimeSeriesInterpCreate(const REAL8TimeSeries *series, int kernel_length, void (*kernel)(double *, int, double, void *), void *kernel_data)
{
    LALREAL8TimeSeriesInterp *interp;
    LALREAL8SequenceInterp *seqinterp;
    
    interp = XLALMalloc(sizeof(*interp));
    seqinterp = XLALREAL8SequenceInterpCreate(series->data, kernel_length, kernel, kernel_data);
    if(!interp || !seqinterp) {
        XLALFree(interp);
        XLALREAL8SequenceInterpDestroy(seqinterp);
        XLAL_ERROR_NULL(XLAL_EFUNC);
    }
    
    interp->series = series;
    interp->seqinterp = seqinterp;
    
    return interp;
}

int XLALUnitNormalize( LALUnit *unit )
{
    UINT2 commonFactor;
    UINT2 i;
    
    if ( ! unit )
        XLAL_ERROR( XLAL_EFAULT );
    
    for (i=0; i<LALNumUnits; ++i)
    {
        commonFactor = gcd ( unit->unitNumerator[i], unit->unitDenominatorMinusOne[i] + 1 );
        unit->unitNumerator[i] /= commonFactor;
        unit->unitDenominatorMinusOne[i] = ( unit->unitDenominatorMinusOne[i] + 1 ) / commonFactor - 1;
    } /* for i */
    
    return 0;
}

LALREAL8SequenceInterp *XLALREAL8SequenceInterpCreate(const REAL8Sequence *s, int kernel_length, void (*kernel)(double *, int, double, void *), void *kernel_data)
{
    LALREAL8SequenceInterp *interp;
    double *cached_kernel;
    
    if(kernel_length < 3)
        XLAL_ERROR_NULL(XLAL_EDOM);
    if(!kernel && kernel_data) {
        /* FIXME:  print error message */
        XLAL_ERROR_NULL(XLAL_EINVAL);
    }
    /* interpolator induces phase shifts unless this is odd */
    kernel_length -= (~kernel_length) & 1;
    
    interp = XLALMalloc(sizeof(*interp));
    cached_kernel = XLALMalloc(kernel_length * sizeof(*cached_kernel));
    if(!interp || !cached_kernel) {
        XLALFree(interp);
        XLALFree(cached_kernel);
        XLAL_ERROR_NULL(XLAL_EFUNC);
    }
    
    interp->s = s;
    interp->kernel_length = kernel_length;
    interp->cached_kernel = cached_kernel;
    /* >= 1 --> impossible.  forces kernel init on first eval */
    interp->residual = 2.;
    /* set no-op threshold.  the kernel is recomputed when the residual
     * changes by this much */
    interp->noop_threshold = 1. / (4 * interp->kernel_length);
    
    /* install interpolator, using default if needed */
    if(!kernel) {
        struct default_kernel_data *default_kernel_data = XLALMalloc(sizeof(*default_kernel_data));
        if(!default_kernel_data) {
            XLALFree(interp);
            XLALFree(cached_kernel);
            XLAL_ERROR_NULL(XLAL_EFUNC);
        }
        
        default_kernel_data->welch_factor = 1.0 / ((kernel_length - 1.) / 2. + 1.);
        
        kernel = default_kernel;
        kernel_data = default_kernel_data;
    } else if(kernel == default_kernel) {
        /* not allowed because destroy method checks for the
         * default kernel to decide if it must free the kernel_data
         * memory */
        XLALFree(interp);
        XLALFree(cached_kernel);
        XLAL_ERROR_NULL(XLAL_EINVAL);
    }
    interp->kernel = kernel;
    interp->kernel_data = kernel_data;
    
    return interp;
}

void XLALREAL8SequenceInterpDestroy(LALREAL8SequenceInterp *interp)
{
    if(interp) {
        XLALFree(interp->cached_kernel);
        /* unref the REAL8Sequence.  place-holder in case this code
         * is ported to a language where this matters */
        interp->s = NULL;
        /* only free if we allocated it ourselves */
        if(interp->kernel == default_kernel)
            XLALFree(interp->kernel_data);
        /* unref kernel and kernel_data.  place-holder in case this
         * code is ported to a language where this matters */
        interp->kernel = NULL;
        interp->kernel_data = NULL;
    }
    XLALFree(interp);
}

void XLALREAL8TimeSeriesInterpDestroy(LALREAL8TimeSeriesInterp *interp)
{
    if(interp) {
        XLALREAL8SequenceInterpDestroy(interp->seqinterp);
        /* unref the REAL8TimeSeries.  place-holder in case this
         * code is ported to a language where this matters */
        interp->series = NULL;
    }
    XLALFree(interp);
}

REAL8 XLALREAL8TimeSeriesInterpEval(LALREAL8TimeSeriesInterp *interp, const LIGOTimeGPS *t, int bounds_check)
{
    return XLALREAL8SequenceInterpEval(interp->seqinterp, XLALGPSDiff(t, &interp->series->epoch) / interp->series->deltaT, bounds_check);
}

REAL8 XLALREAL8SequenceInterpEval(LALREAL8SequenceInterp *interp, double x, int bounds_check)
{
    const REAL8 *data = interp->s->data;
    double *cached_kernel = interp->cached_kernel;
    double *stop = cached_kernel + interp->kernel_length;
    /* split the real-valued sample index into integer and fractional
     * parts.  the fractional part (residual) is the offset in samples
     * from where we want to evaluate the function to where we know its
     * value.  the interpolating kernel depends only on this quantity.
     * when we compute a kernel, we record the value of this quantity,
     * and only recompute the kernel if this quantity differs from the
     * one for which the kernel was computed by more than the no-op
     * threshold */
    int start = lround(x);
    double residual = start - x;
    REAL8 val;
    
    if(!isfinite(x) || (bounds_check && (x < 0 || x >= interp->s->length)))
        XLAL_ERROR_REAL8(XLAL_EDOM);
    
    /* special no-op case for default kernel */
    if(fabs(residual) < interp->noop_threshold && interp->kernel == default_kernel)
        return 0 <= start && start < (int) interp->s->length ? data[start] : 0.0;
    
    /* need new kernel? */
    if(fabs(residual - interp->residual) >= interp->noop_threshold) {
        interp->kernel(cached_kernel, interp->kernel_length, residual, interp->kernel_data);
        interp->residual = residual;
    }
    
    /* inner product of kernel and samples */
    start -= (interp->kernel_length - 1) / 2;
    if(start + interp->kernel_length > (signed) interp->s->length)
        stop -= start + interp->kernel_length - interp->s->length;
    if(start < 0)
        cached_kernel -= start;
    else
        data += start;
    for(val = 0.0; cached_kernel < stop;)
        val += *cached_kernel++ * *data++;
    
    return val;
}


static void default_kernel(double *cached_kernel, int kernel_length, double residual, void *data)
{
    /* kernel is Welch-windowed sinc function.  the sinc component
     * takes the form
     *
     *    x = pi (i - x);
     *    kern = sin(x) / x
     *
     * we don't check for 0/0 because that can only occur if x is an
     * integer, which is trapped by no-op path above.  note that the
     * argument of sin(x) increases by pi each iteration, so we just
     * need to compute its value for the first iteration then flip sign
     * for each subsequent iteration.  for numerical reasons, it's
     * better to compute sin(x) from residual rather than from (start -
     * x), i.e. what it's argument should be for the first iteration,
     * so we also have to figure out how many factors of -1 to apply to
     * get its sign right for the first iteration.
     */
    
    double welch_factor = ((struct default_kernel_data *) data)->welch_factor;
    double *stop = cached_kernel + kernel_length;
    /* put a factor of welch_factor in this.  see below */
    double sinx_over_pi = sin(LAL_PI * residual) / LAL_PI * welch_factor;
    int i;
    if(kernel_length & 2)
        sinx_over_pi = -sinx_over_pi;
    for(i = -(kernel_length - 1) / 2; cached_kernel < stop; i++, sinx_over_pi = -sinx_over_pi) {
        double y = welch_factor * (i + residual);
        if(fabs(y) < 1.)
        /* the window is
         *
         * sinx_over_pi / i * (1. - y * y)
         *
         * but by putting an extra factor of welch_factor
         * into sinx_over_pi we can replace i with y,
         * and then move the factor of 1/y into the
         * parentheses to reduce the total number of
         * arithmetic operations in the loop
         */
            *cached_kernel++ = sinx_over_pi * (1. / y - y);
        else
            *cached_kernel++ = 0.;
    }
}


/** END Data Fac **/














/** GPS **/
char *XLALGPSToStr(char *s, const LIGOTimeGPS *t)
{
    char *end;
    /* so we can play with it */
    LIGOTimeGPS copy = *t;
    
    /* make sure we've got a buffer */
    
    if(!s) {
        /* 22 = 9 digits to the right of the decimal point +
         * decimal point + upto 10 digits to the left of the
         * decimal point plus an optional sign + a null */
        s = XLALMalloc(22 * sizeof(*s));
        if(!s)
            XLAL_ERROR_NULL(XLAL_EFUNC);
    }
    
    /* normalize the fractional part */
    
    while(labs(copy.gpsNanoSeconds) >= XLAL_BILLION_INT4) {
        if(copy.gpsNanoSeconds < 0) {
            copy.gpsSeconds -= 1;
            copy.gpsNanoSeconds += XLAL_BILLION_INT4;
        } else {
            copy.gpsSeconds += 1;
            copy.gpsNanoSeconds -= XLAL_BILLION_INT4;
        }
    }
    
    /* if both components are non-zero, make sure they have the same
     * sign */
    
    if(copy.gpsSeconds > 0 && copy.gpsNanoSeconds < 0) {
        copy.gpsSeconds -= 1;
        copy.gpsNanoSeconds += XLAL_BILLION_INT4;
    } else if(copy.gpsSeconds < 0 && copy.gpsNanoSeconds > 0) {
        copy.gpsSeconds += 1;
        copy.gpsNanoSeconds -= XLAL_BILLION_INT4;
    }
    
    /* print */
    
    if(copy.gpsSeconds < 0 || copy.gpsNanoSeconds < 0)
    /* number is negative */
        end = s + sprintf(s, "-%ld.%09ld", labs(copy.gpsSeconds), labs(copy.gpsNanoSeconds));
    else
    /* number is non-negative */
        end = s + sprintf(s, "%ld.%09ld", (long) copy.gpsSeconds, (long) copy.gpsNanoSeconds);
    
    /* remove trailing 0s and decimal point */
    
    while(*(--end) == '0')
        *end = 0;
    if(*end == '.')
        *end = 0;
    
    /* done */
    
    return s;
}

struct tm * XLALGPSToUTC(
                         struct tm *utc, /**< [Out] Pointer to tm struct where result is stored. */
                         INT4 gpssec /**< [In] Seconds since the GPS epoch. */
)
{
    time_t unixsec;
    int leapsec;
    int delta;
    leapsec = XLALLeapSeconds( gpssec );
    if ( leapsec < 0 )
        XLAL_ERROR_NULL( XLAL_EFUNC );
    unixsec  = gpssec - leapsec + XLAL_EPOCH_GPS_TAI_UTC; /* get rid of leap seconds */
    unixsec += XLAL_EPOCH_UNIX_GPS; /* change to unix epoch */
    memset( utc, 0, sizeof( *utc ) ); /* blank out utc structure */
    utc = gmtime_r( &unixsec, utc );
    /* now check to see if we need to add a 60th second to UTC */
    if ( ( delta = delta_tai_utc( gpssec ) ) > 0 )
        utc->tm_sec += 1; /* delta only ever is one, right?? */
    return utc;
}

int XLALLeapSeconds( INT4 gpssec /**< [In] Seconds relative to GPS epoch.*/ )
{
    int leap;
    
    if ( gpssec < leaps[0].gpssec )
    {
        fprintf(stderr, "XLAL Error - Don't know leap seconds before GPS time %d\n",
                       leaps[0].gpssec );
        XLAL_ERROR( XLAL_EDOM );
    }
    
    /* scan leap second table and locate the appropriate interval */
    for ( leap = 1; leap < numleaps; ++leap )
        if ( gpssec < leaps[leap].gpssec )
            break;
    
    return leaps[leap-1].taiutc;
}

static int isbase10(const char *s, int radix)
{
    if(*s == radix)
        s++;
    return isdigit(*s);
}


static int isbase16(const char *s, int radix)
{
    if(*s == '0') {
        s++;
        if(*s == 'X' || *s == 'x') {
            s++;
            if(*s == radix)
                s++;
            return isxdigit(*s);
        }
    }
    return 0;
}


/*
 * Check that a string contains an exponent.
 */


static int isdecimalexp(const char *s)
{
    if(*s == 'E' || *s == 'e') {
        s++;
        if(*s == '+' || *s == '-')
            s++;
        return isdigit(*s);
    }
    return 0;
}


static int isbinaryexp(const char *s)
{
    if(*s == 'P' || *s == 'p') {
        s++;
        if(*s == '+' || *s == '-')
            s++;
        return isdigit(*s);
    }
    return 0;
}

int XLALStrToGPS(LIGOTimeGPS *t, const char *nptr, char **endptr)
{
    union { char *s; const char *cs; } pconv; /* this is bad */
    int radix;
    char *digits;
    int len=0;
    int sign;
    int base;
    int radixpos;
    int exppart;
    
    
    /* retrieve the radix character */
    radix = localeconv()->decimal_point[0];
    
    /* this is bad ... there is a reason for warnings! */
    pconv.cs = nptr;
    
    /* consume leading white space */
    while(isspace(*(pconv.cs)))
        (pconv.cs)++;
    if(endptr)
        *endptr = pconv.s;
    
    /* determine the sign */
    if(*(pconv.cs) == '-') {
        sign = -1;
        (pconv.cs)++;
    } else if(*(pconv.cs) == '+') {
        sign = +1;
        (pconv.cs)++;
    } else
        sign = +1;
    
    /* determine the base */
    if(isbase16((pconv.cs), radix)) {
        base = 16;
        (pconv.cs) += 2;
    } else if(isbase10((pconv.cs), radix)) {
        base = 10;
    } else {
        /* this isn't a recognized number */
        XLALGPSSet(t, 0, 0);
        return 0;
    }
    
    /* count the number of digits including the radix but not including
     * the exponent. */
    radixpos = -1;
    switch(base) {
        case 10:
            for(len = 0; 1; len++) {
                if(isdigit((pconv.cs)[len]))
                    continue;
                if((pconv.cs)[len] == radix && radixpos < 0) {
                    radixpos = len;
                    continue;
                }
                break;
            }
            break;
            
        case 16:
            for(len = 0; 1; len++) {
                if(isxdigit((pconv.cs)[len]))
                    continue;
                if((pconv.cs)[len] == radix && radixpos < 0) {
                    radixpos = len;
                    continue;
                }
                break;
            }
            break;
    }
    
    /* copy the digits into a scratch space, removing the radix character
     * if one was found */
    if(radixpos >= 0) {
        digits = malloc(len + 1);
        memcpy(digits, (pconv.cs), radixpos);
        memcpy(digits + radixpos, (pconv.cs) + radixpos + 1, len - radixpos - 1);
        digits[len - 1] = '\0';
        (pconv.cs) += len;
        len--;
    } else {
        digits = malloc(len + 2);
        memcpy(digits, (pconv.cs), len);
        digits[len] = '\0';
        radixpos = len;
        (pconv.cs) += len;
    }
    
    /* check for and parse an exponent, performing an adjustment of the
     * radix position */
    exppart = 1;
    switch(base) {
        case 10:
            /* exponent is the number of powers of 10 */
            if(isdecimalexp((pconv.cs)))
                radixpos += strtol((pconv.cs) + 1, &pconv.s, 10);
            break;
            
        case 16:
            /* exponent is the number of powers of 2 */
            if(isbinaryexp((pconv.cs))) {
                exppart = strtol((pconv.cs) + 1, &pconv.s, 10);
                radixpos += exppart / 4;
                exppart %= 4;
                if(exppart < 0) {
                    radixpos--;
                    exppart += 4;
                }
                exppart = 1 << exppart;
            }
            break;
    }
    
    /* save end of converted characters */
    if(endptr)
        *endptr = pconv.s;
    
    /* insert the radix character, padding the scratch digits with zeroes
     * if needed */
    if(radixpos < 2) {
        digits = realloc(digits, len + 2 + (2 - radixpos));
        memmove(digits + (2 - radixpos) + 1, digits, len + 1);
        memset(digits, '0', (2 - radixpos) + 1);
        if(radixpos == 1)
            digits[1] = digits[2];
        radixpos = 2;
    } else if(radixpos > len) {
        digits = realloc(digits, radixpos + 2);
        memset(digits + len, '0', radixpos - len);
        digits[radixpos + 1] = '\0';
    } else {
        memmove(digits + radixpos + 1, digits + radixpos, len - radixpos + 1);
    }
    digits[radixpos] = radix;
    
    /* parse the integer part */
    XLALINT8NSToGPS(t, sign * strtol(digits, NULL, base) * exppart * XLAL_BILLION_INT8);
    
    /* parse the fractional part */
    switch(base) {
        case 10:
            break;
            
        case 16:
            digits[radixpos - 2] = '0';
            digits[radixpos - 1] = 'x';
            radixpos -= 2;
            break;
    XLALGPSAdd(t, sign * strtod(digits + radixpos, NULL) * exppart);
    }
    
    /* free the scratch space */
    free(digits);
    
    /* check for failures and restore errno if there weren't any */
    
    /* success */
    return 0;
}

REAL8 XLALGPSModf( REAL8 *iptr, const LIGOTimeGPS *epoch )
{
    INT8 ns = XLALGPSToINT8NS(epoch);
    INT8 rem; /* remainder */
    *iptr = ns < 0 ? -floor(-ns / XLAL_BILLION_REAL8) : floor(ns / XLAL_BILLION_REAL8);
    rem = ns - ((INT8)(*iptr) * XLAL_BILLION_INT8);
    return (REAL8)(rem) / XLAL_BILLION_REAL8;
}



/** END GPS **/
