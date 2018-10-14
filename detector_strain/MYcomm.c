#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// #include <time.h>
#include "time.h"
#include <stdarg.h>
#include <pthread.h>


#include "MYdatatypes.h"
#include "MYconstants.h"
#include "MYcomm.h"
#define XLAL_REAL8_FAIL_NAN ( XLALREAL8FailNaN() )

#define XLAL_PRINT_ERROR(...) \
    XLALPrintErrorMessage(__func__, __FILE__, __LINE__, __VA_ARGS__)
#define XLAL_PRINT_WARNING(...) \
    XLALPrintWarningMessage(__func__, __FILE__, __LINE__, __VA_ARGS__)


#define _XLAL_ERROR_IMPL_(statement, errnum, ...) \
do { \
char _XLAL_ERROR_IMPL_buf_[1024]; \
snprintf(_XLAL_ERROR_IMPL_buf_, sizeof(_XLAL_ERROR_IMPL_buf_), "X" __VA_ARGS__); \
if (_XLAL_ERROR_IMPL_buf_[1] != 0) { \
XLAL_PRINT_ERROR("%s", &_XLAL_ERROR_IMPL_buf_[1]); \
} \
XLALError(__func__, __FILE__, __LINE__, errnum); \
statement; \
} while (0)

#define XLAL_ERROR_REAL8(...) _XLAL_ERROR_IMPL_(return XLAL_REAL8_FAIL_NAN, __VA_ARGS__)
static int lalDebugLevel = 1;
static int xlalErrno = 1;
pthread_once_t xlalErrorHandlerKeyOnce = PTHREAD_ONCE_INIT;
pthread_key_t xlalErrorHandlerKey;



void XLALPrintWarningMessage(const char *func, const char *file, int line,
                             const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    XLALVPrintWarningMessage(func, file, line, fmt, ap);
    va_end(ap);
    return;
}

static void XLALCreateErrorHandlerKey(void)
{
    pthread_key_create(&xlalErrorHandlerKey, XLALDestroyErrorHandlerPtr);
    return;
}

static void XLALDestroyErrorHandlerPtr(void *xlalErrorHandlerPtr)
{
    free(xlalErrorHandlerPtr);
    return;
}

void XLALVPrintWarningMessage(const char *func, const char *file, int line,
                              const char *fmt, va_list ap)
{
    XLALPrintWarning("XLAL Warning");
    if (func && *func)
        XLALPrintWarning(" - %s", func);
    if (file && *file)
        XLALPrintWarning(" (%s:%d)", file, line);
    XLALPrintWarning(": ");
    XLALVPrintWarning(fmt, ap);
    XLALPrintWarning("\n");
    return;
}

int XLALSetErrno(int errnum)
{
    if (errnum == 0) {
        xlalErrno = 0;
        return xlalErrno;
    }
    
    /*
     * if this is an error indicating an internal error then set the bit
     * that indicates this; otherwise, xlalErrno should presumably be zero
     */
    if (errnum & XLAL_EFUNC) {
        xlalErrno |= XLAL_EFUNC;        /* make sure XLAL_EFUNC bit is set */
        return xlalErrno;
    }
    
    /*
     * if xlalErrno is not zero, probably forgot to deal with previous
     * error
     */
    if (xlalErrno)
        XLAL_PRINT_WARNING("Ignoring previous error (xlalErrno=%d) %s\n",
                           xlalErrno, XLALErrorString(xlalErrno));
    xlalErrno = errnum;
    return xlalErrno;
}
void XLALDefaultErrorHandler(const char *func, const char *file, int line,
                             int errnum)
{
    XLALPerror(func, file, line, errnum);
    return;
}

void (*lalAbortHook) (const char *, ...) = LALAbort;
void LALAbort(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    (void) vfprintf(stderr, fmt, ap);
    va_end(ap);
#if defined(HAVE_BACKTRACE) && defined(BACKTRACE_LEVELMAX)
    void *callstack[BACKTRACE_LEVELMAX];
    size_t frames = backtrace(callstack, BACKTRACE_LEVELMAX);
    fprintf(stderr, "backtrace:\n");
    backtrace_symbols_fd(callstack, frames, fileno(stderr));
#endif
    abort();
}

/**
XLALErrorHandlerType **XLALGetErrorHandlerPtr(void)
{
    XLALErrorHandlerType **xlalErrorHandlerPtr;
    

    pthread_once(&xlalErrorHandlerKeyOnce, XLALCreateErrorHandlerKey);
    

    xlalErrorHandlerPtr = pthread_getspecific(xlalErrorHandlerKey);
    if (!xlalErrorHandlerPtr) {
        xlalErrorHandlerPtr = malloc(sizeof(*xlalErrorHandlerPtr));
        if (!xlalErrorHandlerPtr)
            lalAbortHook
            ("could not set xlal error handler: malloc failed\n");
        *xlalErrorHandlerPtr = NULL;
        if (pthread_setspecific(xlalErrorHandlerKey, xlalErrorHandlerPtr))
            lalAbortHook
            ("could not set xlal error handler: pthread_setspecific failed\n");
    }
    return xlalErrorHandlerPtr;
}
**/

void XLALError(const char *func, const char *file, int line, int errnum)
{
    XLALSetErrno(errnum);
    if (!XLALErrorHandler)
        XLALErrorHandler = XLALDefaultErrorHandler;
    XLALErrorHandler(func, file, line, xlalErrno);
    return;
}

void XLALPrintErrorMessage(const char *func, const char *file, int line,
                           const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    XLALVPrintErrorMessage(func, file, line, fmt, ap);
    va_end(ap);
    return;
}

int LALPrintError(const char *fmt, ...)
{
    int n;
    va_list ap;
    va_start(ap, fmt);
    n = vfprintf(stderr, fmt, ap);
    va_end(ap);
    return n;
}

int LALError(LALStatus * status, const char *statement)
{
    int n = 0;
    if (lalDebugLevel & LALERROR) {
        n = LALPrintError
        ("Error[%d] %d: function %s, file %s, line %d, %s\n"
         "        %s %s\n", status->level, status->statusCode,
         status->function, status->file, status->line, status->Id,
         statement ? statement : "", status->statusDescription);
    }
    return n;
}

void XLALPerror(const char *func, const char *file, int line, int code)
{
    if (code > 0)
        XLALPrintError("XLAL Error");
    else
        XLALPrintError("XLAL Result");
    if (func && *func)
        XLALPrintError(" - %s", func);
    if (file && *file)
        XLALPrintError(" (%s:%d)", file, line);
    XLALPrintError(": %s\n", XLALErrorString(code));
    return;
}
XLALErrorHandlerType *xlalErrorHandlerGlobal = NULL;

XLALErrorHandlerType **XLALGetErrorHandlerPtr(void)
{
    return &xlalErrorHandlerGlobal;
}

const char *XLALErrorString(int code)
{
    
    if (code <= 0) {    /* this is a return code, not an error number */
        if (code == 0)
            return "Success";
        else if (code == -1)
            return "Failure";
        else
            return "Unknown return code";
    }
    
    /* check to see if an internal function call has failed, but the error
     * number was not "or"ed against the mask XLAL_EFUNC */
    if (code == XLAL_EFUNC)
        return "Internal function call failed";
    
    /* use this to report error strings... deals with possible mask for
     * errors arising from internal function calls */
# define XLAL_ERROR_STRING(s) \
( ( code & XLAL_EFUNC ) ? "Internal function call failed: " s : (const char *) s )
    switch (code & ~XLAL_EFUNC) {
            /* these are standard error numbers */
        case XLAL_ENOENT:
            return XLAL_ERROR_STRING("No such file or directory");
        case XLAL_EIO:
            return XLAL_ERROR_STRING("I/O error");
        case XLAL_ENOMEM:
            return XLAL_ERROR_STRING("Memory allocation error");
        case XLAL_EFAULT:
            return XLAL_ERROR_STRING("Invalid pointer");
        case XLAL_EINVAL:
            return XLAL_ERROR_STRING("Invalid argument");
        case XLAL_EDOM:
            return XLAL_ERROR_STRING("Input domain error");
        case XLAL_ERANGE:
            return XLAL_ERROR_STRING("Output range error");
        case XLAL_ENOSYS:
            return XLAL_ERROR_STRING("Function not implemented");
            
            /* extended error numbers start at 128 ...
             * should be beyond normal errnos */
            
            /* these are common errors for XLAL functions */
        case XLAL_EFAILED:
            return XLAL_ERROR_STRING("Generic failure");
        case XLAL_EBADLEN:
            return XLAL_ERROR_STRING("Inconsistent or invalid vector length");
        case XLAL_ESIZE:
            return XLAL_ERROR_STRING("Wrong size");
        case XLAL_EDIMS:
            return XLAL_ERROR_STRING("Wrong dimensions");
        case XLAL_ETYPE:
            return XLAL_ERROR_STRING("Wrong or unknown type");
        case XLAL_ETIME:
            return XLAL_ERROR_STRING("Invalid time");
        case XLAL_EFREQ:
            return XLAL_ERROR_STRING("Invalid freqency");
        case XLAL_EUNIT:
            return XLAL_ERROR_STRING("Invalid units");
        case XLAL_ENAME:
            return XLAL_ERROR_STRING("Wrong name");
        case XLAL_EDATA:
            return XLAL_ERROR_STRING("Invalid data");
            
            /* user-defined errors */
        case XLAL_EUSR0:
            return XLAL_ERROR_STRING("User-defined error 0");
        case XLAL_EUSR1:
            return XLAL_ERROR_STRING("User-defined error 1");
        case XLAL_EUSR2:
            return XLAL_ERROR_STRING("User-defined error 2");
        case XLAL_EUSR3:
            return XLAL_ERROR_STRING("User-defined error 3");
        case XLAL_EUSR4:
            return XLAL_ERROR_STRING("User-defined error 4");
        case XLAL_EUSR5:
            return XLAL_ERROR_STRING("User-defined error 5");
        case XLAL_EUSR6:
            return XLAL_ERROR_STRING("User-defined error 6");
        case XLAL_EUSR7:
            return XLAL_ERROR_STRING("User-defined error 7");
        case XLAL_EUSR8:
            return XLAL_ERROR_STRING("User-defined error 8");
        case XLAL_EUSR9:
            return XLAL_ERROR_STRING("User-defined error 9");
            
            /* external or internal errors */
        case XLAL_ESYS:
            return XLAL_ERROR_STRING("System error");
        case XLAL_EERR:
            return XLAL_ERROR_STRING("Internal error");
            
            /* specific mathematical and numerical errors start at 256 */
            
            /* IEEE floating point errors */
        case XLAL_EFPINVAL:
            return
            XLAL_ERROR_STRING
            ("Invalid floating point operation, eg sqrt(-1), 0/0");
        case XLAL_EFPDIV0:
            return XLAL_ERROR_STRING("Division by zero floating point error");
        case XLAL_EFPOVRFLW:
            return XLAL_ERROR_STRING("Floating point overflow error");
        case XLAL_EFPUNDFLW:
            return XLAL_ERROR_STRING("Floating point underflow error");
        case XLAL_EFPINEXCT:
            return XLAL_ERROR_STRING("Floating point inexact error");
            
            /* numerical algorithm errors */
        case XLAL_EMAXITER:
            return XLAL_ERROR_STRING("Exceeded maximum number of iterations");
        case XLAL_EDIVERGE:
            return XLAL_ERROR_STRING("Series is diverging");
        case XLAL_ESING:
            return XLAL_ERROR_STRING("Apparent singularity detected");
        case XLAL_ETOL:
            return XLAL_ERROR_STRING("Failed to reach specified tolerance");
        case XLAL_ELOSS:
            return XLAL_ERROR_STRING("Loss of accuracy");
            
            /* unrecognized error number */
        default:
            return "Unknown error";
    }
# undef XLAL_ERROR_STRING
    return NULL;        /* impossible to get here */
}

int XLALVPrintError(const char *fmt, va_list ap)
{
    return (lalDebugLevel & LALERROR) ? vfprintf(stderr, fmt, ap) : 0;
}

int XLALPrintError(const char *fmt, ...)
{
    int n = 0;
    va_list ap;
    va_start(ap, fmt);
    n = XLALVPrintError(fmt, ap);
    va_end(ap);
    return n;
}


/* XLAL error handler to abort on error and print a backtrace (if possible). */
void XLALBacktraceErrorHandler(const char *func, const char *file,
                               int line, int errnum)
{
    XLALPerror(func, file, line, errnum);
#if defined(HAVE_BACKTRACE) && defined(BACKTRACE_LEVELMAX)
    void *callstack[BACKTRACE_LEVELMAX];
    size_t frames = backtrace(callstack, BACKTRACE_LEVELMAX);
    fprintf(stderr, "backtrace:\n");
    backtrace_symbols_fd(callstack, frames, fileno(stderr));
#endif
    abort();
}

/* Set the XLAL error handler to newHandler; return the old handler. */
XLALErrorHandlerType *XLALSetErrorHandler(XLALErrorHandlerType *
                                          newHandler)
{
    XLALErrorHandlerType *oldHandler;
    oldHandler = XLALErrorHandler;
    XLALErrorHandler = newHandler;
    return oldHandler;
}

LIGOTimeGPS * XLALGPSAddGPS( LIGOTimeGPS *epoch, const LIGOTimeGPS *dt )
{
    return XLALINT8NSToGPS( epoch, XLALGPSToINT8NS( epoch ) + XLALGPSToINT8NS( dt ) );
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

REAL8 XLALGPSDiff( const LIGOTimeGPS *t1, const LIGOTimeGPS *t0 )
{
    double hi = t1->gpsSeconds - t0->gpsSeconds;
    double lo = t1->gpsNanoSeconds - t0->gpsNanoSeconds;
    return hi + lo / XLAL_BILLION_REAL8;
}


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

REAL8 XLALGPSGetREAL8( const LIGOTimeGPS *epoch )
{
    return epoch->gpsSeconds + (epoch->gpsNanoSeconds / XLAL_BILLION_REAL8);
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


int
LALgetopt_long_only (int argc, char *const *argv, const char *options,
                     const struct LALoption *long_options, int *opt_index)
{
    return _getopt_internal (argc, argv, options, long_options, opt_index, 1);
}

int XLALStrToGPS(LIGOTimeGPS *t, const char *nptr, char **endptr)
{
    union { char *s; const char *cs; } pconv; /* this is bad */
    int olderrno;
    int radix;
    char *digits;
    int len=0;
    int sign;
    int base;
    int radixpos;
    int exppart;
    
    /* save and clear C library errno so we can check for failures */
    olderrno = errno;
    errno = 0;
    
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
    if(errno != ERANGE) {
        switch(base) {
            case 10:
                break;
                
            case 16:
                digits[radixpos - 2] = '0';
                digits[radixpos - 1] = 'x';
                radixpos -= 2;
                break;
        }
        XLALGPSAdd(t, sign * strtod(digits + radixpos, NULL) * exppart);
    }
    
    /* free the scratch space */
    free(digits);
    
    /* check for failures and restore errno if there weren't any */
    if(errno == ERANGE)
        XLAL_ERROR(XLAL_ERANGE);
    errno = olderrno;
    
    /* success */
    return 0;
}



void XLALComputeDetAMResponse(
                              double *fplus,        /**< Returned value of F+ */
                              double *fcross,        /**< Returned value of Fx */
                              const REAL4 D[3][3],    /**< Detector response 3x3 matrix */
                              const double ra,    /**< Right ascention of source (radians) */
                              const double dec,    /**< Declination of source (radians) */
                              const double psi,    /**< Polarization angle of source (radians) */
                              const double gmst    /**< Greenwich mean sidereal time (radians) */
)
{
    int i;
    double X[3];
    double Y[3];
    
    /* Greenwich hour angle of source (radians). */
    const double gha = gmst - ra;
    
    /* pre-compute trig functions */
    const double cosgha = cos(gha);
    const double singha = sin(gha);
    const double cosdec = cos(dec);
    const double sindec = sin(dec);
    const double cospsi = cos(psi);
    const double sinpsi = sin(psi);
    
    /* Eq. (B4) of [ABCF].  Note that dec = pi/2 - theta, and gha =
     * -phi where theta and phi are the standard spherical coordinates
     * used in that paper. */
    X[0] = -cospsi * singha - sinpsi * cosgha * sindec;
    X[1] = -cospsi * cosgha + sinpsi * singha * sindec;
    X[2] =  sinpsi * cosdec;
    
    /* Eq. (B5) of [ABCF].  Note that dec = pi/2 - theta, and gha =
     * -phi where theta and phi are the standard spherical coordinates
     * used in that paper. */
    Y[0] =  sinpsi * singha - cospsi * cosgha * sindec;
    Y[1] =  sinpsi * cosgha + cospsi * singha * sindec;
    Y[2] =  cospsi * cosdec;
    
    /* Now compute Eq. (B7) of [ABCF] for each polarization state, i.e.,
     * with s+=1 and sx=0 to get F+, with s+=0 and sx=1 to get Fx */
    *fplus = *fcross = 0.0;
    for(i = 0; i < 3; i++) {
        const double DX = D[i][0] * X[0] + D[i][1] * X[1] + D[i][2] * X[2];
        const double DY = D[i][0] * Y[0] + D[i][1] * Y[1] + D[i][2] * Y[2];
        *fplus  += X[i] * DX - Y[i] * DY;
        *fcross += X[i] * DY + Y[i] * DX;
    }
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
{
    return XLALGreenwichSiderealTime(gpstime, 0.0);
}

void LALCheckMemoryLeaks(void)
{
    int leak = 0;
    if (!(lalDebugLevel & LALMEMDBGBIT)) {
        return;
    }
    
    /* alloc_data_len should be zero */
    if ((lalDebugLevel & LALMEMTRKBIT) && alloc_data_len > 0) {
        XLALPrintError("LALCheckMemoryLeaks: allocation list\n");
        for (int k = 0; k < alloc_data_len; ++k) {
            if (alloc_data[k] != NULL && alloc_data[k] != DEL) {
                XLALPrintError("%p: %zu bytes (%s:%d)\n", alloc_data[k]->addr,
                               alloc_data[k]->size, alloc_data[k]->file,
                               alloc_data[k]->line);
            }
        }
        leak = 1;
    }
    
    /* lalMallocTotal and alloc_n should be zero */
    if ((lalDebugLevel & LALMEMPADBIT) && (lalMallocTotal || alloc_n)) {
        XLALPrintError("LALCheckMemoryLeaks: %d allocs, %zd bytes\n", alloc_n, lalMallocTotal);
        leak = 1;
    }
    
    if (leak) {
        lalRaiseHook(SIGSEGV, "LALCheckMemoryLeaks: memory leak\n");
    } else if (lalDebugLevel & LALMEMINFOBIT) {
        XLALPrintError
        ("LALCheckMemoryLeaks meminfo: no memory leaks detected\n");
    }
    
    return;
}


void XLALDestroyREAL8TimeSeries( REAL8TimeSeries * series )
{
    if(series)
        XLALDestroyREAL8Vector(series->data);
    free(series);
}

LIGOTimeGPS * XLALGPSAdd( LIGOTimeGPS *epoch, REAL8 dt )
{
    LIGOTimeGPS dt_gps;
    if(!XLALGPSSetREAL8(&dt_gps, dt))
        XLAL_ERROR_NULL(XLAL_EFUNC);
    return XLALGPSAddGPS(epoch, &dt_gps);
}

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



REAL8TimeSeries *XLALSimDetectorStrainREAL8TimeSeries(
                                                      const REAL8TimeSeries *hplus,
                                                      const REAL8TimeSeries *hcross,
                                                      REAL8 right_ascension,
                                                      REAL8 declination,
                                                      REAL8 psi,
                                                      const LALDetector *detector
                                                      )
{
    /* samples */
    const int kernel_length = 19;
    /* 0.25 s or 1 sample whichever is larger */
    const unsigned det_resp_interval = round(0.25 / hplus->deltaT) < 1 ? 1 : round(.25 / hplus->deltaT);
    LALREAL8TimeSeriesInterp *hplusinterp = NULL;
    LALREAL8TimeSeriesInterp *hcrossinterp = NULL;
    double fplus = XLAL_REAL8_FAIL_NAN;
    double fcross = XLAL_REAL8_FAIL_NAN;
    double geometric_delay = XLAL_REAL8_FAIL_NAN;
    LIGOTimeGPS t;    /* a time */
    double dt;    /* an offset */
    char *name;
    REAL8TimeSeries *h = NULL;
    unsigned i;
    
    /* check input */
    
    LAL_CHECK_VALID_SERIES(hplus, NULL);
    LAL_CHECK_VALID_SERIES(hcross, NULL);
    LAL_CHECK_CONSISTENT_TIME_SERIES(hplus, hcross, NULL);
    /* test that the input's length can be treated as a signed valued
     * without overflow, and that adding the kernel length plus the an
     * Earth diameter's worth of samples won't overflow */
    if((int) hplus->data->length < 0 || (int) (hplus->data->length + kernel_length + 2.0 * LAL_REARTH_SI / LAL_C_SI / hplus->deltaT) < 0) {
        XLALPrintError("%s(): error: input series too long\n", __func__);
        XLAL_ERROR_NULL(XLAL_EBADLEN);
    }
    
    /* generate name */
    
    name = XLALMalloc(strlen(detector->frDetector.prefix) + 11);
    if(!name)
        goto error;
    sprintf(name, "%s injection", detector->frDetector.prefix);
    
    /* allocate output time series.  the time series' duration is
     * adjusted to account for Doppler-induced dilation of the
     * waveform, and is padded to accomodate ringing of the
     * interpolation kernel.  the sign of dt follows from the
     * observation that time stamps in the output time series are
     * mapped to time stamps in the input time series by adding the
     * output of XLALTimeDelayFromEarthCenter(), so if that number is
     * larger at the start of the waveform than at the end then the
     * output time series must be longer than the input.  (the Earth's
     * rotation is not super-luminal so we don't have to account for
     * time reversals in the mapping) */
    
    /* time (at geocentre) of end of waveform */
    t = hplus->epoch;
    if(!XLALGPSAdd(&t, hplus->data->length * hplus->deltaT)) {
        XLALFree(name);
        goto error;
    }
    /* change in geometric delay from start to end */
    dt = XLALTimeDelayFromEarthCenter(detector->location, right_ascension, declination, &hplus->epoch) - XLALTimeDelayFromEarthCenter(detector->location, right_ascension, declination, &t);
    /* allocate */
    h = XLALCreateREAL8TimeSeries(name, &hplus->epoch, hplus->f0, hplus->deltaT, &hplus->sampleUnits, (int) hplus->data->length + kernel_length - 1 + ceil(dt / hplus->deltaT));
    XLALFree(name);
    if(!h)
        goto error;
    
    /* shift the epoch so that the start of the input time series
     * passes through this detector at the time of the sample at offset
     * (kernel_length-1)/2   we assume the kernel is sufficiently short
     * that it doesn't matter whether we compute the geometric delay at
     * the start or middle of the kernel. */
    
    geometric_delay = XLALTimeDelayFromEarthCenter(detector->location, right_ascension, declination, &h->epoch);
    if(XLAL_IS_REAL8_FAIL_NAN(geometric_delay))
        goto error;
    if(!XLALGPSAdd(&h->epoch, geometric_delay - (kernel_length - 1) / 2 * h->deltaT))
        goto error;
    
    /* round epoch to an integer sample boundary so that
     * XLALSimAddInjectionREAL8TimeSeries() can use no-op code path.
     * note:  we assume a sample boundary occurs on the integer second.
     * if this isn't the case (e.g, time-shifted injections or some GEO
     * data) that's OK, but we might end up paying for a second
     * sub-sample time shift when adding the the time series into the
     * target data stream in XLALSimAddInjectionREAL8TimeSeries().
     * don't bother checking for errors, this is changing the timestamp
     * by less than 1 sample, if we're that close to overflowing it'll
     * be caught in the loop below. */
    
    dt = XLALGPSModf(&dt, &h->epoch);
    XLALGPSAdd(&h->epoch, round(dt / h->deltaT) * h->deltaT - dt);
    
    /* project + and x time series onto detector */
    
    hplusinterp = XLALREAL8TimeSeriesInterpCreate(hplus, kernel_length);
    hcrossinterp = XLALREAL8TimeSeriesInterpCreate(hcross, kernel_length);
    if(!hplusinterp || !hcrossinterp)
        goto error;
    
    for(i = 0; i < h->data->length; i++) {
        /* time of sample in detector */
        t = h->epoch;
        if(!XLALGPSAdd(&t, i * h->deltaT))
            goto error;
        
        /* detector's response and geometric delay from geocentre
         * at that time */
        if(!(i % det_resp_interval)) {
            XLALComputeDetAMResponse(&fplus, &fcross, (const REAL4(*)[3])detector->response, right_ascension, declination, psi, XLALGreenwichMeanSiderealTime(&t));
            geometric_delay = -XLALTimeDelayFromEarthCenter(detector->location, right_ascension, declination, &t);
        }
        if(XLAL_IS_REAL8_FAIL_NAN(fplus) || XLAL_IS_REAL8_FAIL_NAN(fcross) || XLAL_IS_REAL8_FAIL_NAN(geometric_delay))
            goto error;
        
        /* time of sample at geocentre */
        if(!XLALGPSAdd(&t, geometric_delay))
            goto error;
        
        /* evaluate linear combination of interpolators */
        h->data->data[i] = fplus * XLALREAL8TimeSeriesInterpEval(hplusinterp, &t, 0) + fcross * XLALREAL8TimeSeriesInterpEval(hcrossinterp, &t, 0);
        if(XLAL_IS_REAL8_FAIL_NAN(h->data->data[i]))
            goto error;
    }
    XLALREAL8TimeSeriesInterpDestroy(hplusinterp);
    XLALREAL8TimeSeriesInterpDestroy(hcrossinterp);
    
    /* done */
    
    return h;
    
error:
    XLALREAL8TimeSeriesInterpDestroy(hplusinterp);
    XLALREAL8TimeSeriesInterpDestroy(hcrossinterp);
    XLALDestroyREAL8TimeSeries(h);
    XLAL_ERROR_NULL(XLAL_EFUNC);
}

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
        XLALPrintError( "%s(): overflow: %" LAL_INT8_FORMAT, __func__, ns );
        XLAL_ERROR_NULL( XLAL_EDOM );
    }
    return epoch;
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

int XLALPrintWarning(const char *fmt, ...)
{
    int n = 0;
    va_list ap;
    va_start(ap, fmt);
    n = XLALVPrintWarning(fmt, ap);
    va_end(ap);
    return n;
}

int XLALVPrintWarning(const char *fmt, va_list ap)
{
    return (lalDebugLevel & LALWARNING) ? vfprintf(stderr, fmt, ap) : 0;
}
void XLALVPrintErrorMessage(const char *func, const char *file, int line,
                            const char *fmt, va_list ap)
{
    XLALPrintError("XLAL Error");
    if (func && *func)
        XLALPrintError(" - %s", func);
    if (file && *file)
        XLALPrintError(" (%s:%d)", file, line);
    XLALPrintError(": ");
    XLALVPrintError(fmt, ap);
    XLALPrintError("\n");
    return;
}

