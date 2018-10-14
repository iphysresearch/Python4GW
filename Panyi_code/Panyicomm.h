// $Id: Panyicomm.h,v 1.1.1.1 2016/12/30 06:03:09 zjcao Exp $
#include "Panyidatatypes.h"
#include <gsl/gsl_matrix.h>


REAL8 XLALGPSGetREAL8( const LIGOTimeGPS *epoch );

COMPLEX16Vector* XLALCreateCOMPLEX16Vector(UINT4 length);
void XLALDestroyCOMPLEX16Vector(COMPLEX16Vector* v);

REAL8Vector* XLALCreateREAL8Vector(UINT4 length);
void XLALDestroyREAL8Vector(REAL8Vector* v);

UINT4Vector* XLALCreateUINT4Vector(UINT4 length);
void XLALDestroyUINT4Vector(UINT4Vector* v);

REAL8Array* XLALCreateREAL8ArrayL(UINT4 ndim,...);
void XLALDestroyREAL8Array(REAL8Array* v);

REAL8VectorSequence * XLALCreateREAL8VectorSequence ( UINT4 length, UINT4 veclen );
void XLALDestroyREAL8VectorSequence ( REAL8VectorSequence * vseq );

REAL8TimeSeries *XLALCreateREAL8TimeSeries ( const char *name, const LIGOTimeGPS *epoch, REAL8 f0, REAL8 deltaT, const LALUnit *sampleUnits, int length );
void XLALDestroyREAL8TimeSeries( REAL8TimeSeries * series );

int XLALSimIMREOBFinalMassSpin(
  REAL8    *finalMass,	/**<< OUTPUT, the final mass (scaled by original total mass) */
  REAL8    *finalSpin,	/**<< OUTPUT, the final spin (scaled by final mass) */
const REAL8 	mass1,		/**<< The mass of the 1st component of the system */
const REAL8 	mass2,		/**<< The mass of the 2nd component of the system */
const REAL8 	spin1[3],	/**<< The spin of the 1st object; only needed for spin waveforms */
const REAL8 	spin2[3]	/**<< The spin of the 2nd object; only needed for spin waveforms */
);

int XLALSimIMREOBGenerateQNMFreqV2(
  COMPLEX16Vector *modefreqs, /**<< OUTPUT, complex freqs of overtones in unit of Hz */
  const REAL8      mass1,     /**<< The mass of the 1st component (in Solar masses) */
  const REAL8      mass2,     /**<< The mass of the 2nd component (in Solar masses) */
  const REAL8      spin1[3],  /**<< The spin of the 1st object; only needed for spin waveforms */
  const REAL8      spin2[3],  /**<< The spin of the 2nd object; only needed for spin waveforms */
  UINT4            l,         /**<< The l value of the mode in question */
  UINT4            m,         /**<< The m value of the mode in question */
  UINT4            nmodes    /**<< The number of overtones that should be included (max 8) */
);

 int XLALSimIMRSpinEOBCalculateSigmaStar( 
                  REAL8Vector *sigmaStar, /**<< OUTPUT, normalized (to total mass) spin of test particle */
                  REAL8        mass1,     /**<< mass 1 */
                  REAL8        mass2,     /**<< mass 2 */
                  REAL8Vector *s1,        /**<< spin vector 1 */
                  REAL8Vector *s2         /**<< spin vector 2 */);

 int XLALSimIMRSpinEOBCalculateSigmaKerr( 
                REAL8Vector *sigmaKerr, /**<< OUTPUT, normalized (to total mass) spin of deformed-Kerr */
                REAL8        mass1,     /**<< mass 1 */
                REAL8        mass2,     /**<< mass 2 */
                REAL8Vector *s1,        /**<< spin vector 1 */
                REAL8Vector *s2         /**<< spin vector 2 */);

 int XLALSimIMREOBCalcSpinFacWaveformCoefficients(
          FacWaveformCoeffs * const coeffs, /**< OUTPUT, pre-computed waveform coefficients */
          const REAL8               m1,     /**< mass 1 */
          const REAL8               m2,     /**< mass 2 */
          const REAL8               eta,    /**< symmetric mass ratio */
          const REAL8               a,      /**< Kerr spin parameter for test-particle terms */
          const REAL8               chiS,   /**< (chi1+chi2)/2 */
          const REAL8               chiA    /**< (chi1-chi2)/2 */
          );

 int XLALSimIMREOBComputeNewtonMultipolePrefixes(
              NewtonMultipolePrefixes *prefix, /**<< OUTPUT Structure containing the coeffs */
              const REAL8             m1,      /**<< Mass of first component */
              const REAL8             m2       /**<< Nass of second component */
              );

 int XLALSimIMRSpinEOBInitialConditions(
                    REAL8Vector   *initConds, /**<< OUTPUT, Initial dynamical variables */
                    const REAL8    mass1,     /**<< mass 1 */
                    const REAL8    mass2,     /**<< mass 2 */
                    const REAL8    fMin,      /**<< Initial frequency (given) */
                    const REAL8    inc,       /**<< Inclination */
                    const REAL8    spin1[],   /**<< Initial spin vector 1 */
                    const REAL8    spin2[],   /**<< Initial spin vector 2 */
                    SpinEOBParams *params     /**<< Spin EOB parameters */
                    );

 int
CalculateRotationMatrix(
                gsl_matrix *rotMatrix,  /**< OUTPUT, rotation matrix */
                gsl_matrix *rotInverse, /**< OUTPUT, rotation matrix inversed */
                REAL8       r[],        /**< position vector */
                REAL8       v[],        /**< velocity vector */
                REAL8       L[]         /**< orbital angular momentum */
                );
 inline REAL8 CalculateCrossProduct( const int i, const REAL8 a[], const REAL8 b[] );

 inline int NormalizeVector( REAL8 a[] );
 int ApplyRotationMatrix(
             gsl_matrix *rotMatrix, /**< rotation matrix */
             REAL8      a[]         /**< OUTPUT, vector rotated */
                   );

 int
XLALFindSphericalOrbit( const gsl_vector *x, /**<< Parameters requested by gsl root finder */
                       void *params,        /**<< Spin EOB parameters */
                       gsl_vector *f        /**<< Function values for the given parameters */
                     );

 REAL8 XLALSpinHcapNumDerivWRTParam(
              const INT4 paramIdx,      /**<< Index of the parameters */
              const REAL8 values[],     /**<< Dynamical variables */
              SpinEOBParams *funcParams /**<< EOB Parameters */
              );
 double GSLSpinHamiltonianWrapper( double x, void *params );

 int XLALSimIMRCalculateSpinEOBHCoeffs(
        SpinEOBHCoeffs *coeffs, /**<< OUTPUT, EOB parameters including pre-computed coefficients */
        const REAL8    eta,     /**<< symmetric mass ratio */
        const REAL8    a        /**<< Normalized deformed Kerr spin */
        );


REAL8 XLALSimIMRSpinEOBHamiltonian( 
               const REAL8    eta,                  /**<< Symmetric mass ratio */
               REAL8Vector    *x,         /**<< Position vector */
               REAL8Vector    *p,	    /**<< Momentum vector (tortoise radial component pr*) */
               REAL8Vector    *sigmaKerr, /**<< Spin vector sigma_kerr */
               REAL8Vector    *sigmaStar, /**<< Spin vector sigma_star */
               INT4                      tortoise,  /**<< flag to state whether the momentum is the tortoise co-ord */
	           SpinEOBHCoeffs *coeffs               /**<< Structure containing various coefficients */
               );


 int XLALSimIMRSpinEOBGetSpinFactorizedWaveform( 
                 COMPLEX16         * hlm,    /**< OUTPUT, hlm waveforms */
                 REAL8Vector       * values, /**< dyanmical variables */
                 const REAL8         v,               /**< velocity */
                 const REAL8         Hreal,           /**< real Hamiltonian */
                 const int          l,               /**< l mode index */
                 const int          m,               /**< m mode index */
                 SpinEOBParams     * params  /**< Spin EOB parameters */
                 );

 REAL8 XLALCalculateSphHamiltonianDeriv2(
                const int      idx1,     /**<< Derivative w.r.t. index 1 */
                const int      idx2,     /**<< Derivative w.r.t. index 2 */
                const REAL8    values[], /**<< Dynamical variables in spherical coordinates */
                SpinEOBParams *params    /**<< Spin EOB Parameters */
                );
 int SphericalToCartesian(
                 REAL8 qCart[],      /**<< OUTPUT, position vector in Cartesean coordinates */
                 REAL8 pCart[],      /**<< OUTPUT, momentum vector in Cartesean coordinates */
                 const REAL8 qSph[], /**<< position vector in spherical coordinates */
                 const REAL8 pSph[]  /**<< momentum vector in spherical coordinates */
                 );
 double GSLSpinHamiltonianDerivWrapper( double x,    /**<< Derivative at x */
                                           void  *params /**<< Function parameters */);
 REAL8 XLALInspiralSpinFactorizedFlux(
         REAL8Vector           *values, /**< dynamical variables */
         const REAL8           omega,   /**< orbital frequency */
         SpinEOBParams         *ak,     /**< physical parameters */
         const REAL8            H,      /**< real Hamiltonian */
         const int             lMax    /**< upper limit of the summation over l */
        );


 int
XLALSimIMRSpinEOBCalculateNewtonianMultipole(
                 COMPLEX16 *multipole, /**<< OUTPUT, Newtonian multipole */
                 REAL8 x,              /**<< Dimensionless parameter \f$\equiv v^2\f$ */
                 REAL8 r,       /**<< Orbital separation (units of total mass M */
                 REAL8 phi,            /**<< Orbital phase (in radians) */
                 UINT4  l,             /**<< Mode l */
                 int  m,              /**<< Mode m */
                 EOBParams *params     /**<< Pre-computed coefficients, parameters, etc. */
                 );
 int
XLALScalarSphHarmThetaPiBy2(
              COMPLEX16 *y, /**<< OUTPUT, Ylm(0,phi) */
              int l,       /**<< Mode l */
              int  m,      /**<< Mode m */
              REAL8 phi     /**<< Orbital phase (in radians) */
              );
 REAL8
XLALAssociatedLegendreXIsZero( const int l,
                             const int m );

 REAL8 XLALSimIMRSpinEOBHamiltonianDeltaR(
     SpinEOBHCoeffs *coeffs, /**<< Pre-computed coefficients which appear in the function */
     const REAL8    r,       /**<< Current orbital radius (in units of total mass) */
     const REAL8    eta,     /**<< Symmetric mass ratio */
     const REAL8    a        /**<< Normalized deformed Kerr spin */
     );

 REAL8 XLALSimIMRSpinEOBHamiltonianDeltaT( 
     SpinEOBHCoeffs *coeffs, /**<< Pre-computed coefficients which appear in the function */
     const REAL8    r,       /**<< Current orbital radius (in units of total mass) */
     const REAL8    eta,     /**<< Symmetric mass ratio */
     const REAL8    a        /**<< Normalized deformed Kerr spin */
     );
ark4GSLIntegrator *XLALAdaptiveRungeKutta4Init( int dim,
                          int (* dydt) (double t, const double y[], double dydt[], void * params),  /* These are XLAL functions! */
                          int (* stop) (double t, const double y[], double dydt[], void * params),
                          double eps_abs, double eps_rel
                              );
void XLALAdaptiveRungeKutta4Free( ark4GSLIntegrator *integrator );
 int XLALSpinAlignedHcapDerivative(
                  double  t,          /**< UNUSED */
                  const REAL8   values[],   /**< dynamical varables */
                  REAL8         dvalues[],  /**< time derivative of dynamical variables */
                  void         *funcParams  /**< EOB parameters */
                  );
 double GSLSpinAlignedHamiltonianWrapper( double x, void *params );

 int
XLALEOBSpinAlignedStopCondition(double  t,  /**< UNUSED */
                           const double values[], /**< dynamical variable values */
                           double dvalues[],      /**< dynamical variable time derivative values */
                           void *funcParams       /**< physical parameters */
                          );

int XLALAdaptiveRungeKutta4( ark4GSLIntegrator *integrator,
                       void *params,
                       REAL8 *yinit,
                       REAL8 tinit, REAL8 tend, REAL8 deltat,
                       REAL8Array **yout
                       );
int XLALSpinAlignedHiSRStopCondition(double t,  /**< UNUSED */
                          const double values[], /**< dynamical variable values */
                          double dvalues[],      /**< dynamical variable time derivative values */
                          void *funcParams       /**< physical parameters */
                         );
// scaling omega with (vy/(vy+vx))^gamma
 REAL8
XLALSimIMRSpinAlignedEOBCalcOmega(
                          const REAL8           values[],   /**<< Dynamical variables */
                          SpinEOBParams         *funcParams /**<< EOB parameters */
                          );
 REAL8
XMYSimIMRSpinAlignedEOBCalcOmega(
                          const REAL8           values[],   /**<< Dynamical variables */
                          SpinEOBParams         *funcParams /**<< EOB parameters */
                          );
LIGOTimeGPS * XLALGPSAdd( LIGOTimeGPS *epoch, REAL8 dt );
LIGOTimeGPS * XLALGPSSetREAL8( LIGOTimeGPS *epoch, REAL8 t );
LIGOTimeGPS * XLALGPSAddGPS( LIGOTimeGPS *epoch, const LIGOTimeGPS *dt );
LIGOTimeGPS * XLALGPSSet( LIGOTimeGPS *epoch, INT4 gpssec, INT8 gpsnan );
LIGOTimeGPS * XLALINT8NSToGPS( LIGOTimeGPS *epoch, INT8 ns );
INT8 XLALGPSToINT8NS( const LIGOTimeGPS *epoch );

 int XLALSimIMRGetEOBCalibratedSpinNQC( EOBNonQCCoeffs *coeffs, 
                                    INT4  l, 
                                    INT4  m, 
                                    REAL8 eta, 
                                    REAL8 a );
 int XLALSimIMRSpinEOBCalculateNQCCoefficients(
                 REAL8Vector    *amplitude,   /**<< Waveform amplitude, func of time */
                 REAL8Vector    *phase,       /**<< Waveform phase(rad), func of time */
                 REAL8Vector    *rVec,        /**<< Position-vector, function of time */
                 REAL8Vector    *prVec,       /**<< Momentum vector, function of time */
                 REAL8Vector    *orbOmegaVec, /**<< Orbital frequency, func of time */
                 INT4                      l,           /**<< Mode index l */
                 INT4                      m,           /**<< Mode index m */
                 REAL8                     timePeak,    /**<< Time of peak orbital frequency */
                 REAL8                     deltaT,      /**<< Sampling interval */
                 REAL8                     eta,         /**<< Symmetric mass ratio */
                 REAL8                     a,           /**<< Normalized spin of deformed-Kerr */
                 EOBNonQCCoeffs *coeffs       /**<< OUTPUT, NQC coefficients */);

 inline REAL8 CalculateCrossProduct( const int i, const REAL8 a[], const REAL8 b[] )
{
  return a[(i+1)%3]*b[(i+2)%3] - a[(i+2)%3]*b[(i+1)%3];
}
 inline int
NormalizeVector( REAL8 a[] )
{
  REAL8 norm = sqrt( a[0]*a[0] + a[1]*a[1] + a[2]*a[2] );

  a[0] /= norm;
  a[1] /= norm;
  a[2] /= norm;

  return 0;
}
#if 1
 inline REAL8 XLALSimIMREOBGetNRSpinPeakDeltaT( 
                 INT4 l,           /**<< Mode l */
                 INT4 m,           /**<< Mode m */
                 REAL8 eta, /**<< Symmetric mass ratio */
                 REAL8 a           /**<< Dimensionless spin */
                 )
{

  switch ( l )
  {
    case 2:
      switch ( m )
      {
        case 2:
          /* DeltaT22 defined here is a minus sign different from Eq. (33) of Taracchini et al. */
          if ( a <= 0.0 )
          {
            return 2.5;
          }
          else
          {
            return (2.5 + 1.77*a*a*a*a/(0.43655*0.43655*0.43655*0.43655)/(1.0-2.0*eta)/(1.0-2.0*eta)/(1.0-2.0*eta)/(1.0-2.0*eta));
          }
          break;
        default:
          printf("Error(%d)", XLAL_EINVAL );
      }
      break;
    default:
		printf("Error(%d)", XLAL_EINVAL );
  }

  /* We should never get here, but I expect a compiler whinge without it... */
  printf( "XLAL Error - We should never get here!!\n" );
  printf("Error(%d)", XLAL_EINVAL );
  return 0;
}
 inline REAL8 GetNRSpinPeakOmega( INT4 l, INT4  m, REAL8  eta, REAL8 a )
{
  /* Fit for HOMs missing */
  return 0.27581190323955274 + 0.19347381066059993*eta
       - 0.08898338208573725*log(1.0 - a/(1.0-2.0*eta))
       + eta*eta*(1.78832*(0.2690779744133912 + a/(2.0-4.0*eta))*(1.2056469070395925
       + a/(2.0-4.0*eta)) + 1.423734113371796*log(1.0 - a/(1.0-2.0*eta)));
}
#else
 inline REAL8 XLALSimIMREOBGetNRSpinPeakDeltaT( 
                 int l,           /**<< Mode l */
                 int m,           /**<< Mode m */
                 REAL8 eta, /**<< Symmetric mass ratio */
                 REAL8 a           /**<< Dimensionless spin */
                 )
{

  switch ( l )
  {
    case 2:
      switch ( m )
      {
        case 2:
          /* DeltaT22 defined here is a minus sign different from Eq. (33) of Taracchini et al. */
          if ( a <= 0.0 )
          {
            return 2.5;
          }
          else
          {
            return (2.5 + 1.77*a*a*a*a/(0.43655*0.43655*0.43655*0.43655)/(1.0-2.0*eta)/(1.0-2.0*eta)/(1.0-2.0*eta)/(1.0-2.0*eta));
          }
          break;
        default:
          printf("Error(%d)", XLAL_EINVAL );
      }
      break;
    default:
		printf("Error(%d)", XLAL_EINVAL );
  }

  /* We should never get here, but I expect a compiler whinge without it... */
  printf( "XLAL Error - We should never get here!!\n" );
  printf("Error(%d)", XLAL_EINVAL );
  return 0;
}

inline REAL8 GetNRSpinPeakOmega( int l, int  m, REAL8  eta, REAL8 a )
{
  /* Fit for HOMs missing */
  return 0.27581190323955274 + 0.19347381066059993*eta
       - 0.08898338208573725*log(1.0 - a/(1.0-2.0*eta))
       + eta*eta*(1.78832*(0.2690779744133912 + a/(2.0-4.0*eta))*(1.2056469070395925
       + a/(2.0-4.0*eta)) + 1.423734113371796*log(1.0 - a/(1.0-2.0*eta)));
}
#endif
 inline REAL8 GetNRSpinPeakOmegaDot( INT4 l, INT4  m, REAL8 eta, REAL8 a )
{
  /* Fit for HOMs missing */
  return 0.006075014646800278 + 0.012040017219351778*eta
       + (0.0007353536801336875 + 0.0015592659912461832*a/(1.0-2.0*eta))*log(1.0-a/(1.0-2.0*eta))
       + eta*eta*(0.03575969677378844 + (-0.011765658882139 - 0.02494825585993893*a/(1.0-2.0*eta))
       * log(1.0 - a/(1.0-2.0*eta)));
}
 int  XLALSimIMREOBNonQCCorrection(
                      COMPLEX16      *nqc,    /**<< OUTPUT, The NQC correction */
                      REAL8Vector    *values, /**<< Dynamics r, phi, pr, pphi */
                      const REAL8               omega,  /**<< Angular frequency */
                      EOBNonQCCoeffs *coeffs  /**<< NQC coefficients */
                     );

 INT4 XLALGenerateHybridWaveDerivatives (
 REAL8Vector *rwave,      /**<< OUTPUT, values of the waveform at comb points */
 REAL8Vector *dwave,      /**<< OUTPUT, 1st deriv of the waveform at comb points */
 REAL8Vector *ddwave,     /**<< OUTPUT, 2nd deriv of the waveform at comb points */
     REAL8Vector *timeVec,    /**<< Vector containing the time */
 REAL8Vector *wave,       /**<< Last part of inspiral waveform */
 REAL8Vector *matchrange, /**<< Times which determine the size of the comb */
     REAL8           dt,          /**<< Sample time step */
     REAL8           mass1,       /**<< First component mass (in Solar masses) */
     REAL8           mass2        /**<< Second component mass (in Solar masses) */
 );

COMPLEX16 XLALSpinWeightedSphericalHarmonic(
                                    REAL8 theta,  /**< polar angle (rad) */
                                    REAL8 phi,    /**< azimuthal angle (rad) */
                                    int s,        /**< spin weight */
                                    int l,        /**< mode number l */
                                    int m         /**< mode number m */
);

 int
CalculateThisMultipolePrefix(
                 COMPLEX16 *prefix, /**<< OUTPUT, Prefix value */
                 const REAL8 m1,    /**<< mass 1 */
                 const REAL8 m2,    /**<< mass 2 */
                 int l,      /**<< Mode l */
                 int m       /**<< Mode m */
                 );
 int XLALIsREAL8FailNaN(REAL8 val);
 int CartesianToSpherical(
                 REAL8 qSph[],        /**<< OUTPUT, position vector in spherical coordinates */
                 REAL8 pSph[],        /**<< OUTPUT, momentum vector in Cartesean coordinates */
                 const REAL8 qCart[], /**<< position vector in spherical coordinates */
                 const REAL8 pCart[]  /**<< momentum vector in Cartesean coordinates */
                 );
 REAL8
XLALSimIMRSpinAlignedEOBNonKeplerCoeff(
                   const REAL8           values[],   /**<< Dynamical variables */
                   SpinEOBParams         *funcParams /**<< EOB parameters */
                   );
 inline REAL8 GetNRSpinPeakAmplitude( INT4 l, INT4 m, REAL8 eta, REAL8 a );
 inline REAL8 GetNRSpinPeakADDot( INT4 l, INT4 m, REAL8 eta, REAL8 a );

COMPLEX16 XLALCOMPLEX16Rect (REAL8 x, REAL8 y);
double MYlog2(double x);
double MYcbrt(double x);
double carg (COMPLEX16 z);
COMPLEX16 cexp (COMPLEX16 a);
COMPLEX16 CX16polar(double r,double phi);
COMPLEX16 cpow(COMPLEX16 a,UINT4 n);
double cabs(COMPLEX16 z);
void Mymemory(double &hLM,const double v,const int l,SpinEOBParams* params);
void Mymemory(double &hLM,const double v,const int l,SpinEOBParams* params,const double dr);
void Mymemory(double &hLM,const double v,const int l,SpinEOBParams* params,const double s1z,const double s2z);
