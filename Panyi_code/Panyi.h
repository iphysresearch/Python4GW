// $Id: Panyi.h,v 1.1.1.1 2016/12/30 06:03:09 zjcao Exp $

#include "Panyidatatypes.h"

int XLALSimInspiralChooseTDWaveform(
    REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
    REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
    REAL8 phiRef,                               /**< reference orbital phase (rad) */
    REAL8 deltaT,                               /**< sampling interval (s) */
    REAL8 m1,                                   /**< mass of companion 1 (kg) */
    REAL8 m2,                                   /**< mass of companion 2 (kg) */
    REAL8 S1x,                                  /**< x-component of the dimensionless spin of object 1 */
    REAL8 S1y,                                  /**< y-component of the dimensionless spin of object 1 */
    REAL8 S1z,                                  /**< z-component of the dimensionless spin of object 1 */
    REAL8 S2x,                                  /**< x-component of the dimensionless spin of object 2 */
    REAL8 S2y,                                  /**< y-component of the dimensionless spin of object 2 */
    REAL8 S2z,                                  /**< z-component of the dimensionless spin of object 2 */
    REAL8 f_min,                                /**< starting GW frequency (Hz) */
    REAL8 r,                                    /**< distance of source (m) */
    REAL8 i                                   /**< inclination of source (rad) */
    );


int XLALSimIMRSpinAlignedEOBWaveform(
        REAL8TimeSeries **hplus,     /**<< OUTPUT, +-polarization waveform */
        REAL8TimeSeries **hcross,    /**<< OUTPUT, x-polarization waveform */
        const REAL8     phiC,        /**<< coalescence orbital phase (rad) */ 
        REAL8           deltaT,      /**<< sampling time step */
        const REAL8     m1SI,        /**<< mass-1 in SI unit */ 
        const REAL8     m2SI,        /**<< mass-2 in SI unit */
        const REAL8     fMin,        /**<< starting frequency (Hz) */
        const REAL8     r,           /**<< distance in SI unit */
        const REAL8     inc,         /**<< inclination angle */
        const REAL8     spin1z,      /**<< z-component of spin-1, dimensionless */
        const REAL8     spin2z       /**<< z-component of spin-2, dimensionless */
     );

  INT4 XLALSimIMREOBHybridRingdownWave(
   REAL8Vector          *rdwave1,   /**<< OUTPUT, Real part of ringdown waveform */
   REAL8Vector          *rdwave2,   /**<< OUTPUT, Imag part of ringdown waveform */
   const REAL8           dt,        /**<< Sampling interval */
   const REAL8           mass1,     /**<< First component mass (in Solar masses) */
   const REAL8           mass2,     /**<< Second component mass (in Solar masses) */
   REAL8VectorSequence  *inspwave1, /**<< Values and derivs of real part inspiral waveform */
   REAL8VectorSequence  *inspwave2, /**<< Values and derivs of imag part inspiral waveform */
   COMPLEX16Vector      *modefreqs, /**<< Complex freqs of ringdown (scaled by total mass) */
   REAL8Vector          *matchrange /**<< Times which determine the comb of ringdown attachment */
   );
 INT4 XLALSimIMREOBHybridAttachRingdown(
   REAL8Vector *signal1,    /**<< OUTPUT, Real of inspiral waveform to which we attach ringdown */
   REAL8Vector *signal2,    /**<< OUTPUT, Imag of inspiral waveform to which we attach ringdown */
   const INT4   l,          /**<< Current mode l */
   const INT4   m,          /**<< Current mode m */
   const REAL8  dt,         /**<< Sample time step (in seconds) */
   const REAL8  mass1,      /**<< First component mass (in Solar masses) */
   const REAL8  mass2,      /**<< Second component mass (in Solar masses) */
   const REAL8  spin1x,     /**<<The spin of the first object; only needed for spin waveforms */
   const REAL8  spin1y,     /**<<The spin of the first object; only needed for spin waveforms */
   const REAL8  spin1z,     /**<<The spin of the first object; only needed for spin waveforms */
   const REAL8  spin2x,     /**<<The spin of the second object; only needed for spin waveforms */
   const REAL8  spin2y,     /**<<The spin of the second object; only needed for spin waveforms */
   const REAL8  spin2z,     /**<<The spin of the second object; only needed for spin waveforms */
   REAL8Vector *timeVec,    /**<< Vector containing the time values */
   REAL8Vector *matchrange /**<< Time values chosen as points for performing comb matching */
   );
