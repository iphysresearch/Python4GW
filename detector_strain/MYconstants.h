// $Id: PanyiLALConstants.h,v 1.1.1.1 2016/12/30 06:03:09 zjcao Exp $
/*
*  Copyright (C) 2007 Jolien Creighton
*
*  This program is free software; you can redistribute it and/or modify
*  it under the terms of the GNU General Public License as published by
*  the Free Software Foundation; either version 2 of the License, or
*  (at your option) any later version.
*
*  This program is distributed in the hope that it will be useful,
*  but WITHOUT ANY WARRANTY; without even the implied warranty of
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*  GNU General Public License for more details.
*
*  You should have received a copy of the GNU General Public License
*  along with with program; see the file COPYING. If not, write to the
*  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
*  MA  02111-1307  USA
*/

/**
 * \addtogroup LALConstants_h
 * \author Creighton, T. D.
 * \brief Provides standard numerical constants for LAL.
 *
 * This header defines a number of useful numerical constants
 * for use in LAL routines.  These constants come in three basic
 * flavours: arithmetic and mathematical constants, fundamental (or
 * defined) physical constants, and measured astrophysical and
 * cosmological parameters.
 *
 * Note that this header is not included automatically by the header
 * <tt>LALStdlib.h</tt>.  Include it explicitly if you need any of these
 * constants.
 */
/*@{*/

#ifndef _LALCONSTANTS_H
#define _LALCONSTANTS_H

#ifdef  __cplusplus
extern "C" {
#endif
    
/** Detector Constants **/

#define LAL_TAMA_300_DETECTOR_NAME                   "TAMA_300"    /**< TAMA_300 detector name string */
#define LAL_TAMA_300_DETECTOR_PREFIX                 "T1"    /**< TAMA_300 detector prefix string */
#define LAL_TAMA_300_DETECTOR_LONGITUDE_RAD          2.43536359469    /**< TAMA_300 vertex longitude (rad) */
#define LAL_TAMA_300_DETECTOR_LATITUDE_RAD           0.62267336022    /**< TAMA_300 vertex latitude (rad) */
#define LAL_TAMA_300_DETECTOR_ELEVATION_SI           90    /**< TAMA_300 vertex elevation (m) */
#define LAL_TAMA_300_DETECTOR_ARM_X_AZIMUTH_RAD      4.71238898038    /**< TAMA_300 x arm azimuth (rad) */
#define LAL_TAMA_300_DETECTOR_ARM_Y_AZIMUTH_RAD      3.14159265359    /**< TAMA_300 y arm azimuth (rad) */
#define LAL_TAMA_300_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< TAMA_300 x arm altitude (rad) */
#define LAL_TAMA_300_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< TAMA_300 y arm altitude (rad) */
#define LAL_TAMA_300_DETECTOR_ARM_X_MIDPOINT_SI      150.00000000000    /**< TAMA_300 x arm midpoint (m) */
#define LAL_TAMA_300_DETECTOR_ARM_Y_MIDPOINT_SI      150.00000000000    /**< TAMA_300 y arm midpoint (m) */
#define LAL_TAMA_300_VERTEX_LOCATION_X_SI            -3.94640899111e+06    /**< TAMA_300 x-component of vertex location in Earth-centered frame (m) */
#define LAL_TAMA_300_VERTEX_LOCATION_Y_SI            3.36625902802e+06    /**< TAMA_300 y-component of vertex location in Earth-centered frame (m) */
#define LAL_TAMA_300_VERTEX_LOCATION_Z_SI            3.69915069233e+06    /**< TAMA_300 z-component of vertex location in Earth-centered frame (m) */
#define LAL_TAMA_300_ARM_X_DIRECTION_X               0.64896940530    /**< TAMA_300 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_TAMA_300_ARM_X_DIRECTION_Y               0.76081450498    /**< TAMA_300 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_TAMA_300_ARM_X_DIRECTION_Z               -0.00000000000    /**< TAMA_300 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_TAMA_300_ARM_Y_DIRECTION_X               -0.44371376921    /**< TAMA_300 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_TAMA_300_ARM_Y_DIRECTION_Y               0.37848471479    /**< TAMA_300 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_TAMA_300_ARM_Y_DIRECTION_Z               -0.81232223390    /**< TAMA_300 z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name VIRGO 3km Interferometric Detector constants
     * The following constants describe the location and geometry of the
     * VIRGO 3km Interferometric Detector.
     */
    /*@{*/
#define LAL_VIRGO_DETECTOR_NAME                   "VIRGO"    /**< VIRGO detector name string */
#define LAL_VIRGO_DETECTOR_PREFIX                 "V1"    /**< VIRGO detector prefix string */
#define LAL_VIRGO_DETECTOR_LONGITUDE_RAD          0.18333805213    /**< VIRGO vertex longitude (rad) */
#define LAL_VIRGO_DETECTOR_LATITUDE_RAD           0.76151183984    /**< VIRGO vertex latitude (rad) */
#define LAL_VIRGO_DETECTOR_ELEVATION_SI           51.884    /**< VIRGO vertex elevation (m) */
#define LAL_VIRGO_DETECTOR_ARM_X_AZIMUTH_RAD      0.33916285222    /**< VIRGO x arm azimuth (rad) */
#define LAL_VIRGO_DETECTOR_ARM_Y_AZIMUTH_RAD      5.05155183261    /**< VIRGO y arm azimuth (rad) */
#define LAL_VIRGO_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< VIRGO x arm altitude (rad) */
#define LAL_VIRGO_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< VIRGO y arm altitude (rad) */
#define LAL_VIRGO_DETECTOR_ARM_X_MIDPOINT_SI      1500.00000000000    /**< VIRGO x arm midpoint (m) */
#define LAL_VIRGO_DETECTOR_ARM_Y_MIDPOINT_SI      1500.00000000000    /**< VIRGO y arm midpoint (m) */
#define LAL_VIRGO_VERTEX_LOCATION_X_SI            4.54637409900e+06    /**< VIRGO x-component of vertex location in Earth-centered frame (m) */
#define LAL_VIRGO_VERTEX_LOCATION_Y_SI            8.42989697626e+05    /**< VIRGO y-component of vertex location in Earth-centered frame (m) */
#define LAL_VIRGO_VERTEX_LOCATION_Z_SI            4.37857696241e+06    /**< VIRGO z-component of vertex location in Earth-centered frame (m) */
#define LAL_VIRGO_ARM_X_DIRECTION_X               -0.70045821479    /**< VIRGO x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_VIRGO_ARM_X_DIRECTION_Y               0.20848948619    /**< VIRGO y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_VIRGO_ARM_X_DIRECTION_Z               0.68256166277    /**< VIRGO z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_VIRGO_ARM_Y_DIRECTION_X               -0.05379255368    /**< VIRGO x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_VIRGO_ARM_Y_DIRECTION_Y               -0.96908180549    /**< VIRGO y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_VIRGO_ARM_Y_DIRECTION_Z               0.24080451708    /**< VIRGO z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name GEO 600m Interferometric Detector constants
     * The following constants describe the location and geometry of the
     * GEO 600m Interferometric Detector.
     */
    /*@{*/
#define LAL_GEO_600_DETECTOR_NAME                   "GEO_600"    /**< GEO_600 detector name string */
#define LAL_GEO_600_DETECTOR_PREFIX                 "G1"    /**< GEO_600 detector prefix string */
#define LAL_GEO_600_DETECTOR_LONGITUDE_RAD          0.17116780435    /**< GEO_600 vertex longitude (rad) */
#define LAL_GEO_600_DETECTOR_LATITUDE_RAD           0.91184982752    /**< GEO_600 vertex latitude (rad) */
#define LAL_GEO_600_DETECTOR_ELEVATION_SI           114.425    /**< GEO_600 vertex elevation (m) */
#define LAL_GEO_600_DETECTOR_ARM_X_AZIMUTH_RAD      1.19360100484    /**< GEO_600 x arm azimuth (rad) */
#define LAL_GEO_600_DETECTOR_ARM_Y_AZIMUTH_RAD      5.83039279401    /**< GEO_600 y arm azimuth (rad) */
#define LAL_GEO_600_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< GEO_600 x arm altitude (rad) */
#define LAL_GEO_600_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< GEO_600 y arm altitude (rad) */
#define LAL_GEO_600_DETECTOR_ARM_X_MIDPOINT_SI      300.00000000000    /**< GEO_600 x arm midpoint (m) */
#define LAL_GEO_600_DETECTOR_ARM_Y_MIDPOINT_SI      300.00000000000    /**< GEO_600 y arm midpoint (m) */
#define LAL_GEO_600_VERTEX_LOCATION_X_SI            3.85630994926e+06    /**< GEO_600 x-component of vertex location in Earth-centered frame (m) */
#define LAL_GEO_600_VERTEX_LOCATION_Y_SI            6.66598956317e+05    /**< GEO_600 y-component of vertex location in Earth-centered frame (m) */
#define LAL_GEO_600_VERTEX_LOCATION_Z_SI            5.01964141725e+06    /**< GEO_600 z-component of vertex location in Earth-centered frame (m) */
#define LAL_GEO_600_ARM_X_DIRECTION_X               -0.44530676905    /**< GEO_600 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_GEO_600_ARM_X_DIRECTION_Y               0.86651354130    /**< GEO_600 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_GEO_600_ARM_X_DIRECTION_Z               0.22551311312    /**< GEO_600 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_GEO_600_ARM_Y_DIRECTION_X               -0.62605756776    /**< GEO_600 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_GEO_600_ARM_Y_DIRECTION_Y               -0.55218609524    /**< GEO_600 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_GEO_600_ARM_Y_DIRECTION_Z               0.55058372486    /**< GEO_600 z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name LIGO Hanford Observatory 2km Interferometric Detector constants
     * The following constants describe the location and geometry of the
     * LIGO Hanford Observatory 2km Interferometric Detector.
     */
    /*@{*/
#define LAL_LHO_2K_DETECTOR_NAME                   "LHO_2k"    /**< LHO_2k detector name string */
#define LAL_LHO_2K_DETECTOR_PREFIX                 "H2"    /**< LHO_2k detector prefix string */
#define LAL_LHO_2K_DETECTOR_LONGITUDE_RAD          -2.08405676917    /**< LHO_2k vertex longitude (rad) */
#define LAL_LHO_2K_DETECTOR_LATITUDE_RAD           0.81079526383    /**< LHO_2k vertex latitude (rad) */
#define LAL_LHO_2K_DETECTOR_ELEVATION_SI           142.554    /**< LHO_2k vertex elevation (m) */
#define LAL_LHO_2K_DETECTOR_ARM_X_AZIMUTH_RAD      5.65487724844    /**< LHO_2k x arm azimuth (rad) */
#define LAL_LHO_2K_DETECTOR_ARM_Y_AZIMUTH_RAD      4.08408092164    /**< LHO_2k y arm azimuth (rad) */
#define LAL_LHO_2K_DETECTOR_ARM_X_ALTITUDE_RAD     -0.00061950000    /**< LHO_2k x arm altitude (rad) */
#define LAL_LHO_2K_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00001250000    /**< LHO_2k y arm altitude (rad) */
#define LAL_LHO_2K_DETECTOR_ARM_X_MIDPOINT_SI      1004.50000000000    /**< LHO_2k x arm midpoint (m) */
#define LAL_LHO_2K_DETECTOR_ARM_Y_MIDPOINT_SI      1004.50000000000    /**< LHO_2k y arm midpoint (m) */
#define LAL_LHO_2K_VERTEX_LOCATION_X_SI            -2.16141492636e+06    /**< LHO_2k x-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_2K_VERTEX_LOCATION_Y_SI            -3.83469517889e+06    /**< LHO_2k y-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_2K_VERTEX_LOCATION_Z_SI            4.60035022664e+06    /**< LHO_2k z-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_2K_ARM_X_DIRECTION_X               -0.22389266154    /**< LHO_2k x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_2K_ARM_X_DIRECTION_Y               0.79983062746    /**< LHO_2k y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_2K_ARM_X_DIRECTION_Z               0.55690487831    /**< LHO_2k z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_2K_ARM_Y_DIRECTION_X               -0.91397818574    /**< LHO_2k x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LHO_2K_ARM_Y_DIRECTION_Y               0.02609403989    /**< LHO_2k y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LHO_2K_ARM_Y_DIRECTION_Z               -0.40492342125    /**< LHO_2k z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name LIGO Hanford Observatory 4km Interferometric Detector constants
     * The following constants describe the location and geometry of the
     * LIGO Hanford Observatory 4km Interferometric Detector.
     */
    /*@{*/
#define LAL_LHO_4K_DETECTOR_NAME                   "LHO_4k"    /**< LHO_4k detector name string */
#define LAL_LHO_4K_DETECTOR_PREFIX                 "H1"    /**< LHO_4k detector prefix string */
#define LAL_LHO_4K_DETECTOR_LONGITUDE_RAD          -2.08405676917    /**< LHO_4k vertex longitude (rad) */
#define LAL_LHO_4K_DETECTOR_LATITUDE_RAD           0.81079526383    /**< LHO_4k vertex latitude (rad) */
#define LAL_LHO_4K_DETECTOR_ELEVATION_SI           142.554    /**< LHO_4k vertex elevation (m) */
#define LAL_LHO_4K_DETECTOR_ARM_X_AZIMUTH_RAD      5.65487724844    /**< LHO_4k x arm azimuth (rad) */
#define LAL_LHO_4K_DETECTOR_ARM_Y_AZIMUTH_RAD      4.08408092164    /**< LHO_4k y arm azimuth (rad) */
#define LAL_LHO_4K_DETECTOR_ARM_X_ALTITUDE_RAD     -0.00061950000    /**< LHO_4k x arm altitude (rad) */
#define LAL_LHO_4K_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00001250000    /**< LHO_4k y arm altitude (rad) */
#define LAL_LHO_4K_DETECTOR_ARM_X_MIDPOINT_SI      1997.54200000000    /**< LHO_4k x arm midpoint (m) */
#define LAL_LHO_4K_DETECTOR_ARM_Y_MIDPOINT_SI      1997.52200000000    /**< LHO_4k y arm midpoint (m) */
#define LAL_LHO_4K_VERTEX_LOCATION_X_SI            -2.16141492636e+06    /**< LHO_4k x-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_4K_VERTEX_LOCATION_Y_SI            -3.83469517889e+06    /**< LHO_4k y-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_4K_VERTEX_LOCATION_Z_SI            4.60035022664e+06    /**< LHO_4k z-component of vertex location in Earth-centered frame (m) */
#define LAL_LHO_4K_ARM_X_DIRECTION_X               -0.22389266154    /**< LHO_4k x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_X_DIRECTION_Y               0.79983062746    /**< LHO_4k y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_X_DIRECTION_Z               0.55690487831    /**< LHO_4k z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_Y_DIRECTION_X               -0.91397818574    /**< LHO_4k x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_Y_DIRECTION_Y               0.02609403989    /**< LHO_4k y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LHO_4K_ARM_Y_DIRECTION_Z               -0.40492342125    /**< LHO_4k z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name LIGO Livingston Observatory 4km Interferometric Detector constants
     * The following constants describe the location and geometry of the
     * LIGO Livingston Observatory 4km Interferometric Detector.
     */
    /*@{*/
#define LAL_LLO_4K_DETECTOR_NAME                   "LLO_4k"    /**< LLO_4k detector name string */
#define LAL_LLO_4K_DETECTOR_PREFIX                 "L1"    /**< LLO_4k detector prefix string */
#define LAL_LLO_4K_DETECTOR_LONGITUDE_RAD          -1.58430937078    /**< LLO_4k vertex longitude (rad) */
#define LAL_LLO_4K_DETECTOR_LATITUDE_RAD           0.53342313506    /**< LLO_4k vertex latitude (rad) */
#define LAL_LLO_4K_DETECTOR_ELEVATION_SI           -6.574    /**< LLO_4k vertex elevation (m) */
#define LAL_LLO_4K_DETECTOR_ARM_X_AZIMUTH_RAD      4.40317772346    /**< LLO_4k x arm azimuth (rad) */
#define LAL_LLO_4K_DETECTOR_ARM_Y_AZIMUTH_RAD      2.83238139666    /**< LLO_4k y arm azimuth (rad) */
#define LAL_LLO_4K_DETECTOR_ARM_X_ALTITUDE_RAD     -0.00031210000    /**< LLO_4k x arm altitude (rad) */
#define LAL_LLO_4K_DETECTOR_ARM_Y_ALTITUDE_RAD     -0.00061070000    /**< LLO_4k y arm altitude (rad) */
#define LAL_LLO_4K_DETECTOR_ARM_X_MIDPOINT_SI      1997.57500000000    /**< LLO_4k x arm midpoint (m) */
#define LAL_LLO_4K_DETECTOR_ARM_Y_MIDPOINT_SI      1997.57500000000    /**< LLO_4k y arm midpoint (m) */
#define LAL_LLO_4K_VERTEX_LOCATION_X_SI            -7.42760447238e+04    /**< LLO_4k x-component of vertex location in Earth-centered frame (m) */
#define LAL_LLO_4K_VERTEX_LOCATION_Y_SI            -5.49628371971e+06    /**< LLO_4k y-component of vertex location in Earth-centered frame (m) */
#define LAL_LLO_4K_VERTEX_LOCATION_Z_SI            3.22425701744e+06    /**< LLO_4k z-component of vertex location in Earth-centered frame (m) */
#define LAL_LLO_4K_ARM_X_DIRECTION_X               -0.95457412153    /**< LLO_4k x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_X_DIRECTION_Y               -0.14158077340    /**< LLO_4k y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_X_DIRECTION_Z               -0.26218911324    /**< LLO_4k z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_Y_DIRECTION_X               0.29774156894    /**< LLO_4k x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_Y_DIRECTION_Y               -0.48791033647    /**< LLO_4k y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LLO_4K_ARM_Y_DIRECTION_Z               -0.82054461286    /**< LLO_4k z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    
    /**
     * \name LIGO India 4km Interferometric Detector constants
     * @warning These numbers are subject to change.
     * The following constants describe hypothetical location and geometry
     * of the LIGO India 4km Interferometric Detector that have been used
     * in several studies with LALInference. Note that these data do not
     * represent an actual prospective site.
     */
#define LAL_LIO_4K_DETECTOR_NAME                 "LIO_4k" /**< LIO_4K detector name string */
#define LAL_LIO_4K_DETECTOR_PREFIX               "I1"    /**< LIO_4K detector prefix string */
#define LAL_LIO_4K_DETECTOR_LONGITUDE_RAD        1.3340133249409993   /**< LIO_4K vertex longitude (rad; equal to 76°26') */
#define LAL_LIO_4K_DETECTOR_LATITUDE_RAD         0.2484185302005262   /**< LIO_4K vertex latitude (rad; equal to 14°14') */
#define LAL_LIO_4K_DETECTOR_ELEVATION_SI         0.0  /**< LIO_4K vertex elevation (m) */
#define LAL_LIO_4K_DETECTOR_ARM_X_AZIMUTH_RAD    1.5707963705062866   /**< LIO_4K x arm azimuth (rad) */
#define LAL_LIO_4K_DETECTOR_ARM_Y_AZIMUTH_RAD    0.0   /**< LIO_4K y arm azimuth (rad) */
#define LAL_LIO_4K_DETECTOR_ARM_X_ALTITUDE_RAD   0.0   /**< LIO_4K x arm altitude (rad) */
#define LAL_LIO_4K_DETECTOR_ARM_Y_ALTITUDE_RAD   0.0   /**< LIO_4K y arm altitude (rad) */
#define LAL_LIO_4K_DETECTOR_ARM_X_MIDPOINT_SI    2000.00000000000        /**< LIO_4K x arm midpoint (m) */
#define LAL_LIO_4K_DETECTOR_ARM_Y_MIDPOINT_SI    2000.00000000000        /**< LIO_4K y arm midpoint (m) */
#define LAL_LIO_4K_VERTEX_LOCATION_X_SI          1450526.82294155       /**< LIO_4K x-component of vertex location in Earth-centered frame (m) */
#define LAL_LIO_4K_VERTEX_LOCATION_Y_SI          6011058.39047265       /**< LIO_4K y-component of vertex location in Earth-centered frame (m) */
#define LAL_LIO_4K_VERTEX_LOCATION_Z_SI          1558018.27884102       /**< LIO_4K z-component of vertex location in Earth-centered frame (m) */
#define LAL_LIO_4K_ARM_X_DIRECTION_X            -9.72097635269165039e-01  /**< LIO_4K x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LIO_4K_ARM_X_DIRECTION_Y             2.34576612710952759e-01   /**< LIO_4K y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LIO_4K_ARM_X_DIRECTION_Z            -4.23695567519644101e-08 /**< LIO_4K z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_LIO_4K_ARM_Y_DIRECTION_X             -5.76756671071052551e-02  /**< LIO_4K x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LIO_4K_ARM_Y_DIRECTION_Y            -2.39010959863662720e-01   /**< LIO_4K y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_LIO_4K_ARM_Y_DIRECTION_Z             9.69302475452423096e-01  /**< LIO_4K z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name Caltech 40m Prototype Detector constants
     * The following constants describe the location and geometry of the
     * Caltech 40m Prototype Detector.
     */
    /*@{*/
#define LAL_CIT_40_DETECTOR_NAME                   "CIT_40"    /**< CIT_40 detector name string */
#define LAL_CIT_40_DETECTOR_PREFIX                 "P1"    /**< CIT_40 detector prefix string */
#define LAL_CIT_40_DETECTOR_LONGITUDE_RAD          -2.06175744538    /**< CIT_40 vertex longitude (rad) */
#define LAL_CIT_40_DETECTOR_LATITUDE_RAD           0.59637900541    /**< CIT_40 vertex latitude (rad) */
#define LAL_CIT_40_DETECTOR_ELEVATION_SI           0    /**< CIT_40 vertex elevation (m) */
#define LAL_CIT_40_DETECTOR_ARM_X_AZIMUTH_RAD      3.14159265359    /**< CIT_40 x arm azimuth (rad) */
#define LAL_CIT_40_DETECTOR_ARM_Y_AZIMUTH_RAD      1.57079632679    /**< CIT_40 y arm azimuth (rad) */
#define LAL_CIT_40_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< CIT_40 x arm altitude (rad) */
#define LAL_CIT_40_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< CIT_40 y arm altitude (rad) */
#define LAL_CIT_40_DETECTOR_ARM_X_MIDPOINT_SI      19.12500000000    /**< CIT_40 x arm midpoint (m) */
#define LAL_CIT_40_DETECTOR_ARM_Y_MIDPOINT_SI      19.12500000000    /**< CIT_40 y arm midpoint (m) */
#define LAL_CIT_40_VERTEX_LOCATION_X_SI            -2.49064958347e+06    /**< CIT_40 x-component of vertex location in Earth-centered frame (m) */
#define LAL_CIT_40_VERTEX_LOCATION_Y_SI            -4.65869968211e+06    /**< CIT_40 y-component of vertex location in Earth-centered frame (m) */
#define LAL_CIT_40_VERTEX_LOCATION_Z_SI            3.56206411403e+06    /**< CIT_40 z-component of vertex location in Earth-centered frame (m) */
#define LAL_CIT_40_ARM_X_DIRECTION_X               -0.26480331633    /**< CIT_40 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_CIT_40_ARM_X_DIRECTION_Y               -0.49530818538    /**< CIT_40 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_CIT_40_ARM_X_DIRECTION_Z               -0.82737476706    /**< CIT_40 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_CIT_40_ARM_Y_DIRECTION_X               0.88188012386    /**< CIT_40 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_CIT_40_ARM_Y_DIRECTION_Y               -0.47147369718    /**< CIT_40 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_CIT_40_ARM_Y_DIRECTION_Z               0.00000000000    /**< CIT_40 z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /**
     * \name Einstein Telescop 10km Interferometric Detector constants
     * The following constants describe the locations and geometrys of the
     * three 10km Interferometric Detectors for the planned third generation
     * Einstein Telescop detector as well as the theoretical null stream.
     * See T1400308
     */
    /*@{*/
#define LAL_ET1_DETECTOR_NAME                      "ET1_T1400308"    /**< ET1 detector name string */
#define LAL_ET1_DETECTOR_PREFIX                    "E1"    /**< ET1 detector prefix string */
#define LAL_ET1_DETECTOR_LONGITUDE_RAD             0.18333805213    /**< ET1 vertex longitude (rad) */
#define LAL_ET1_DETECTOR_LATITUDE_RAD              0.76151183984    /**< ET1 vertex latitude (rad) */
#define LAL_ET1_DETECTOR_ELEVATION_SI              51.884    /**< ET1 vertex elevation (m) */
#define LAL_ET1_DETECTOR_ARM_X_AZIMUTH_RAD         0.33916285222    /**< ET1 x arm azimuth (rad) */
#define LAL_ET1_DETECTOR_ARM_Y_AZIMUTH_RAD         5.57515060820    /**< ET1 y arm azimuth (rad) */
#define LAL_ET1_DETECTOR_ARM_X_ALTITUDE_RAD        0.00000000000    /**< ET1 x arm altitude (rad) */
#define LAL_ET1_DETECTOR_ARM_Y_ALTITUDE_RAD        0.00000000000    /**< ET1 y arm altitude (rad) */
#define LAL_ET1_DETECTOR_ARM_X_MIDPOINT_SI         5000.00000000000    /**< ET1 x arm midpoint (m) */
#define LAL_ET1_DETECTOR_ARM_Y_MIDPOINT_SI         5000.00000000000    /**< ET1 y arm midpoint (m) */
#define LAL_ET1_VERTEX_LOCATION_X_SI               4.54637409900e+06    /**< ET1 x-component of vertex location in Earth-centered frame (m) */
#define LAL_ET1_VERTEX_LOCATION_Y_SI               8.42989697626e+05    /**< ET1 y-component of vertex location in Earth-centered frame (m) */
#define LAL_ET1_VERTEX_LOCATION_Z_SI               4.37857696241e+06    /**< ET1 z-component of vertex location in Earth-centered frame (m) */
#define LAL_ET1_ARM_X_DIRECTION_X                  -0.70045821479    /**< ET1 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET1_ARM_X_DIRECTION_Y                  0.20848948619    /**< ET1 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET1_ARM_X_DIRECTION_Z                  0.68256166277    /**< ET1 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET1_ARM_Y_DIRECTION_X                  -0.39681482542    /**< ET1 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET1_ARM_Y_DIRECTION_Y                  -0.73500471881    /**< ET1 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET1_ARM_Y_DIRECTION_Z                  0.54982366052    /**< ET1 z-component of unit vector pointing along y arm in Earth-centered frame */
    
    
#define LAL_ET2_DETECTOR_NAME                      "ET2_T1400308"    /**< ET2 detector name string */
#define LAL_ET2_DETECTOR_PREFIX                    "E2"    /**< ET2 detector prefix string */
#define LAL_ET2_DETECTOR_LONGITUDE_RAD             0.18405858870    /**< ET2 vertex longitude (rad) */
#define LAL_ET2_DETECTOR_LATITUDE_RAD              0.76299307990    /**< ET2 vertex latitude (rad) */
#define LAL_ET2_DETECTOR_ELEVATION_SI              59.735    /**< ET2 vertex elevation (m) */
#define LAL_ET2_DETECTOR_ARM_X_AZIMUTH_RAD         4.52795305701    /**< ET2 x arm azimuth (rad) */
#define LAL_ET2_DETECTOR_ARM_Y_AZIMUTH_RAD         3.48075550581    /**< ET2 y arm azimuth (rad) */
#define LAL_ET2_DETECTOR_ARM_X_ALTITUDE_RAD        0.00000000000    /**< ET2 x arm altitude (rad) */
#define LAL_ET2_DETECTOR_ARM_Y_ALTITUDE_RAD        0.00000000000    /**< ET2 y arm altitude (rad) */
#define LAL_ET2_DETECTOR_ARM_X_MIDPOINT_SI         5000.00000000000    /**< ET2 x arm midpoint (m) */
#define LAL_ET2_DETECTOR_ARM_Y_MIDPOINT_SI         5000.00000000000    /**< ET2 y arm midpoint (m) */
#define LAL_ET2_VERTEX_LOCATION_X_SI               4.53936951685e+06    /**< ET2 x-component of vertex location in Earth-centered frame (m) */
#define LAL_ET2_VERTEX_LOCATION_Y_SI               8.45074592488e+05    /**< ET2 y-component of vertex location in Earth-centered frame (m) */
#define LAL_ET2_VERTEX_LOCATION_Z_SI               4.38540257904e+06    /**< ET2 z-component of vertex location in Earth-centered frame (m) */
#define LAL_ET2_ARM_X_DIRECTION_X                  0.30364338937    /**< ET2 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET2_ARM_X_DIRECTION_Y                  -0.94349420500    /**< ET2 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET2_ARM_X_DIRECTION_Z                  -0.13273800225    /**< ET2 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET2_ARM_Y_DIRECTION_X                  0.70045821479    /**< ET2 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET2_ARM_Y_DIRECTION_Y                  -0.20848948619    /**< ET2 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET2_ARM_Y_DIRECTION_Z                  -0.68256166277    /**< ET2 z-component of unit vector pointing along y arm in Earth-centered frame */
    
    
#define LAL_ET3_DETECTOR_NAME                      "ET3_T1400308"    /**< ET3 detector name string */
#define LAL_ET3_DETECTOR_PREFIX                    "E3"    /**< ET3 detector prefix string */
#define LAL_ET3_DETECTOR_LONGITUDE_RAD             0.18192996730    /**< ET3 vertex longitude (rad) */
#define LAL_ET3_DETECTOR_LATITUDE_RAD              0.76270463257    /**< ET3 vertex latitude (rad) */
#define LAL_ET3_DETECTOR_ELEVATION_SI              59.727    /**< ET3 vertex elevation (m) */
#define LAL_ET3_DETECTOR_ARM_X_AZIMUTH_RAD         2.43355795462    /**< ET3 x arm azimuth (rad) */
#define LAL_ET3_DETECTOR_ARM_Y_AZIMUTH_RAD         1.38636040342    /**< ET3 y arm azimuth (rad) */
#define LAL_ET3_DETECTOR_ARM_X_ALTITUDE_RAD        0.00000000000    /**< ET3 x arm altitude (rad) */
#define LAL_ET3_DETECTOR_ARM_Y_ALTITUDE_RAD        0.00000000000    /**< ET3 y arm altitude (rad) */
#define LAL_ET3_DETECTOR_ARM_X_MIDPOINT_SI         5000.00000000000    /**< ET3 x arm midpoint (m) */
#define LAL_ET3_DETECTOR_ARM_Y_MIDPOINT_SI         5000.00000000000    /**< ET3 y arm midpoint (m) */
#define LAL_ET3_VERTEX_LOCATION_X_SI               4.54240595075e+06    /**< ET3 x-component of vertex location in Earth-centered frame (m) */
#define LAL_ET3_VERTEX_LOCATION_Y_SI               8.35639650438e+05    /**< ET3 y-component of vertex location in Earth-centered frame (m) */
#define LAL_ET3_VERTEX_LOCATION_Z_SI               4.38407519902e+06    /**< ET3 z-component of vertex location in Earth-centered frame (m) */
#define LAL_ET3_ARM_X_DIRECTION_X                  0.39681482542    /**< ET3 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET3_ARM_X_DIRECTION_Y                  0.73500471881    /**< ET3 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET3_ARM_X_DIRECTION_Z                  -0.54982366052    /**< ET3 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET3_ARM_Y_DIRECTION_X                  -0.30364338937    /**< ET3 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET3_ARM_Y_DIRECTION_Y                  0.94349420500    /**< ET3 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET3_ARM_Y_DIRECTION_Z                  0.13273800225    /**< ET3 z-component of unit vector pointing along y arm in Earth-centered frame */
    
    
#define LAL_ET0_DETECTOR_NAME                      "ET0_T1400308"    /**< ET0 detector name string */
#define LAL_ET0_DETECTOR_PREFIX                    "E0"    /**< ET0 detector prefix string */
#define LAL_ET0_DETECTOR_LONGITUDE_RAD             0.18192996730    /**< ET0 vertex longitude (rad) */
#define LAL_ET0_DETECTOR_LATITUDE_RAD              0.76270463257    /**< ET0 vertex latitude (rad) */
#define LAL_ET0_DETECTOR_ELEVATION_SI              59.727    /**< ET0 vertex elevation (m) */
#define LAL_ET0_DETECTOR_ARM_X_AZIMUTH_RAD         0.00000000000    /**< ET0 x arm azimuth (rad) */
#define LAL_ET0_DETECTOR_ARM_Y_AZIMUTH_RAD         0.00000000000    /**< ET0 y arm azimuth (rad) */
#define LAL_ET0_DETECTOR_ARM_X_ALTITUDE_RAD        0.00000000000    /**< ET0 x arm altitude (rad) */
#define LAL_ET0_DETECTOR_ARM_Y_ALTITUDE_RAD        0.00000000000    /**< ET0 y arm altitude (rad) */
#define LAL_ET0_DETECTOR_ARM_X_MIDPOINT_SI         0.00000000000    /**< ET0 x arm midpoint (m) */
#define LAL_ET0_DETECTOR_ARM_Y_MIDPOINT_SI         0.00000000000    /**< ET0 y arm midpoint (m) */
#define LAL_ET0_VERTEX_LOCATION_X_SI               4.54240595075e+06    /**< ET0 x-component of vertex location in Earth-centered frame (m) */
#define LAL_ET0_VERTEX_LOCATION_Y_SI               8.35639650438e+05    /**< ET0 y-component of vertex location in Earth-centered frame (m) */
#define LAL_ET0_VERTEX_LOCATION_Z_SI               4.38407519902e+06    /**< ET0 z-component of vertex location in Earth-centered frame (m) */
#define LAL_ET0_ARM_X_DIRECTION_X                  0.00000000000    /**< ET0 x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET0_ARM_X_DIRECTION_Y                  0.00000000000    /**< ET0 y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET0_ARM_X_DIRECTION_Z                  0.00000000000    /**< ET0 z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_ET0_ARM_Y_DIRECTION_X                  0.00000000000    /**< ET0 x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET0_ARM_Y_DIRECTION_Y                  0.00000000000    /**< ET0 y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_ET0_ARM_Y_DIRECTION_Z                  0.00000000000    /**< ET0 z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    /**
     * \name KAGRA Interferometric Detector constants
     * The following constants describe the location and geometry of the
     * KAGRA Interferometric Detector.
     * \sa
     * > Yoshio Saito, "KAGRA location", KAGRA Technical Document JGW-G1503824
     * > http://gwdoc.icrr.u-tokyo.ac.jp/cgi-bin/DocDB/ShowDocument?docid=3824
     */
    /*@{*/
#define LAL_KAGRA_DETECTOR_NAME                   "KAGRA"    /**< KAGRA detector name string */
#define LAL_KAGRA_DETECTOR_PREFIX                 "K1"    /**< KAGRA detector prefix string */
#define LAL_KAGRA_DETECTOR_LONGITUDE_RAD          2.396441015    /**< KAGRA vertex longitude (rad) */
#define LAL_KAGRA_DETECTOR_LATITUDE_RAD           0.6355068497    /**< KAGRA vertex latitude (rad) */
#define LAL_KAGRA_DETECTOR_ELEVATION_SI           414.181    /**< KAGRA vertex elevation (m) */
#define LAL_KAGRA_DETECTOR_ARM_X_AZIMUTH_RAD      1.054113    /**< KAGRA x arm azimuth (rad) */
#define LAL_KAGRA_DETECTOR_ARM_Y_AZIMUTH_RAD      -0.5166798    /**< KAGRA y arm azimuth (rad) */
#define LAL_KAGRA_DETECTOR_ARM_X_ALTITUDE_RAD     0.0031414    /**< KAGRA x arm altitude (rad) */
#define LAL_KAGRA_DETECTOR_ARM_Y_ALTITUDE_RAD     -0.0036270    /**< KAGRA y arm altitude (rad) */
#define LAL_KAGRA_DETECTOR_ARM_X_MIDPOINT_SI      1513.2535    /**< KAGRA x arm midpoint (m) */
#define LAL_KAGRA_DETECTOR_ARM_Y_MIDPOINT_SI      1511.611    /**< KAGRA y arm midpoint (m) */
#define LAL_KAGRA_VERTEX_LOCATION_X_SI            -3777336.024    /**< KAGRA x-component of vertex location in Earth-centered frame (m) */
#define LAL_KAGRA_VERTEX_LOCATION_Y_SI            3484898.411    /**< KAGRA y-component of vertex location in Earth-centered frame (m) */
#define LAL_KAGRA_VERTEX_LOCATION_Z_SI            3765313.697    /**< KAGRA z-component of vertex location in Earth-centered frame (m) */
#define LAL_KAGRA_ARM_X_DIRECTION_X               -0.3759040    /**< KAGRA x-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_KAGRA_ARM_X_DIRECTION_Y               -0.8361583    /**< KAGRA y-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_KAGRA_ARM_X_DIRECTION_Z               0.3994189    /**< KAGRA z-component of unit vector pointing along x arm in Earth-centered frame */
#define LAL_KAGRA_ARM_Y_DIRECTION_X               0.7164378    /**< KAGRA x-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_KAGRA_ARM_Y_DIRECTION_Y               0.01114076    /**< KAGRA y-component of unit vector pointing along y arm in Earth-centered frame */
#define LAL_KAGRA_ARM_Y_DIRECTION_Z               0.6975620    /**< KAGRA z-component of unit vector pointing along y arm in Earth-centered frame */
    /*@}*/
    
    
    /* Resonant Mass (Bar) Detectors */
    
    
    /**
     * \name ALLEGRO Resonant Mass Detector with 320 degree azimuth "IGEC axis" constants
     * The following constants describe the location and geometry of the
     * ALLEGRO Resonant Mass Detector with 320 degree azimuth "IGEC axis".
     */
    /*@{*/
#define LAL_ALLEGRO_320_DETECTOR_NAME                   "ALLEGRO_320"    /**< ALLEGRO_320 detector name string */
#define LAL_ALLEGRO_320_DETECTOR_PREFIX                 "A1"    /**< ALLEGRO_320 detector prefix string */
#define LAL_ALLEGRO_320_DETECTOR_LONGITUDE_RAD          -1.59137068496    /**< ALLEGRO_320 vertex longitude (rad) */
#define LAL_ALLEGRO_320_DETECTOR_LATITUDE_RAD           0.53079879206    /**< ALLEGRO_320 vertex latitude (rad) */
#define LAL_ALLEGRO_320_DETECTOR_ELEVATION_SI           0    /**< ALLEGRO_320 vertex elevation (m) */
#define LAL_ALLEGRO_320_DETECTOR_ARM_X_AZIMUTH_RAD      -0.69813170080    /**< ALLEGRO_320 x arm azimuth (rad) */
#define LAL_ALLEGRO_320_DETECTOR_ARM_Y_AZIMUTH_RAD      0.00000000000    /**< ALLEGRO_320 y arm azimuth (rad) UNUSED FOR BARS */
#define LAL_ALLEGRO_320_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< ALLEGRO_320 x arm altitude (rad) */
#define LAL_ALLEGRO_320_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< ALLEGRO_320 y arm altitude (rad) UNUSED FOR BARS */
#define LAL_ALLEGRO_320_DETECTOR_ARM_X_MIDPOINT_SI      0.00000000000    /**< ALLEGRO_320 x arm midpoint (m) UNUSED FOR BARS */
#define LAL_ALLEGRO_320_DETECTOR_ARM_Y_MIDPOINT_SI      0.00000000000    /**< ALLEGRO_320 y arm midpoint (m) UNUSED FOR BARS */
#define LAL_ALLEGRO_320_VERTEX_LOCATION_X_SI            -1.13258964140e+05    /**< ALLEGRO_320 x-component of vertex location in Earth-centered frame (m) */
#define LAL_ALLEGRO_320_VERTEX_LOCATION_Y_SI            -5.50408337391e+06    /**< ALLEGRO_320 y-component of vertex location in Earth-centered frame (m) */
#define LAL_ALLEGRO_320_VERTEX_LOCATION_Z_SI            3.20989567981e+06    /**< ALLEGRO_320 z-component of vertex location in Earth-centered frame (m) */
#define LAL_ALLEGRO_320_AXIS_DIRECTION_X                -0.63467362345    /**< ALLEGRO_320 x-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_ALLEGRO_320_AXIS_DIRECTION_Y                0.40093077976    /**< ALLEGRO_320 y-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_ALLEGRO_320_AXIS_DIRECTION_Z                0.66063901000    /**< ALLEGRO_320 z-component of unit vector pointing along axis in Earth-centered frame */
    /*@}*/
    
    /**
     * \name AURIGA Resonant Mass Detector constants
     * The following constants describe the location and geometry of the
     * AURIGA Resonant Mass Detector.
     */
    /*@{*/
#define LAL_AURIGA_DETECTOR_NAME                   "AURIGA"    /**< AURIGA detector name string */
#define LAL_AURIGA_DETECTOR_PREFIX                 "O1"    /**< AURIGA detector prefix string */
#define LAL_AURIGA_DETECTOR_LONGITUDE_RAD          0.20853775679    /**< AURIGA vertex longitude (rad) */
#define LAL_AURIGA_DETECTOR_LATITUDE_RAD           0.79156499342    /**< AURIGA vertex latitude (rad) */
#define LAL_AURIGA_DETECTOR_ELEVATION_SI           0    /**< AURIGA vertex elevation (m) */
#define LAL_AURIGA_DETECTOR_ARM_X_AZIMUTH_RAD      0.76794487088    /**< AURIGA x arm azimuth (rad) */
#define LAL_AURIGA_DETECTOR_ARM_Y_AZIMUTH_RAD      0.00000000000    /**< AURIGA y arm azimuth (rad) UNUSED FOR BARS */
#define LAL_AURIGA_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< AURIGA x arm altitude (rad) */
#define LAL_AURIGA_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< AURIGA y arm altitude (rad) UNUSED FOR BARS */
#define LAL_AURIGA_DETECTOR_ARM_X_MIDPOINT_SI      0.00000000000    /**< AURIGA x arm midpoint (m) UNUSED FOR BARS */
#define LAL_AURIGA_DETECTOR_ARM_Y_MIDPOINT_SI      0.00000000000    /**< AURIGA y arm midpoint (m) UNUSED FOR BARS */
#define LAL_AURIGA_VERTEX_LOCATION_X_SI            4.39246733007e+06    /**< AURIGA x-component of vertex location in Earth-centered frame (m) */
#define LAL_AURIGA_VERTEX_LOCATION_Y_SI            9.29508666967e+05    /**< AURIGA y-component of vertex location in Earth-centered frame (m) */
#define LAL_AURIGA_VERTEX_LOCATION_Z_SI            4.51502913071e+06    /**< AURIGA z-component of vertex location in Earth-centered frame (m) */
#define LAL_AURIGA_AXIS_DIRECTION_X                -0.64450412225    /**< AURIGA x-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_AURIGA_AXIS_DIRECTION_Y                0.57365538956    /**< AURIGA y-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_AURIGA_AXIS_DIRECTION_Z                0.50550364038    /**< AURIGA z-component of unit vector pointing along axis in Earth-centered frame */
    /*@}*/
    
    /**
     * \name EXPLORER Resonant Mass Detector constants
     * The following constants describe the location and geometry of the
     * EXPLORER Resonant Mass Detector.
     */
    /*@{*/
#define LAL_EXPLORER_DETECTOR_NAME                   "EXPLORER"    /**< EXPLORER detector name string */
#define LAL_EXPLORER_DETECTOR_PREFIX                 "C1"            /**< EXPLORER detector prefix string */
#define LAL_EXPLORER_DETECTOR_LONGITUDE_RAD          0.10821041362    /**< EXPLORER vertex longitude (rad) */
#define LAL_EXPLORER_DETECTOR_LATITUDE_RAD           0.81070543755    /**< EXPLORER vertex latitude (rad) */
#define LAL_EXPLORER_DETECTOR_ELEVATION_SI           0    /**< EXPLORER vertex elevation (m) */
#define LAL_EXPLORER_DETECTOR_ARM_X_AZIMUTH_RAD      0.68067840828    /**< EXPLORER x arm azimuth (rad) */
#define LAL_EXPLORER_DETECTOR_ARM_Y_AZIMUTH_RAD      0.00000000000    /**< EXPLORER y arm azimuth (rad) UNUSED FOR BARS */
#define LAL_EXPLORER_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< EXPLORER x arm altitude (rad) */
#define LAL_EXPLORER_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< EXPLORER y arm altitude (rad) UNUSED FOR BARS */
#define LAL_EXPLORER_DETECTOR_ARM_X_MIDPOINT_SI      0.00000000000    /**< EXPLORER x arm midpoint (m) UNUSED FOR BARS */
#define LAL_EXPLORER_DETECTOR_ARM_Y_MIDPOINT_SI      0.00000000000    /**< EXPLORER y arm midpoint (m) UNUSED FOR BARS */
#define LAL_EXPLORER_VERTEX_LOCATION_X_SI            4.37645395452e+06    /**< EXPLORER x-component of vertex location in Earth-centered frame (m) */
#define LAL_EXPLORER_VERTEX_LOCATION_Y_SI            4.75435044067e+05    /**< EXPLORER y-component of vertex location in Earth-centered frame (m) */
#define LAL_EXPLORER_VERTEX_LOCATION_Z_SI            4.59985274450e+06    /**< EXPLORER z-component of vertex location in Earth-centered frame (m) */
#define LAL_EXPLORER_AXIS_DIRECTION_X                -0.62792641437    /**< EXPLORER x-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_EXPLORER_AXIS_DIRECTION_Y                0.56480832712    /**< EXPLORER y-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_EXPLORER_AXIS_DIRECTION_Z                0.53544371484    /**< EXPLORER z-component of unit vector pointing along axis in Earth-centered frame */
    /*@}*/
    
    /**
     * \name Nautilus Resonant Mass Detector constants
     * The following constants describe the location and geometry of the
     * Nautilus Resonant Mass Detector.
     */
    /*@{*/
#define LAL_NAUTILUS_DETECTOR_NAME                   "Nautilus"    /**< Nautilus detector name string */
#define LAL_NAUTILUS_DETECTOR_PREFIX                 "N1"    /**< Nautilus detector prefix string */
#define LAL_NAUTILUS_DETECTOR_LONGITUDE_RAD          0.22117684946    /**< Nautilus vertex longitude (rad) */
#define LAL_NAUTILUS_DETECTOR_LATITUDE_RAD           0.72996456710    /**< Nautilus vertex latitude (rad) */
#define LAL_NAUTILUS_DETECTOR_ELEVATION_SI           0    /**< Nautilus vertex elevation (m) */
#define LAL_NAUTILUS_DETECTOR_ARM_X_AZIMUTH_RAD      0.76794487088    /**< Nautilus x arm azimuth (rad) */
#define LAL_NAUTILUS_DETECTOR_ARM_Y_AZIMUTH_RAD      0.00000000000    /**< Nautilus y arm azimuth (rad) UNUSED FOR BARS */
#define LAL_NAUTILUS_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< Nautilus x arm altitude (rad) */
#define LAL_NAUTILUS_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< Nautilus y arm altitude (rad) UNUSED FOR BARS */
#define LAL_NAUTILUS_DETECTOR_ARM_X_MIDPOINT_SI      0.00000000000    /**< Nautilus x arm midpoint (m) UNUSED FOR BARS */
#define LAL_NAUTILUS_DETECTOR_ARM_Y_MIDPOINT_SI      0.00000000000    /**< Nautilus y arm midpoint (m) UNUSED FOR BARS */
#define LAL_NAUTILUS_VERTEX_LOCATION_X_SI            4.64410999868e+06    /**< Nautilus x-component of vertex location in Earth-centered frame (m) */
#define LAL_NAUTILUS_VERTEX_LOCATION_Y_SI            1.04425342477e+06    /**< Nautilus y-component of vertex location in Earth-centered frame (m) */
#define LAL_NAUTILUS_VERTEX_LOCATION_Z_SI            4.23104713307e+06    /**< Nautilus z-component of vertex location in Earth-centered frame (m) */
#define LAL_NAUTILUS_AXIS_DIRECTION_X                -0.62039441384    /**< Nautilus x-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_NAUTILUS_AXIS_DIRECTION_Y                0.57250373141    /**< Nautilus y-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_NAUTILUS_AXIS_DIRECTION_Z                0.53605060283    /**< Nautilus z-component of unit vector pointing along axis in Earth-centered frame */
    /*@}*/
    
    /**
     * \name NIOBE Resonant Mass Detector constants
     * The following constants describe the location and geometry of the
     * NIOBE Resonant Mass Detector.
     */
    /*@{*/
#define LAL_NIOBE_DETECTOR_NAME                   "NIOBE"    /**< NIOBE detector name string */
#define LAL_NIOBE_DETECTOR_PREFIX                 "B1"    /**< NIOBE detector prefix string */
#define LAL_NIOBE_DETECTOR_LONGITUDE_RAD          2.02138216202    /**< NIOBE vertex longitude (rad) */
#define LAL_NIOBE_DETECTOR_LATITUDE_RAD           -0.55734180780    /**< NIOBE vertex latitude (rad) */
#define LAL_NIOBE_DETECTOR_ELEVATION_SI           0    /**< NIOBE vertex elevation (m) */
#define LAL_NIOBE_DETECTOR_ARM_X_AZIMUTH_RAD      0.00000000000    /**< NIOBE x arm azimuth (rad) */
#define LAL_NIOBE_DETECTOR_ARM_Y_AZIMUTH_RAD      0.00000000000    /**< NIOBE y arm azimuth (rad) UNUSED FOR BARS */
#define LAL_NIOBE_DETECTOR_ARM_X_ALTITUDE_RAD     0.00000000000    /**< NIOBE x arm altitude (rad) */
#define LAL_NIOBE_DETECTOR_ARM_Y_ALTITUDE_RAD     0.00000000000    /**< NIOBE y arm altitude (rad) UNUSED FOR BARS */
#define LAL_NIOBE_DETECTOR_ARM_X_MIDPOINT_SI      0.00000000000    /**< NIOBE x arm midpoint (m) UNUSED FOR BARS */
#define LAL_NIOBE_DETECTOR_ARM_Y_MIDPOINT_SI      0.00000000000    /**< NIOBE y arm midpoint (m) UNUSED FOR BARS */
#define LAL_NIOBE_VERTEX_LOCATION_X_SI            -2.35948871453e+06    /**< NIOBE x-component of vertex location in Earth-centered frame (m) */
#define LAL_NIOBE_VERTEX_LOCATION_Y_SI            4.87721571259e+06    /**< NIOBE y-component of vertex location in Earth-centered frame (m) */
#define LAL_NIOBE_VERTEX_LOCATION_Z_SI            -3.35416003274e+06    /**< NIOBE z-component of vertex location in Earth-centered frame (m) */
#define LAL_NIOBE_AXIS_DIRECTION_X                -0.23034623759    /**< NIOBE x-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_NIOBE_AXIS_DIRECTION_Y                0.47614056486    /**< NIOBE y-component of unit vector pointing along axis in Earth-centered frame */
#define LAL_NIOBE_AXIS_DIRECTION_Z                0.84866411101    /**< NIOBE z-component of unit vector pointing along axis in Earth-centered frame */
    /*@}*/


/** END Detector Constants **/


/** \name Floating-point constants
 * The following constants define the precision and range of
 * floating-point arithmetic in LAL.  They are taken from the IEEE
 * standard 754 for binary arithmetic.  All numbers are dimensionless. */
/*@{*/
#define LAL_REAL4_MANT 24 /**< Bits of precision in the mantissa of a REAL4 */
#define LAL_REAL4_MAX 3.40282347e+38 /**< Largest REAL4 */
#define LAL_REAL4_MIN 1.17549435e-38 /**< Smallest nonzero REAL4 */
#define LAL_REAL4_EPS 1.19209290e-07 /**< 0.5^(LAL_REAL4_MANT-1), ie the difference between 1 and the next resolveable REAL4 */
#define LAL_REAL8_MANT 53 /**< Bits of precision in the mantissa of a REAL8 */
#define LAL_REAL8_MAX 1.7976931348623157e+308 /**< Largest REAL8 */
#define LAL_REAL8_MIN 2.2250738585072014e-308 /**< Smallest nonzero REAL8 */
#define LAL_REAL8_EPS 2.2204460492503131e-16  /**< 0.5^(LAL_REAL8_MANT-1), ie the difference between 1 and the next resolveable REAL8 */
/*@}*/
    
#define XLAL_BILLION_INT4 1000000000
#define XLAL_BILLION_INT8 LAL_INT8_C( 1000000000 )
#define XLAL_BILLION_REAL8 1e9

/** \name Mathematical constants
 * The following are fundamental mathematical constants.  They are mostly
 * taken from the GNU C <tt>math.h</tt> header (with the exception of
 * <tt>LAL_TWOPI</tt>, which was computed using Maple).  All numbers are
 * dimensionless. The value of exp(gamma) is taken from
 * http://www.research.att.com/~njas/sequences/A073004 */
/*@{*/
#define LAL_E         2.7182818284590452353602874713526625  /**< e */
#define LAL_LOG2E     1.4426950408889634073599246810018922  /**< log_2 e */
#define LAL_LOG10E    0.4342944819032518276511289189166051  /**< log_10 e */
#define LAL_LN2       0.6931471805599453094172321214581766  /**< log_e 2 */
#define LAL_LN10      2.3025850929940456840179914546843642  /**< log_e 10 */
#define LAL_SQRT2     1.4142135623730950488016887242096981  /**< sqrt(2) */
#define LAL_SQRT1_2   0.7071067811865475244008443621048490  /**< 1/sqrt(2) */
#define LAL_GAMMA     0.5772156649015328606065120900824024  /**< gamma */
#define LAL_EXPGAMMA  1.7810724179901979852365041031071795  /**< exp(gamma) */
/* Assuming we're not near a black hole or in Tennessee... */
#define LAL_PI        3.1415926535897932384626433832795029  /**< pi */
#define LAL_TWOPI     6.2831853071795864769252867665590058  /**< 2*pi */
#define LAL_PI_2      1.5707963267948966192313216916397514  /**< pi/2 */
#define LAL_PI_4      0.7853981633974483096156608458198757  /**< pi/4 */
#define LAL_1_PI      0.3183098861837906715377675267450287  /**< 1/pi */
#define LAL_2_PI      0.6366197723675813430755350534900574  /**< 2/pi */
#define LAL_2_SQRTPI  1.1283791670955125738961589031215452  /**< 2/sqrt(pi) */
#define LAL_PI_180    1.7453292519943295769236907684886127e-2 /**< pi/180 */
#define LAL_180_PI    57.295779513082320876798154814105170 /**< 180/pi */
/*@}*/

/** \name Exact physical constants
 * The following physical constants are defined to have exact values.
 * The values of \f$c\f$ and \f$g\f$ are taken from \ref Barnet_1996,
 * \f$p_\mathrm{atm}\f$ is from \ref Lang_1992, while \f$\epsilon_0\f$ and
 * \f$\mu_0\f$ are computed from \f$c\f$ using exact formulae.  The use
 * of a Julian year (365.25 days) as standard is specified by the IAU.
 * They are given in the SI units shown. */
/*@{*/
#define LAL_C_SI      299792458 /**< Speed of light in vacuo, m s^-1 */
#define LAL_EPSILON0_SI  8.8541878176203898505365630317107503e-12 /**< Permittivity of free space, C^2 N^-1 m^-2 */
#define LAL_MU0_SI    1.2566370614359172953850573533118012e-6 /**< Permeability of free space, N A^-2 */
#define LAL_GEARTH_SI 9.80665 /**< Standard gravity, m s^-2 */
#define LAL_PATM_SI 101325 /**< Standard atmosphere, Pa */
#define LAL_YRJUL_SI 31557600 /**< Julian year, s */
#define LAL_LYR_SI 9.4607304725808e15 /**< (Julian) Lightyear, m */
/*@}*/

/** \name Physical constants
 * The following are measured fundamental physical constants, with values
 * given in \ref Barnet_1996.  When not dimensionless, they are given
 * in the SI units shown. */
/*@{*/
#define LAL_G_SI      6.67259e-11    /**< Gravitational constant, N m^2 kg^-2 */
#define LAL_H_SI      6.6260755e-34  /**< Planck constant, J s */
#define LAL_HBAR_SI   1.05457266e-34 /**< Reduced Planck constant, J s */
#define LAL_MPL_SI    2.17671e-8     /**< Planck mass, kg */
#define LAL_LPL_SI    1.61605e-35    /**< Planck length, m */
#define LAL_TPL_SI    5.39056e-44    /**< Planck time, s */
#define LAL_K_SI      1.380658e-23   /**< Boltzmann constant, J K^-1 */
#define LAL_R_SI      8.314511       /**< Ideal gas constant, J K^-1 */
#define LAL_MOL       6.0221367e23   /**< Avogadro constant, dimensionless */
#define LAL_BWIEN_SI  2.897756e-3    /**< Wien displacement law constant, m K */
#define LAL_SIGMA_SI  5.67051e-8  /**< Stefan-Boltzmann constant, W m^-2 K^-4 */
#define LAL_AMU_SI    1.6605402e-27  /**< Atomic mass unit, kg */
#define LAL_MP_SI     1.6726231e-27  /**< Proton mass, kg */
#define LAL_ME_SI     9.1093897e-31  /**< Electron mass, kg */
#define LAL_QE_SI     1.60217733e-19 /**< Electron charge, C */
#define LAL_ALPHA  7.297354677e-3 /**< Fine structure constant, dimensionless */
#define LAL_RE_SI     2.81794092e-15 /**< Classical electron radius, m */
#define LAL_LAMBDAE_SI 3.86159323e-13 /**< Electron Compton wavelength, m */
#define LAL_AB_SI     5.29177249e-11 /**< Bohr radius, m */
#define LAL_MUB_SI    9.27401543e-24 /**< Bohr magneton, J T^-1 */
#define LAL_MUN_SI    5.05078658e-27 /**< Nuclear magneton, J T^-1 */
/*@}*/

/** \name Astrophysical parameters
 * The following parameters are derived from measured properties of the
 * Earth and Sun.  The values are taken from \ref Barnet_1996, except
 * for the obliquity of the ecliptic plane and the eccentricity of
 * Earth's orbit, which are taken from \ref Lang_1992.  All values are
 * given in the SI units shown.  Note that the ``year'' and
 * ``light-year'' have exactly defined values, and appear under
 * ``Exact physical constants''.
 */
/*@{*/
#define LAL_REARTH_SI 6.378140e6      /**< Earth equatorial radius, m */
#define LAL_AWGS84_SI 6.378137e6      /**< Semimajor axis of WGS-84 Reference Ellipsoid, m */
#define LAL_BWGS84_SI 6.356752314e6   /**< Semiminor axis of WGS-84 Reference Ellipsoid, m */
#define LAL_MEARTH_SI 5.97370e24      /**< Earth mass, kg */
#define LAL_IEARTH    0.409092804     /**< Earth inclination (2000), radians */
#define LAL_EEARTH    0.0167          /**< Earth orbital eccentricity */
#define LAL_RSUN_SI   6.960e8         /**< Solar equatorial radius, m */
#define LAL_MSUN_SI   1.98892e30      /**< Solar mass, kg */
#define LAL_MRSUN_SI  1.47662504e3    /**< Geometrized solar mass, m */
#define LAL_MTSUN_SI  4.92549095e-6   /**< Geometrized solar mass, s */
#define LAL_LSUN_SI   3.846e26        /**< Solar luminosity, W */
#define LAL_AU_SI     1.4959787066e11 /**< Astronomical unit, m */
#define LAL_PC_SI     3.0856775807e16 /**< Parsec, m */
#define LAL_YRTROP_SI 31556925.2      /**< Tropical year (1994), s */
#define LAL_YRSID_SI  31558149.8      /**< Sidereal year (1994), s */
#define LAL_DAYSID_SI 86164.09053     /**< Mean sidereal day, s */
/*@}*/

/** \name Cosmological parameters
 * The following cosmological parameters are derived from measurements of
 * the Hubble expansion rate and of the cosmic background radiation
 * (CBR).  Data are taken from \ref Barnet_1996.  In what follows, the
 * normalized Hubble constant \f$h_0\f$ is equal to the actual Hubble
 * constant \f$H_0\f$ divided by \f$\langle H
 * \rangle=100\,\mathrm{km}\,\mathrm{s}^{-1}\mathrm{Mpc}^{-1}\f$.  Thus the
 * Hubble constant can be written as:
 * \f$H_0 = \langle H \rangle h_0\f$.
 * Similarly, the critical energy density \f$\rho_c\f$ required for spatial
 * flatness is given by: \f$\rho_c = \langle\rho\rangle h_0^2\f$.
 * Current estimates give \f$h_0\f$ a value of around 0.65, which is what is
 * assumed below.  All values are in the SI units shown. */
/*@{*/
#define LAL_H0FAC_SI  3.2407792903e-18 /**< Hubble constant prefactor, s^-1 */
#define LAL_H0_SI     2e-18            /**< Approximate Hubble constant, s^-1 */
/* Hubble constant H0 = h0*HOFAC, where h0 is around 0.65 */
#define LAL_RHOCFAC_SI 1.68860e-9   /**< Critical density prefactor, J m^-3 */
#define LAL_RHOC_SI   7e-10         /**< Approximate critical density, J m^-3 */
/* Critical density RHOC = h0*h0*RHOCFAC, where h0 is around 0.65 */
#define LAL_TCBR_SI   2.726   /**< Cosmic background radiation temperature, K */
#define LAL_VCBR_SI   3.695e5 /**< Solar velocity with respect to CBR, m s^-1 */
#define LAL_RHOCBR_SI 4.177e-14 /**< Energy density of CBR, J m^-3 */
#define LAL_NCBR_SI   4.109e8   /**< Number density of CBR photons, m^-3 */
#define LAL_SCBR_SI   3.993e-14 /**< Entropy density of CBR, J K^-1 m^-3 */
/*@}*/

#define XLAL_EPOCH_UNIX_GPS 315964800
    
#define XLAL_EPOCH_J2000_0_JD 2451545.0         /**< Julian Day (UTC) of the J2000.0 epoch (2000 JAN 1 12h UTC). */
#define XLAL_EPOCH_J2000_0_TAI_UTC 32           /**< Leap seconds (TAI-UTC) on the J2000.0 epoch (2000 JAN 1 12h UTC). */
#define XLAL_EPOCH_J2000_0_GPS 630763213        /**< GPS seconds of the J2000.0 epoch (2000 JAN 1 12h UTC). */
#define XLAL_EPOCH_GPS_JD 2444244.5             /**< Julian Day (UTC) of the GPS epoch (1980 JAN 6 0h UTC) */
#define XLAL_EPOCH_GPS_TAI_UTC 19               /**< Leap seconds (TAI-UTC) on the GPS epoch (1980 JAN 6 0h UTC) */
#define XLAL_MJD_REF 2400000.5                  /**< Reference Julian Day for Mean Julian Day. */
#define XLAL_JD_TO_MJD(jd) ((jd) - XLAL_MJD_REF) /**< Modified Julian Day for specified civil time structure. */
    
/*@}*/
#ifdef  __cplusplus
}
#endif

#endif /* _LALCONSTANTS_H */
