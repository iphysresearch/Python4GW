// $Id: Panyimain.cpp,v 1.1.1.1 2016/12/30 06:03:09 zjcao Exp $
#include <stdio.h>
#include <time.h>
//#include <ctime>
#include <string.h>
#include <stdlib.h>
#include "Panyidatatypes.h"
#include "PanyiLALConstants.h"
#include "Panyicomm.h"
#include "Panyi.h"




const char * usage =
"Generate a simulation using the lalsimulation library\n\n"
"The following options can be given (will assume a default value if omitted):\n"
//"--domain DOM               'TD' for time domain (default) or 'FD' for frequency\n"
//"                           domain; not all approximants support all domains\n"
"--amp-phase                If given, will output:\n"
"                           |h+ - i hx|, Arg(h+ - i hx) (TD) or\n"
"                           |h+(f)|, Arg(h+(f)), |hx(f)|, Arg(hx(f)) (FD)\n"
"                           If not given, will output h+ and hx (TD and FD)\n"

"                           NOTE: Other approximants may be available if the\n"
"                           developer forgot to edit this help message\n"
//"--phase-order ORD          Twice PN order of phase (default ORD=7 <==> 3.5PN)\n"
"--amp-order ORD            Twice PN order of amplitude (default 0 <==> Newt.)\n"
"--phiRef phiRef            Phase at the reference frequency (default 0)\n"
//"--fRef FREF                Reference frequency in Hz\n"
"                           (default: 0)\n"
"--sample-rate SRATE        Sampling rate of TD approximants in Hz (default 4096)\n"
//"--deltaF DF                Frequency bin size for FD approximants in Hz (default 1/8)\n"
"--m1 M1                    Mass of the 1st object in solar masses (default 10)\n"
"--m2 M2                    Mass of the 2nd object in solar masses (default 1.4)\n"
"--inclination IOTA         Angle in radians between line of sight (N) and \n"
"                           orbital angular momentum (L) at the reference\n"
"                           (default 0, face on)\n"
"--spin1x S1X               Vector components for spin of mass1 (default all 0)\n"
"--spin1y S1Y               z-axis=line of sight, L in x-z plane at reference\n"
"--spin1z S1Z               Kerr limit: s1x^2 + s1y^2 + s1z^2 <= 1\n"
"--spin2x S2X               Vector components for spin of mass2 (default all 0)\n"
"--spin2y S2Y               z-axis=line of sight, L in x-z plane at reference\n"
"--spin2z S2Z               Kerr limit: s2x^2 + s2y^2 + s2z^2 <= 1\n"
//"--tidal-lambda1 L1         (tidal deformability of mass 1) / (mass of body 1)^5\n"
//"                           (~128-2560 for NS, 0 for BH) (default 0)\n"
//"--tidal-lambda2 L2         (tidal deformability of mass 2) / (mass of body 2)^5\n"
//"                           (~128-2560 for NS, 0 for BH) (default 0)\n"
//"--spin-order ORD           Twice PN order of spin effects\n"
//"                           (default ORD=-1 <==> All spin effects)\n"
//"--tidal-order ORD          Twice PN order of tidal effects\n"
//"                           (default ORD=-1 <==> All tidal effects)\n"
"--f-min FMIN               Lower frequency to start waveform in Hz (default 40)\n"
"--f-max FMAX               Frequency at which to stop waveform in Hz\n"
"                           (default: generate as much as possible)\n"
"--distance D               Distance in Mpc (default 100)\n"
//"--axis AXIS                for PhenSpin: 'View' (default), 'TotalJ', 'OrbitalL'\n"
"--outname FNAME            Output to file FNAME (default 'simulation.dat')\n"
"--verbose                  If included, add verbose output\n"
;

/* Parse command line, sanity check arguments, and return a newly
 * allocated GSParams object */
GSParams *parse_args(unsigned int argc, char **argv) {
    unsigned int i;
    GSParams *params;
    params = (GSParams *) malloc(sizeof(GSParams));
    memset(params, 0, sizeof(GSParams));

    /* Set default values to the arguments */
    params->ampO = 0;
    params->phiRef = 0.;
    params->deltaT = 1./4096.;
    params->m1 = 10 * LAL_MSUN_SI;
    params->m2 = 1.4 * LAL_MSUN_SI;
    params->f_min = 40;
    params->f_max = 0.; /* Generate as much as possible */
    params->distance = 100. * 1e6 * LAL_PC_SI;
    params->inclination = 0.;
    params->s1x = 0.;
    params->s1y = 0.;
    params->s1z = 0.;
    params->s2x = 0.;
    params->s2y = 0.;
    params->s2z = 0.;
    params->verbose = 1; /* No verbosity */

    strncpy(params->outname, "simulation.dat", 256); /* output to this file */
    params->ampPhase = 0; /* output h+ and hx */

    /* consume command line */
    for (i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-h") == 0) || (strcmp(argv[i], "--help") == 0)) {
            goto fail;
        
        } else if (strcmp(argv[i], "--amp-order") == 0) {
            params->ampO = atof(argv[++i]);
        } else if (strcmp(argv[i], "--phiRef") == 0) {
            params->phiRef = atof(argv[++i]);
        } else if (strcmp(argv[i], "--sample-rate") == 0) {
            params->deltaT = 1./atof(argv[++i]);
        } else if (strcmp(argv[i], "--m1") == 0) {
            params->m1 = atof(argv[++i]) * LAL_MSUN_SI;
        } else if (strcmp(argv[i], "--m2") == 0) {
            params->m2 = atof(argv[++i]) * LAL_MSUN_SI;
        } else if (strcmp(argv[i], "--spin1x") == 0) {
            params->s1x = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin1y") == 0) {
            params->s1y = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin1z") == 0) {
            params->s1z = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin2x") == 0) {
            params->s2x = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin2y") == 0) {
            params->s2y = atof(argv[++i]);
        } else if (strcmp(argv[i], "--spin2z") == 0) {
            params->s2z = atof(argv[++i]);
        } else if (strcmp(argv[i], "--f-min") == 0) {
            params->f_min = atof(argv[++i]);
        } else if (strcmp(argv[i], "--f-max") == 0) {
            params->f_max = atof(argv[++i]);
        } else if (strcmp(argv[i], "--distance") == 0) {
            params->distance = atof(argv[++i]) * 1e6 * LAL_PC_SI;
        } else if (strcmp(argv[i], "--inclination") == 0) {
            params->inclination = atof(argv[++i]);
        } else if (strcmp(argv[i], "--outname") == 0) {
            strncpy(params->outname, argv[++i], 256);
        } else if (strcmp(argv[i], "--amp-phase") == 0) {
            params->ampPhase = 1;
        } else if (strcmp(argv[i], "--verbose") == 0) {
            params->verbose = 1;
        } else {
            printf("Error: invalid option: %s\n", argv[i]);
            goto fail;
        }
    }

    //printf("M=m1+m2=%e(s), M/R=%e\n",(params->m1+params->m2)/LAL_MSUN_SI*LAL_MTSUN_SI,(params->m1+params->m2)/LAL_MSUN_SI*LAL_MRSUN_SI/params->distance);

    return params;

    fail:
    printf("%s", usage);
    exit(1);
}
int dump_TD(FILE *f, REAL8TimeSeries *hplus, REAL8TimeSeries *hcross) {
    size_t i;
    REAL8 t0 = XLALGPSGetREAL8(&(hplus->epoch));
    if (hplus->data->length != hcross->data->length) {
        printf("Error: hplus and hcross are not the same length\n");
        return 1;
    } else if (hplus->deltaT != hcross->deltaT) {
        printf("Error: hplus and hcross do not have the same sample rate\n");
        return 1;
    }

    fprintf(f, "# t hplus hcross\n");
    for (i=0; i < hplus->data->length; i++)
        fprintf(f, "%.16e %.16e %.16e\n", t0 + i * hplus->deltaT, 
                hplus->data->data[i], hcross->data->data[i]);

    return 0;
}

int dump_convertTD(GSParams *params, REAL8TimeSeries *hplus, REAL8TimeSeries *hcross) {
    size_t i;
    if (hplus->data->length != hcross->data->length) {
        printf("Error: hplus and hcross are not the same length\n");
        return 1;
    } else if (hplus->deltaT != hcross->deltaT) {
        printf("Error: hplus and hcross do not have the same sample rate\n");
        return 1;
    }

    const REAL8 UM = (params->m1+params->m2)/LAL_MSUN_SI*LAL_MTSUN_SI;
    const REAL8 UMoR = (params->m1+params->m2)/LAL_MSUN_SI*LAL_MRSUN_SI/params->distance;

    FILE *f;
    f = fopen(params->outname, "w");
    fprintf(f, "# t hplus hcross\n");
    for (i=0; i < hplus->data->length; i++)
        // convert to [M] for t and [M/R] for h
        fprintf(f, "%.16e %.16e %.16e\n", (i * hplus->deltaT)/UM, 
                (hplus->data->data[i])/UMoR, (hcross->data->data[i])/UMoR);
    
    fclose(f);

    return 0;
}

int main (int argc , char **argv) {

	FILE *f;

    int start_time;
    REAL8TimeSeries *hplus = NULL;
    REAL8TimeSeries *hcross = NULL;
    GSParams *params;


    /* parse commandline */
    params = parse_args(argc, argv);

    /* generate waveform */
    // start_time = time(NULL);
 
    XLALSimInspiralChooseTDWaveform(&hplus, &hcross, params->phiRef, 
                    params->deltaT, params->m1, params->m2, params->s1x, 
                    params->s1y, params->s1z, params->s2x, params->s2y, 
                    params->s2z, params->f_min, 
                    params->distance, params->inclination
                    );
          
    // if (params->verbose)
    //     printf("Generation took %.0f seconds\n", 
    //             difftime(time(NULL), start_time));
  

    /* dump file */
    if(1)
    {
        // printf("here\n");
      f = fopen(params->outname, "w");
      dump_TD(f, hplus, hcross);
      fclose(f);
    }
    else
      dump_convertTD(params,hplus, hcross);

    return 1;
}

