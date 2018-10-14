#include <stdio.h>
#include <string.h>
#include "MYdatatypes.h"
#include "MYconstants.h"
# define my_index    strchr
#define SWAP_FLAGS(ch1, ch2)

static const char *_getopt_initialize (int, char *const *, const char *);
static void exchange (char **);
static char *posixly_correct;



static enum
{
    REQUIRE_ORDER, PERMUTE, RETURN_IN_ORDER
} ordering;

char *LALoptarg;
int LALoptind = 1;

static int __getopt_initialized;
static char *nextchar;
int LALopterr = 1;
int LALoptopt = '?';
static int first_nonopt;
static int last_nonopt;

static int
_getopt_internal (int argc, char *const *argv, const char *optstring,
                  const struct LALoption *longopts, int *longind, int long_only);

int
LALgetopt_long_only (int argc, char *const *argv, const char *options,
                     const struct LALoption *long_options, int *opt_index)
{
    return _getopt_internal (argc, argv, options, long_options, opt_index, 1);
}

static int
_getopt_internal (int argc, char *const *argv, const char *optstring,
                  const struct LALoption *longopts, int *longind, int long_only)
{
    int print_errors = LALopterr;
    if (optstring[0] == ':')
        print_errors = 0;
    
    if (argc < 1)
        return -1;
    
    LALoptarg = NULL;
    
    if (LALoptind == 0 || !__getopt_initialized)
    {
        if (LALoptind == 0)
            LALoptind = 1;    /* Don't scan ARGV[0], the program name.  */
        optstring = _getopt_initialize (argc, argv, optstring);
        __getopt_initialized = 1;
    }
    
    /* Test whether ARGV[LALoptind] points to a non-option argument.
     Either it does not have option syntax, or there is an environment flag
     from the shell indicating it is not an option.  The later information
     is only used when the used in the GNU libc.  */
# define NONOPTION_P (argv[LALoptind][0] != '-' || argv[LALoptind][1] == '\0')
    
    if (nextchar == NULL || *nextchar == '\0')
    {
        /* Advance to the next ARGV-element.  */
        
        /* Give FIRST_NONOPT & LAST_NONOPT rational values if LALOPTIND has been
         moved back by the user (who may also have changed the arguments).  */
        if (last_nonopt > LALoptind)
            last_nonopt = LALoptind;
        if (first_nonopt > LALoptind)
            first_nonopt = LALoptind;
        
        if (ordering == PERMUTE)
        {
            /* If we have just processed some options following some non-options,
             exchange them so that the options come first.  */
            union {char * const *pcs; char **ps;} bad = { argv };
            
            if (first_nonopt != last_nonopt && last_nonopt != LALoptind)
                exchange ((char **) bad.ps);
            else if (last_nonopt != LALoptind)
                first_nonopt = LALoptind;
            
            /* Skip any additional non-options
             and extend the range of non-options previously skipped.  */
            
            while (LALoptind < argc && NONOPTION_P)
                LALoptind++;
            last_nonopt = LALoptind;
        }
        
        /* The special ARGV-element `--' means premature end of options.
         Skip it like a null option,
         then exchange with previous non-options as if it were an option,
         then skip everything else like a non-option.  */
        
        if (LALoptind != argc && !strcmp (argv[LALoptind], "--"))
        {
            union {char * const *pcs; char **ps;} bad = { argv };
            LALoptind++;
            
            if (first_nonopt != last_nonopt && last_nonopt != LALoptind)
                exchange ((char **) bad.ps);
            else if (first_nonopt == last_nonopt)
                first_nonopt = LALoptind;
            last_nonopt = argc;
            
            LALoptind = argc;
        }
        
        /* If we have done all the ARGV-elements, stop the scan
         and back over any non-options that we skipped and permuted.  */
        
        if (LALoptind == argc)
        {
            /* Set the next-arg-index to point at the non-options
             that we previously skipped, so the caller will digest them.  */
            if (first_nonopt != last_nonopt)
                LALoptind = first_nonopt;
            return -1;
        }
        
        /* If we have come to a non-option and did not permute it,
         either stop the scan or describe it to the caller and pass it by.  */
        
        if (NONOPTION_P)
        {
            if (ordering == REQUIRE_ORDER)
                return -1;
            LALoptarg = argv[LALoptind++];
            return 1;
        }
        
        /* We have found another option-ARGV-element.
         Skip the initial punctuation.  */
        
        nextchar = (argv[LALoptind] + 1
                    + (longopts != NULL && argv[LALoptind][1] == '-'));
    }
    
    /* Decode the current option-ARGV-element.  */
    
    /* Check whether the ARGV-element is a long option.
     
     If long_only and the ARGV-element has the form "-f", where f is
     a valid short option, don't consider it an abbreviated form of
     a long option that starts with f.  Otherwise there would be no
     way to give the -f short option.
     
     On the other hand, if there's a long option "fubar" and
     the ARGV-element is "-fu", do consider that an abbreviation of
     the long option, just like "--fu", and not "-f" with arg "u".
     
     This distinction seems to be the most useful approach.  */
    
    if (longopts != NULL
        && (argv[LALoptind][1] == '-'
            || (long_only && (argv[LALoptind][2] || !my_index (optstring, argv[LALoptind][1])))))
    {
        char *nameend;
        const struct LALoption *p;
        const struct LALoption *pfound = NULL;
        int exact = 0;
        int ambig = 0;
        int indfound = -1;
        int option_index;
        
        for (nameend = nextchar; *nameend && *nameend != '='; nameend++)
        /* Do nothing.  */ ;
        
        /* Test all long options for either exact match
         or abbreviated matches.  */
        for (p = longopts, option_index = 0; p->name; p++, option_index++)
            if (!strncmp (p->name, nextchar, nameend - nextchar))
            {
                if ((unsigned int) (nameend - nextchar)
                    == (unsigned int) strlen (p->name))
                {
                    /* Exact match found.  */
                    pfound = p;
                    indfound = option_index;
                    exact = 1;
                    break;
                }
                else if (pfound == NULL)
                {
                    /* First nonexact match found.  */
                    pfound = p;
                    indfound = option_index;
                }
                else if (long_only
                         || pfound->has_arg != p->has_arg
                         || pfound->flag != p->flag
                         || pfound->val != p->val)
                /* Second or later nonexact match found.  */
                    ambig = 1;
            }
        
        if (ambig && !exact)
        {
            if (print_errors)
                fprintf (stderr, "%s: option `%s' is ambiguous\n",
                         argv[0], argv[LALoptind]);
            nextchar += strlen (nextchar);
            LALoptind++;
            LALoptopt = 0;
            return '?';
        }
        
        if (pfound != NULL)
        {
            option_index = indfound;
            LALoptind++;
            if (*nameend)
            {
                /* Don't test has_arg with >, because some C compilers don't
                 allow it to be used on enums.  */
                if (pfound->has_arg)
                    LALoptarg = nameend + 1;
                else
                {
                    if (print_errors)
                    {
                        if (argv[LALoptind - 1][1] == '-')
                        /* --option */
                            fprintf (stderr,
                                     "%s: option `--%s' doesn't allow an argument\n",
                                     argv[0], pfound->name);
                        else
                        /* +option or -option */
                            fprintf (stderr,
                                     "%s: option `%c%s' doesn't allow an argument\n",
                                     argv[0], argv[LALoptind - 1][0], pfound->name);
                    }
                    
                    nextchar += strlen (nextchar);
                    
                    LALoptopt = pfound->val;
                    return '?';
                }
            }
            else if (pfound->has_arg == 1)
            {
                if (LALoptind < argc)
                    LALoptarg = argv[LALoptind++];
                else
                {
                    if (print_errors)
                        fprintf (stderr,
                                 "%s: option `%s' requires an argument\n",
                                 argv[0], argv[LALoptind - 1]);
                    nextchar += strlen (nextchar);
                    LALoptopt = pfound->val;
                    return optstring[0] == ':' ? ':' : '?';
                }
            }
            nextchar += strlen (nextchar);
            if (longind != NULL)
                *longind = option_index;
            if (pfound->flag)
            {
                *(pfound->flag) = pfound->val;
                return 0;
            }
            return pfound->val;
        }
        
        /* Can't find it as a long option.  If this is not getopt_long_only,
         or the option starts with '--' or is not a valid short
         option, then it's an error.
         Otherwise interpret it as a short option.  */
        if (!long_only || argv[LALoptind][1] == '-'
            || my_index (optstring, *nextchar) == NULL)
        {
            union { const char *cs; char *c; } wtf = { "" };
            if (print_errors)
            {
                if (argv[LALoptind][1] == '-')
                /* --option */
                    fprintf (stderr, "%s: unrecognized option `--%s'\n",
                             argv[0], nextchar);
                else
                /* +option or -option */
                    fprintf (stderr, "%s: unrecognized option `%c%s'\n",
                             argv[0], argv[LALoptind][0], nextchar);
            }
            nextchar = wtf.c;
            LALoptind++;
            LALoptopt = 0;
            return '?';
        }
    }
    
    /* Look at and handle the next short option-character.  */
    
    {
        char c = *nextchar++;
        const char *temp = my_index (optstring, c);
        
        /* Increment `LALoptind' when we start to process its last character.  */
        if (*nextchar == '\0')
            ++LALoptind;
        
        if (temp == NULL || c == ':')
        {
            if (print_errors)
            {
                if (posixly_correct)
                /* 1003.2 specifies the format of this message.  */
                    fprintf (stderr, "%s: illegal option -- %c\n",
                             argv[0], c);
                else
                    fprintf (stderr, "%s: invalid option -- %c\n",
                             argv[0], c);
            }
            LALoptopt = c;
            return '?';
        }
        /* Convenience. Treat POSIX -W foo same as long option --foo */
        if (temp[0] == 'W' && temp[1] == ';')
        {
            char *nameend;
            const struct LALoption *p;
            const struct LALoption *pfound = NULL;
            int exact = 0;
            int ambig = 0;
            int indfound = 0;
            int option_index;
            
            /* This is an option that requires an argument.  */
            if (*nextchar != '\0')
            {
                LALoptarg = nextchar;
                /* If we end this ARGV-element by taking the rest as an arg,
                 we must advance to the next element now.  */
                LALoptind++;
            }
            else if (LALoptind == argc)
            {
                if (print_errors)
                {
                    /* 1003.2 specifies the format of this message.  */
                    fprintf (stderr, "%s: option requires an argument -- %c\n",
                             argv[0], c);
                }
                LALoptopt = c;
                if (optstring[0] == ':')
                    c = ':';
                else
                    c = '?';
                return c;
            }
            else
            /* We already incremented `LALoptind' once;
             increment it again when taking next ARGV-elt as argument.  */
                LALoptarg = argv[LALoptind++];
            
            /* LALoptarg is now the argument, see if it's in the
             table of longopts.  */
            
            for (nextchar = nameend = LALoptarg; *nameend && *nameend != '='; nameend++)
            /* Do nothing.  */ ;
            
            /* Test all long options for either exact match
             or abbreviated matches.  */
            for (p = longopts, option_index = 0; p->name; p++, option_index++)
                if (!strncmp (p->name, nextchar, nameend - nextchar))
                {
                    if ((unsigned int) (nameend - nextchar) == strlen (p->name))
                    {
                        /* Exact match found.  */
                        pfound = p;
                        indfound = option_index;
                        exact = 1;
                        break;
                    }
                    else if (pfound == NULL)
                    {
                        /* First nonexact match found.  */
                        pfound = p;
                        indfound = option_index;
                    }
                    else
                    /* Second or later nonexact match found.  */
                        ambig = 1;
                }
            if (ambig && !exact)
            {
                if (print_errors)
                    fprintf (stderr, "%s: option `-W %s' is ambiguous\n",
                             argv[0], argv[LALoptind]);
                nextchar += strlen (nextchar);
                LALoptind++;
                return '?';
            }
            if (pfound != NULL)
            {
                option_index = indfound;
                if (*nameend)
                {
                    /* Don't test has_arg with >, because some C compilers don't
                     allow it to be used on enums.  */
                    if (pfound->has_arg)
                        LALoptarg = nameend + 1;
                    else
                    {
                        if (print_errors)
                            fprintf (stderr, "\
                                               %s: option `-W %s' doesn't allow an argument\n",
                                     argv[0], pfound->name);
                        
                        nextchar += strlen (nextchar);
                        return '?';
                    }
                }
                else if (pfound->has_arg == 1)
                {
                    if (LALoptind < argc)
                        LALoptarg = argv[LALoptind++];
                    else
                    {
                        if (print_errors)
                            fprintf (stderr,
                                     "%s: option `%s' requires an argument\n",
                                     argv[0], argv[LALoptind - 1]);
                        nextchar += strlen (nextchar);
                        return optstring[0] == ':' ? ':' : '?';
                    }
                }
                nextchar += strlen (nextchar);
                if (longind != NULL)
                    *longind = option_index;
                if (pfound->flag)
                {
                    *(pfound->flag) = pfound->val;
                    return 0;
                }
                return pfound->val;
            }
            nextchar = NULL;
            return 'W';    /* Let the application handle it.   */
        }
        if (temp[1] == ':')
        {
            if (temp[2] == ':')
            {
                /* This is an option that accepts an argument optionally.  */
                if (*nextchar != '\0')
                {
                    LALoptarg = nextchar;
                    LALoptind++;
                }
                else
                    LALoptarg = NULL;
                nextchar = NULL;
            }
            else
            {
                /* This is an option that requires an argument.  */
                if (*nextchar != '\0')
                {
                    LALoptarg = nextchar;
                    /* If we end this ARGV-element by taking the rest as an arg,
                     we must advance to the next element now.  */
                    LALoptind++;
                }
                else if (LALoptind == argc)
                {
                    if (print_errors)
                    {
                        /* 1003.2 specifies the format of this message.  */
                        fprintf (stderr,
                                 "%s: option requires an argument -- %c\n",
                                 argv[0], c);
                    }
                    LALoptopt = c;
                    if (optstring[0] == ':')
                        c = ':';
                    else
                        c = '?';
                }
                else
                /* We already incremented `LALoptind' once;
                 increment it again when taking next ARGV-elt as argument.  */
                    LALoptarg = argv[LALoptind++];
                nextchar = NULL;
            }
        }
        return c;
    }
}


static const char *
_getopt_initialize (int argc, char *const *argv, const char *optstring)
{
    /* Start processing options with ARGV-element 1 (since ARGV-element 0
     is the program name); the sequence of previously skipped
     non-option ARGV-elements is empty.  */
    argc = 0;
    argv = NULL;
    
    first_nonopt = last_nonopt = LALoptind;
    
    nextchar = NULL;
    
    // RP 09/01/2014: the POSIXLY_CORRECT feature is undesired in lalsuite and was deactivated:
    // posixly_correct = getenv ("POSIXLY_CORRECT");
    
    /* Determine how to handle the ordering of options and nonoptions.  */
    
    if (optstring[0] == '-')
    {
        ordering = RETURN_IN_ORDER;
        ++optstring;
    }
    else if (optstring[0] == '+')
    {
        ordering = REQUIRE_ORDER;
        ++optstring;
    }
    else if (posixly_correct != NULL)
        ordering = REQUIRE_ORDER;
    else
        ordering = PERMUTE;
    
    return optstring;
    (void)argc;
    (void)argv;
}

static void
exchange (char **argv)
{
    int bottom = first_nonopt;
    int middle = last_nonopt;
    int top = LALoptind;
    char *tem;
    
    /* Exchange the shorter segment with the far end of the longer segment.
     That puts the shorter segment into the right place.
     It leaves the longer segment in the right place overall,
     but it consists of two parts that need to be swapped next.  */
    
    while (top > middle && middle > bottom)
    {
        if (top - middle > middle - bottom)
        {
            /* Bottom segment is the short one.  */
            int len = middle - bottom;
            register int i;
            
            /* Swap it with the top part of the top segment.  */
            for (i = 0; i < len; i++)
            {
                tem = argv[bottom + i];
                argv[bottom + i] = argv[top - (middle - bottom) + i];
                argv[top - (middle - bottom) + i] = tem;
                SWAP_FLAGS (bottom + i, top - (middle - bottom) + i);
            }
            /* Exclude the moved bottom segment from further swapping.  */
            top -= len;
        }
        else
        {
            /* Top segment is the short one.  */
            int len = top - middle;
            register int i;
            
            /* Swap it with the bottom part of the bottom segment.  */
            for (i = 0; i < len; i++)
            {
                tem = argv[bottom + i];
                argv[bottom + i] = argv[middle + i];
                argv[middle + i] = tem;
                SWAP_FLAGS (bottom + i, middle + i);
            }
            /* Exclude the moved top segment from further swapping.  */
            bottom += len;
        }
    }
    
    /* Update records for the slots the non-options now occupy.  */
    
    first_nonopt += (LALoptind - last_nonopt);
    last_nonopt = LALoptind;
}


