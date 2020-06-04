import warnings
import argparse
import os
import logging
import sys
from src.geno2pheno.pipeline import Geno2PhenoPipeline
def checkArgs(args):
    '''
        This function checks the input arguments and returns the errors (if exist) otherwise reads the parameters
    '''
    # keep all errors
    err = "";
    # Using the argument parser in case of -h or wrong usage the correct argument usage
    # will be prompted
    parser = argparse.ArgumentParser()

    # parse #################################################################################################
    parser.add_argument('--genyml', action='store', dest='genyml_path', default=False, type=str,
                        help='Provide the path to the GenYML file (the pipeline config)')

    parser.add_argument('--overwrite', action='store', dest='overwrite',default=1, type=int,
                        help='Would you like to overwrite the existing files?')

    parser.add_argument('--cores', action='store', dest='cores',default=4, type=int,
                        help='Number of cores to be used')

    parsedArgs = parser.parse_args()

    if (not os.access(parsedArgs.genyml_path, os.F_OK)):
        err = err + "\nError: Permission denied or could not find the labels!"
        return err
    G2P = Geno2PhenoPipeline(parsedArgs.genml_path, parsedArgs.overwrite)
    return False


def fxn():
    warnings.warn("deprecated", DeprecationWarning)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
    err = checkArgs(sys.argv)
    if err:
        print(err)
        exit()
