import os, sys, argparse, time
import pandas as pd
import numpy as np
import pickle as pkl
import ESmodules as es

GeneSetFile = "f8300.geneset.gmt"

def run(ExpFile, output):
    ex_ = pd.read_csv(ExpFile, index_col = 0)
    es_ = es.GetES_DF(Exp=ex_, gmtFile=GeneSetFile, core=6)
    es_.to_csv(output)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Make Enrichment Score Feature')

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='expression file(.csv) with Entrez dene id')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='output file(.csv)')

    args = parser.parse_args()
    
    run(args.input, args.output)