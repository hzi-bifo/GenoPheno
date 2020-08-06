import os
import sys
import zipfile
import fnmatch
import os.path
import re
import subprocess
from pdf2image import convert_from_path
import shutil

GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline

def run_test():
    prepare(sys.argv[1])
    os.chdir('/tmp/unpack')
    geno2pheno = Geno2PhenoPipeline(F"config.yaml", overwrite=True, cores=30, replace_list=[])
    generate(sys.argv[2], sys.argv[3])
    shutil.rmtree('/tmp/unpack')

def prepare(zip):
    os.mkdir('/tmp/unpack')
    zipf = zipfile.ZipFile(zip, 'r')
    zipf.extractall('/tmp/unpack')

def generate(html_file, html_dir):

    zipDir = '/tmp/unpack/output/reports/classification'
    zipFile = html_dir + '/genopheno_output.zip'
    htmlFile = open(html_file, "w+")
   
    htmlFile.write("<p><center><strong>GenoPheno</strong></center></p>\n")
    htmlFile.write("<p><a href=\"genopheno_output.zip\">Click to download a ZIP archive of all PDFs</a></p>\n")
    htmlFile.write("<p>Individual PDF files below can also be downloaded individually by clicking on them.</p>\n")
 
    includes = ['*.pdf']
    includes = r'|'.join([fnmatch.translate(x) for x in includes])

    os.chdir(zipDir)
    zipf = zipfile.ZipFile(zipFile, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(zipDir):
        for file in files:
            if re.match(includes, file):
                zipf.write(file)
                pages = convert_from_path(zipDir+'/'+file, 500)
                for page in pages:
                    page.save(html_dir+'/'+file+'.png', 'PNG')
                    htmlFile.write(file+'.png\n')
                    htmlFile.write("<a target=\"_blank\" href=\""+file+"\"><img src=\""+file+".png\" /></a>\n")
    zipf.close()
    htmlFile.close()
     	

run_test()
