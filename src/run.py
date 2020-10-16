import os
import sys
import zipfile
import fnmatch
import os.path
import re
import subprocess
from pdf2image import convert_from_path
import shutil
import smtplib
from email.message import EmailMessage
import email
import string
import time

GENO2PHENO_DIR = os.path.dirname(__file__) + "/../"
sys.path.append(GENO2PHENO_DIR)
from src.geno2pheno.pipeline import Geno2PhenoPipeline

def run_test():

    if len(sys.argv) == 5:
        email_supplied = True
    else:
        email_supplied = False

    if email_supplied:
        try:
            send_email(sys.argv[4], "start")
        except:
            pass

    prepare(sys.argv[1])
    os.chdir('/tmp/unpack')
    geno2pheno = Geno2PhenoPipeline(F"config.yaml", overwrite=True, cores=30, replace_list=[])
    generate(sys.argv[2], sys.argv[3])
    time.sleep(120)
    shutil.rmtree('/tmp/unpack')

    if email_supplied:
       try:
           send_email(sys.argv[4], "end")
       except:
           pass


def prepare(zip):
    os.mkdir('/tmp/unpack')
    zipf = zipfile.ZipFile(zip, 'r')
    zipf.extractall('/tmp/unpack')

def generate(html_file, html_dir):

    os.mkdir(html_dir)


    pickleZipDir = '/tmp/unpack/output/classification/amr_pred'
    pickleZipFile = html_dir + '/genopheno_pickle.zip'
    htmlFile = open(html_file, "w+")
   
    htmlFile.write("<p><center><strong>GenoPheno</strong></center></p>\n")
    htmlFile.write("<p><a href=\"genopheno_output.zip\">Click to download a ZIP archive of all PDF / XLS files</a></p>\n")
    htmlFile.write("<p><a href=\"genopheno_pickle.zip\">Click to download a ZIP archive of all Pickle files</a></p>\n")
    htmlFile.write("<p>Individual PDF files below can also be downloaded individually by clicking on them.</p>\n")
 
    pickleIncludes = ['*.pickle']
    pickleIncludes = r'|'.join([fnmatch.translate(x) for x in pickleIncludes])

    os.chdir(pickleZipDir)
    zipf = zipfile.ZipFile(pickleZipFile, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(pickleZipDir):
        for file in files:
            if re.match(pickleIncludes, file):
                zipf.write(file)
    zipf.close()

    zipDir = '/tmp/unpack/output/reports/classification'
    zipFile = html_dir + '/genopheno_output.zip'
   
    includes = ['*.pdf', '*.xls']
    includes = r'|'.join([fnmatch.translate(x) for x in includes])

    includes_only_pdf = ['*.pdf']
    includes_only_pdf = r'|'.join([fnmatch.translate(x) for x in includes_only_pdf])

    os.chdir(zipDir)
    zipf = zipfile.ZipFile(zipFile, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(zipDir):
        for file in files:
            if re.match(includes_only_pdf, file):
                shutil.copy(file, html_dir)
                pages = convert_from_path(zipDir+'/'+file, size=(1100, None))
                for page in pages:
                    page.save(html_dir+'/'+file+'.png', 'PNG')
                    htmlFile.write(file+'.png\n')
                    htmlFile.write("<a target=\"_blank\" href=\""+file+"\"><img src=\""+file+".png\" /></a>\n")
            if re.match(includes, file):
                zipf.write(file)
    zipf.close()
    htmlFile.close()
     	
def send_email(recipient, mode):
    
    TO = recipient.replace("__at__", "@")
    HOST = "my-smtp:25"
    FROM = "bifo-server@helmholtz-hzi.de"

    if mode == 'start': 
        SUBJECT = "Job started"
        TEXT = "Job started"
    else:
        SUBJECT = "Job finished"
        TEXT = "Job finished"

    SEQUENCE = ["From: %s" % FROM, "To: %s" % TO, "Subject: %s" % SUBJECT , "", TEXT]
    BODY = "\r\n".join(SEQUENCE)

    server = smtplib.SMTP(HOST, timeout=10)
    server.sendmail(FROM, [TO], BODY)
    server.quit()

run_test()
