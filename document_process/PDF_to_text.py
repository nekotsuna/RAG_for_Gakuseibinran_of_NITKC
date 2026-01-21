# PDF_to_text.py FILENAME
#
# FILENAME: ../documents/令和7年度_学生便覧_学生生活.pdf

import sys
import pypdf
import re

args = sys.argv
FILENAME = args[1]

dest = re.search(r'.*\.', FILENAME).group() + 'txt'

reader = pypdf.PdfReader(FILENAME)
txt = ""

for i, page in enumerate(reader.pages):
  txt += page.extract_text()

with open(f"{dest}", 'w') as f:
  f.write(txt)
