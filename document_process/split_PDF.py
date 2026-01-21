# split_PDF.py FILENAME DESTNAME START END
#
# FILENAME: ../documents/令和7年度_学生便覧.pdf
# DESTNAME: ../documents/令和7年度_学生便覧_7.pdf
# START   : 6
# END     : 7

import pypdf
import sys

args = sys.argv

FILENAME = args[1]
DESTNAME = args[2]
START = int(args[3])
END = int(args[4])

writer = pypdf.PdfWriter()
writer.append(FILENAME, pages=pypdf.PageRange(f'{START}:{END}'))
writer.write(DESTNAME)
