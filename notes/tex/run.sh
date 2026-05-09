# /bin/bash

set -e

pdflatex $1
pdf2svg $2.pdf $2.svg
xdg-open $2.svg
