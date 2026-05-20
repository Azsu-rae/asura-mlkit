# /bin/bash

set -e

pdflatex $1
pdf2svg $2.pdf nn/$2.svg
xdg-open nn/$2.svg

rm $2.pdf
