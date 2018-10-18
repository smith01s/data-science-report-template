build:
	pandoc report.md -o report.pdf \
	    -V lang=en-GB --number-sections --listings \
	    --highlight-style pygments \
	    --template ./templates/eisvogel.tex \
	    --latex-engine xelatex
clean:
	rm report.pdf
