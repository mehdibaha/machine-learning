MD := $(wildcard *.md)
default: $(MD:.md=.pdf)

%.pdf : %.md
	@echo "$*.pdf generated"
	@pandoc -o $*.pdf $< --latex-engine=xelatex --variable=geometry:"margin=3cm"

.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -f *.pdf
