LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error
MAIN    = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex sections/*.tex bib/*.bib
	$(LATEXMK) $(MAIN).tex

check:
	@grep -c 'TODO' sections/*.tex || true
	@grep -E 'Warning--|undefined' $(MAIN).blg $(MAIN).log 2>/dev/null | sort | uniq -c | head -40 || true

arxiv: $(MAIN).pdf
	rm -rf arxiv && mkdir -p arxiv/sections arxiv/figures
	cp $(MAIN).tex $(MAIN).bbl arxiv/
	cp sections/*.tex arxiv/sections/
	cp figures/*.pdf figures/*.png arxiv/figures/ 2>/dev/null || true
	cd arxiv && tar czf ../alibz-arxiv.tar.gz .
	@echo "arXiv bundle: alibz-arxiv.tar.gz"

clean:
	latexmk -C $(MAIN).tex
	rm -rf arxiv alibz-arxiv.tar.gz $(MAIN).bbl

.PHONY: all check arxiv clean
