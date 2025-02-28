module.tar.gz: run.sh requirements.txt meta.json src/*.py *.so
	tar czf $@ $^