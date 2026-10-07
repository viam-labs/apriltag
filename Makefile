.PHONY: clean

# run.sh is tracked non-executable in git so Viam cloud builds don't mistake it
# for a prebuilt module and skip the build step. We make it executable only for
# the duration of packaging (viam-server execs the entrypoint), then restore the
# mode so the working tree stays clean.
module.tar.gz: run.sh requirements.txt meta.json src/*.py
	chmod +x run.sh
	tar czf $@ $^
	chmod -x run.sh

clean:
	rm -f module.tar.gz