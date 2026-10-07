.PHONY: clean

# run.sh stays non-executable (in git and on disk): `viam module reload` archives the
# working tree with its file modes, and the cloud builder treats an executable
# entrypoint as a prebuilt module and skips this build step. We make it executable only for
# the duration of packaging (viam-server execs the entrypoint), then restore the
# mode so the working tree stays clean.
module.tar.gz: run.sh requirements.txt meta.json src/*.py
	chmod +x run.sh
	tar czf $@ $^
	chmod -x run.sh

clean:
	rm -f module.tar.gz