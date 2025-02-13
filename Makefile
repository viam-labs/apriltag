dist/testbuild: testbuild.py
	pyinstaller --onefile --collect-binaries dt_apriltags testbuild.py
