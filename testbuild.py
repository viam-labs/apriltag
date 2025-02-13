#!/usr/bin/env python3
"""
testbuild.py -- this is for testing that the apriltags library is usable from a pyinstaller build.
"""

import dt_apriltags as apriltag

def main():
    detector = apriltag.Detector(families="tag16h5")
    print(detector)

if __name__ == '__main__':
    main()
