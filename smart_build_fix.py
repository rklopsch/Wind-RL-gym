# use this script to run the smart build command from the command line
# python smart_build_fix.py build --device cpu --no_tf
import sys
import urllib.request

opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36')]
urllib.request.install_opener(opener)

from smartsim._core._cli.__main__ import main
sys.exit(main())