Please install pyparsing, numpy, python imaging library.

Please install Image Magick (to view images).

To install pyparsing, use easy_install
To install easy_install do:
sudo apt-get install python-setuptools
Then, for pyparsing, do:
easy_install pyparsing

numpy should be already there.

To install python imaging library (PIL), do
sudo apt-get install python-imaging

To install Image Magick, do
sudo apt-get install imagemagick (I think this should work)

To run the matrixTest.py, which is a bunch of unit tests, do
python matrixTest.py

To run transform4x4.py, make sure it is in the same directory as rotationerror.py
Do:
python transform4x4.py
when, prompted, type in the file name, like transform1.tf

To run draw2d.py, do
python draw2d.py xmin xmax ymin ymax xRes yRes < inputfile
So, an example would be:
python draw2d.py -2 2 -2 2 512 512 < binaryclock.2d

Then, to display the .ppm image, do
display binaryclock.ppm
