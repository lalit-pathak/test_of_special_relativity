# test_of_special_relativity

To generate a modified TaylorF2 waveform (phasing given by eq. 4 in the paper), requires the following packages: numpy, scipy, pycbc.
These dependencies should all be installable with conda or pip. First we need to install pycbc python package and then install the extra waveform plugin to make the modified TaylorF2 in pycbc's available waveforms.

Install pycbc pacakge

    $ pip install pycbc

Clone this repository and install modified TaylorF2 waveform plugin

    $ git clone https://github.com/lalit-pathak/test_of_special_relativity.git
    $ cd test_waveform_plugin
    $ python3 setup.py install

Example to generate modified waveform 




