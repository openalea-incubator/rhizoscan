"""Script to run versionclimber from a configuration file.

Data are organised by Sequences (a time point) with different boxes.
We want to group the files by box with the different sequences (time points).


"""

from __future__ import absolute_import
from __future__ import print_function
import sys
import os
from optparse import OptionParser

from rhizodata import reorder, video


def main():
    """This function is called by rhizodata

    To obtain specific help, type::

        rhizodata --help


    """

    usage = """
rhizodata traverse the image files of a directory and reorder it.
Example

       rhizodata -i directory -o result_directory -p *.jpg

rhizodata can also print all the actions without processing it

        rhizodata -n -i directory -o result_directory -p *.jpg
"""

    parser = OptionParser(usage=usage)

    parser.add_option("-i", "--input", dest='input_dir', default='.',
        help="Input directory where the images are ")
    parser.add_option("-o", "--output", dest='output_dir', default='result',
        help="Output directory to store reordered image files")
    parser.add_option("-n", "--noprocessing", action="store_true", dest="simulate", default=False,
        help="Print all the actions without doing anything")
    parser.add_option("-p", "--pattern", dest="pattern", default="*.jpg",
        help="Image file pattern")

    (opts, args)= parser.parse_args()


    input_dir = opts.input_dir
    output_dir = opts.input_dir
    img_pattern = opts.pattern

    simulate = opts.simulate

    env = reorder.DataEnv(input_dir, output_dir, img_pattern, simulate)

