#!/usr/bin/python3

from argparse import ArgumentParser
from haversine_function import hDist as haversine
import numpy as np
import sys

import NGA_location as ngaloc
import NGA_population as ngapop


def main(outfile, csv_file, locations_file, kappa, rho):

    print('Writing contagion transfer matrix to {0}'.format(outfile))
    print('Writing sorted node locations to {0}'.format(locations_file))
    lgas = sorted([lga for lga in ngaloc.dataDic.keys() if lga.count(':') == 4])

    with open(locations_file, 'w') as handle:
        index = 0
        handle.write('id,name,lat,long\n')
        for lga in lgas:
            long, lat = ngaloc.dataDic[lga]
            handle.write('{0},"{1}",{2},{3}\n'.format(index, lga, lat, long))
            index += 1

    count = len(lgas)
    gravity = np.zeros((count, count), dtype=np.float32)

    for source in range(count - 1):     # last lga has all distances already calculated
        long1, lat1 = ngaloc.dataDic[lgas[source]]
        for destination in range(source + 1, count):
            long2, lat2 = ngaloc.dataDic[lgas[destination]]
            distance = haversine(lat1, long1, lat2, long2)
            factor = kappa / pow(distance, rho)
            gravity[source, destination] = factor
            gravity[destination, source] = factor

    gravity.tofile(outfile)

    if csv_file is not None:
        with open(csv_file, 'w') as handle:
            np.savetxt(handle, gravity, fmt='%.5e', delimiter=',')

    return


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-o', '--outfile', default='nga-gravity.bin')
    parser.add_argument('-c', '--csv', default=None)
    parser.add_argument('-l', '--locations', default='node-locations.csv')
    parser.add_argument('-k', '--kappa', type=float, default=0.5)
    parser.add_argument('-r', '--rho', type=float, default=1.40)

    args = parser.parse_args()

    main(args.outfile, args.csv, args.locations, args.kappa, args.rho)

    sys.exit(0)
