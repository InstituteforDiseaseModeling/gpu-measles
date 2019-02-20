#!/usr/bin/python3

from argparse    import ArgumentParser
from collections import OrderedDict
from datetime    import datetime

from NGA_area       import dataDic as nga_area
from NGA_location   import dataDic as nga_loc
from NGA_population import dataDic as nga_pop

import json
import os
import platform
import sys

"""
{
    "Metadata": {
        "DateCreated": "Sun Sep 25 20:17:03 2011",
        "Tool": "convertdemog.py",
        "Author": "jsteinkraus",
        "IdReference": "Legacy",
        "NodeCount": 4
    },
    "Defaults": {
        "NodeAttributes": {
            "InitialPopulation": 1000000,
            "BirthRate": 8.877e-05,
        }, 
        "IndividualAttributes": {
            "AgeDistributionFlag": 3,            "AgeDistribution1": 0.0001096,    "AgeDistribution2": 25550,
            "PrevalenceDistributionFlag": 0,     "PrevalenceDistribution1": 0,     "PrevalenceDistribution2": 0,
            "SusceptibilityDistributionFlag": 0, "SusceptibilityDistribution1": 0, "SusceptibilityDistribution2": 0,
        }
    }, 
    "Nodes": [
        {
            "NodeID": 1,
            "NodeAttributes": { "Latitude": 25.567,  "Longitude": 85.1 }
        }
    ]
}
"""


def main(filename, author, population, birthrate):

    demographics = OrderedDict()
    metadata = OrderedDict()
    metadata['DateCreated'] = datetime.now().strftime('%a %b %d %H:%M:%S %Y')
    metadata['Tool'] = os.path.basename(__file__)
    metadata['Author'] = author
    metadata['IdReference'] = 'Legacy'
    metadata['NodeCount'] = len([key for key in nga_pop.keys() if key.count(':') == 4])
    demographics['Metadata'] = metadata
    default_node_attributes = OrderedDict()
    default_node_attributes['InitialPopulation'] = population
    default_node_attributes['BirthRate'] = birthrate
    default_individual_attributes = OrderedDict()
    defaults = OrderedDict()
    defaults['NodeAttributes'] = default_node_attributes
    defaults['IndividualAttributes'] = default_individual_attributes
    demographics['Defaults'] = defaults

    nodes = []
    for i, key in enumerate(sorted([key for key in nga_pop.keys() if key.count(':') == 4])):
        node = {'Name': key, 'NodeID': i+1, 'NodeAttributes': {'InitialPopulation':nga_pop[key][0], 'Latitude':nga_loc[key][1], 'Longitude':nga_loc[key][0]}}
        nodes.append(node)

    demographics['Nodes'] = nodes

    with open(filename, 'w') as handle:
        json.dump(demographics, handle, indent=2, separators=(',', ':'))

    return

if __name__ == "__main__":

    username = os.environ['USERNAME' if platform.system() == 'Windows' else 'USER']

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='demographics.json')
    parser.add_argument('-a', '--author', type=str, default=username)
    parser.add_argument('-p', '--population', type=int, default=1024)
    parser.add_argument('-b', '--birthrate', type=float, default=0.00003484673133016816)
    args = parser.parse_args()

    main(args.filename, args.author, args.population, args.birthrate)

    sys.exit(0)
