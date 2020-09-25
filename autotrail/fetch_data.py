"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3

    Fetch data needed for this project. All data originates from the
    Boulder County Geospatial Open Data Site:
    https://gis-bouldercounty.opendata.arcgis.com/
"""
import numpy as np
import glob
import os
import requests
import json

# leaving data api urls here as globals
# to make clear
_simple_data_url = "https://gis-bouldercounty.opendata.arcgis.com/datasets/6ecc28f8192a43888b0a4d4924bcd64c_0.geojson"
_full_data_url   = "https://gis-bouldercounty.opendata.arcgis.com/datasets/3ed1045255c34eada007840a5dd84de9_0.geojson"



def fetch_data(outdir = './data'):
    """
    Get ALL of the data files needed for this project using calls to the
    urls provided by the Boudler County Geospatial Open Data Site.

    Downloads two datasets:
        1) Boulder Area Trails (./data/Boulder_Area_Trails.geojson): the full,
           larger dataset. 

        2) Trails (./data/Trails.geojson): smaller dataset containing only trails
           owned and operated by Boulder County. Used for testing.
    """

    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    #
    # First dataset - simplified trails used for testing
    #
    r = requests.get(_simple_data_url)
    data = r.json()
    with open(outdir + '/Trails.geojson', 'w') as f:
        json.dump(data, f)

    #
    # Second dataset - the full thing
    #
    r = requests.get(_full_data_url)
    data = r.json()
    with open(outdir + '/Boulder_Area_Trails.geojson', 'w') as f:
        json.dump(data, f)

    return

if __name__ == "__main__":

    fetch_data()
