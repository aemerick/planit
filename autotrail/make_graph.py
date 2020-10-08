"""

    Author  : Andrew Emerick
    e-mail  : aemerick11@gmail.com
    year    : 2020

    LICENSE :GPLv3


    NOTE: This code was used for plotting of the OLD Boulder, CO datasets
    and is now defunct after the switch to OSM data.

    Auto-generated hiking and trail running routes in Boulder, CO
    based on known trail data and given user-specified contraints.
"""

import geopandas as geopd
import os, sys


from planit.autotrail import process_gpx_data as gpx_process


def generate_segmented_dataset(inname ='./data/Boulder_Area_Trails.geojson',
                               outname = './data/boulder_area_trail_processed'):
    """
    Given an input geojson file and output file name, loads the Boulder
    area input file and passes info to the gpx_process function to do the
    heavy lifting of processing the trail data into segments (edges) and nodes.
    """

    if not (os.path.isfile(inname)):
        print("Raw dataset not found: " + inname)
        raise RuntimeError

    gdf = geopd.read_file(inname)

    segmented_df, nodes, edges = gpx_process.process_data(gdf,
                                                          outname = outname,
                                                          threshold_distance=4.0)

    return segmented_df, nodes, edges

def make_graph(outname='./data/boulder_area_trail_processed',
               raw_data = './data/Boulder_Area_Trails.geojson',
               from_scratch=True):
    """
    Generate the new dataset containing fully segmented trail data
    from the Boulder CO trail dataset AND the TrailMap object containing
    the graph
    """


    if from_scratch:
        segment_df, nodes, edges = generate_segmented_dataset(raw_data, outname)
    else:
        if not os.path.isfile(outname + '.geojson'):
            print("Data file not found: ", outname + '.geojson')
            raise RuntimeError

        segment_df, nodes, edges = gpx_process.load_trail_df(outname)

    trail_map = gpx_process.make_trail_map(segment_df, nodes, edges, outname = outname)

    return segment_df, trail_map


if __name__ == "__main__":

    from_scratch = True
    if len(sys.argv) > 1:
        from_scratch = (sys.argv[1] == "True")


    outname='./data/boulder_area_trail_processed'
    raw_data = './data/Boulder_Area_Trails.geojson'

    if len(sys.argv) > 2:
        if sys.argv[2] == 'small':

            raw_data = './data/Trails.geojson'
            outname = './data/small_trails_processed'

    make_graph(from_scratch = from_scratch, outname = outname, raw_data=raw_data)
