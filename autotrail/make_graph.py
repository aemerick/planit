import geopandas as geopd
import process_gpx_data as gpx_process
import os, sys

def generate_segmented_dataset(inname ='./data/Boulder_Area_Trails.geojson',
                               outname = './data/boulder_area_trail_processed'):
    """
    ---
    """

    if not (os.path.isfile(inname)):
        print("Raw dataset not found: " + inname)
        raise RuntimeError

    gdf = geopd.read_file(inname)

    segmented_df, nodes, edges = gpx_process.process_data(gdf,
                                                          outname = outname,
                                                          threshold_distance=8.0)

    return segmented_df, nodes, edges

def make_graph(outname='./data/boulder_area_trail_processed',
               from_scratch=True):
    """
    Generate the new dataset containing fully segmented trail data
    from the Boulder CO trail dataset AND the TrailMap object containing
    the graph
    """


    if from_scratch:
        segment_df, nodes, edges = generate_segmented_dataset()
    else:
        if not os.path.isfile(outname + '.geojson'):
            print("Data file not found: ", outname + '.geojson')
            raise RuntimeError

        segment_df, nodes, edges = gpx_process.load_trail_df(outname)

    trail_map = gpx_process.make_trail_map(segment_df, nodes, edges)

    return segment_df, trail_map


if __name__ == "__main__":

    from_scratch = True
    if len(sys.argv) > 1:
        from_scratch = (sys.argv[1] == "True")

    make_graph(from_scratch = from_scratch)
