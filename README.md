# Plan-It 

For a high-level overview of Plan-It Trails: https://tinyurl.com/y2kysev8

An application to produce computer-generated hiking and trail running routes
given a series of user-provided contraints, geared towards the avid
hiker / running aiming for longer activities in areas with complex trail
networks.

Initial test case was for Boulder, CO using Boulder county's open access
trail data. However, this has since expanded to integrate with the entire
Open Street Map database and is able to generate trail routes anywhere where
OSM has recorded trail data (which is most places). This uses the OSMNx package
to request and download OSM data in a bounding box around a desired region.


Caching is currently implemented only if desired bounding box matches exactly with 
a previously downloaded box (filename matches desired bbox coordinates). OSM has
available a resource to locally cache and query the full OSM database, but this 
is not yet implemented to work with this application.

Plan-It trails is hosted here : planit-trails.website
(Please be gentle, trying to keep server costs low, but do let me 
know if it breaks: aemerick11@gmail.com)

Code for front-end here: https://github.com/aemerick/autotrail_app

## Installation:

To install planit trails, it should be sufficient to do:

$ pip install -r requirements.txt

which will install a majority of the dependencies. However, this 
relies on an open pull request from the SRTM package:

https://github.com/tkrajina/srtm.py/pull/39

Unfortunately this PR is a bit old by now. However, I have 
a version that is up-to-date with the main branch. You 
will need to download and install this as:

$ git clone https://github.com/aemerick/srtm.py ./srtm
$ cd srtm
$ git pull origin
$ git checkout pr39merge

Then to install, the following should work:

$ python setup.py develop

(depending on your particular environment setup). You may also
need to add the srtm directory to your PYTHONPATH environment
variable 
