#!/usr/bin/python3

import gpxpy
import gpxpy.gpx
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import argparse

parser = argparse.ArgumentParser(description='Scatter plots from gpx files.')
parser.add_argument('gpxfile', metavar='I', help='path to gpx file')
parser.add_argument('refpoint', metavar='XXXXXXX.XXX', type=float, nargs=2, help='reference point (SWEREF99)')
parser.add_argument('--output', metavar='O', help='output file (none = show plot)')
args = parser.parse_args()


###################################################################

def eigsorted(cov):
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:,order]

def dist(x1, y1, x2, y2):
    return ((x2-x1)**2 + (y2-y1)**2)**(1/2.0)

###################################################################

ref_x = args.refpoint[0]
ref_y = args.refpoint[1]
gpx_file = open(args.gpxfile, 'r')


# Define projections

wgs84 = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')
sweref99 = pyproj.Proj(init='epsg:3006')

# Parse gpx
gpx = gpxpy.parse(gpx_file)

#Get points list
points = []

for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            points.append(point)

lat = [p.latitude for p in points]
lon = [p.longitude for p in points]
el  = [p.elevation for p in points]

#Transform to sweref
x, y = pyproj.transform(wgs84, sweref99, lon, lat)

#Get mean
mean_x = np.mean(x)
mean_y = np.mean(y)

# Center around mean
x = np.subtract(x, mean_x)
y = np.subtract(y, mean_y)

# Plot std dev ellipse
nstd = 2
ax = plt.subplot(111)

cov = np.cov(x, y)
vals, vecs = eigsorted(cov)
theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
w, h = 2 * nstd * np.sqrt(vals)
ell = Ellipse(xy=(np.mean(x), np.mean(y)),
      width=w, height=h,
      angle=theta, color='black')
ell.set_facecolor('none')
ax.add_artist(ell)

# Plot GPX points
plt.scatter(x, y, s=4)

#Plot mean point
plt.scatter(0, 0, marker='+', color='k', s=200) 

#Plot true location
plt.scatter(ref_x - mean_x, ref_y - mean_y, marker='x', color='r', s=200)


if (args.output):
    plt.savefig(args.output)
else:
    plt.show()


#Print stats
print("std dev x = " + str(np.std(x)))
print("std dev y = " + str(np.std(y)))

print("error = " + str(dist(mean_x, mean_y, ref_x, ref_y)))

