from __future__ import division

import numpy as np


def get_size_vgg_feat_map(input_W, input_H):
    output_W = input_W
    output_H = input_H
    for i in range(1,6):
        output_H = np.floor(output_H/2)
        output_W = np.floor(output_W/2)

    return output_W, output_H


def devide_line(total_line, region_line, ovr):
    # print "total_line: " + str(total_line)
    # print "region_line: " + str(region_line)

    start_locations = np.array([0.0])
    end_location = total_line - region_line

    # region_count = 1
    if not end_location == 0:
        start_locations = np.append(start_locations, end_location)
        # region_count = 2
        if not region_line - end_location > total_line * ovr:
            region_count = np.floor(end_location / (region_line * (1 - ovr))) + 2
            interval = end_location / (region_count - 1)
            start_locations = np.floor(np.arange(0, region_count) * interval)

    # print "region_count: " + str(region_count)

    return start_locations


def rmac_regions(W, H, L):
    mw = min(W, H)
    A = mw ** 2
    ovr = 0.3 # desired overlap of neighboring regions
    regions = []

    for l in range(1, L+1):
        ra = A * ((L + 1 - l) / L)
        rw = np.floor(np.sqrt(ra))
        for hy in devide_line(H, rw, ovr):
            for hx in devide_line(W, rw, ovr):
                R = np.array([hx, hy, rw, rw], dtype=np.int)
                if not min(R[2:]):
                    continue
                regions.append(R)

    regions = np.asarray(regions)

    return regions
