import numpy as np
import random

from skimage import io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries

import cv2
import maxflow
from scipy.spatial import Delaunay
import os
import math
import argparse


def scribble(osm, img_marking, a2_p):
    # generate points randomly
    point_x = []
    point_y = []
    for i in range(50):
        point_x.append(random.randrange(0, osm.shape[0], 1))
        point_y.append(random.randrange(0, osm.shape[1], 1))

    # draw lines based on the above points
    # background -> Blue
    blank_mask = np.zeros(img_marking.shape)
    for i in range(len(point_x)-1):
        blank_mask = cv2.line(blank_mask, (point_x[i], point_y[i]), (point_x[i+1], point_y[i+1]), color=(255, 0, 0), thickness=3)# 3 pixels
    # these lines should be outside the buffer of OSM
    kernels = np.ones((a2_p+3, a2_p+3))
    dilate_osm = cv2.dilate(osm, kernels)
    blank_mask[:, :, 0][dilate_osm > 128] = 0

    red = img_marking[:, :, 2]
    green = img_marking[:, :, 1]
    blue = img_marking[:, :, 0]

    # foreground -> Red
    red[osm > 128] = 255

    # combine the foreground (red) and background (blue)
    img_marking += blank_mask

    # change black to white
    for i in range(img_marking.shape[0]):
        for j in range(img_marking.shape[1]):
            if red[i, j] < 128 and green[i, j] < 128 and blue[i, j] < 128:
                red[i, j] = 255
                green[i, j] = 255
                blue[i, j] = 255

    return img_marking


# Calculate the superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img, superpix_dir):
    segments = slic(img, n_segments=400, compactness=20)
    if superpix_dir is not None:
        io.imsave(superpix_dir, mark_boundaries(img, segments))
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments == i), axis=1) for i in segments_ids])

    # Calculate color histograms for all superpixels; color histograms are 2D over H-S (from the HSV)
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv], [0, 1], np.uint8(segments == i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers, colors_hists, segments, tri.vertex_neighbor_vertices)


# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:, :, 0] != 255])# non-blue
    bg_segments = np.unique(superpixels[marking[:, :, 2] != 255])# non-red
    return (fg_segments, bg_segments)


# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids], axis=0)
    return h / h.sum()


# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask


# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])


# Calculate the probability of each superpixel using FCN prediction
def cal_fcn_prediction(fcn_pred, segments, segments_ids):
    bg_loss = np.array([np.mean(fcn_pred[segments == i], axis=0) for i in segments_ids])
    bg_loss /= 255
    return bg_loss


# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors, bg_loss):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr, indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg), 20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i, h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            if bg_loss is not None:
                fcn_bg = -np.log(bg_loss[i])
                fcn_fg = -np.log(1-bg_loss[i])
                g.add_tedge(nodes[i], fcn_fg+cv2.compareHist(fgbg_hists[0], h, hist_comp_alg), fcn_bg+cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))
            else:
                g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg), cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))
    g.maxflow()
    return g.get_grid_segments(nodes)


def buffer_inference(osm, buffer_mask, a1_p, a2_p):
    # background should be outside the buffer of OSM
    kernels = np.ones((a2_p-a1_p+1, a2_p-a1_p+1))
    dilate_osm = cv2.dilate(osm, kernels)

    buffer_mask[dilate_osm > 128] = 128
    buffer_mask[osm > 128] = 255
    return buffer_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Road Label Propagation Algorithm')
    parser.add_argument('--img_root', default='./data/train/sat/', type=str,
                        help='path to imagery')
    parser.add_argument('--osm_root', default='./data/train/osm/', type=str,
                        help='path to osm centerlines')
    parser.add_argument('--is_save_scri', default=True, type=bool,
                        help='whether scribbles should be saved (or not)')
    parser.add_argument('--marking_root', default='./data/train/scribble/', type=str,
                        help='path to scribbles')
    parser.add_argument('--is_save_superpix', default=True, type=bool,
                        help='whether superpixels should be saved (or not)')
    parser.add_argument('--superpix_root', default='./data/train/superpix/', type=str,
                        help='path to superpixels')
    parser.add_argument('--is_fcn_pred', default=False, type=bool,
                        help='whether FCN prediction involved in graph construction (or not) refer ScribbleSup')
    parser.add_argument('--pred_root', default='./data/train/pred/', type=str,
                        help='path to FCN prediction')
    parser.add_argument('--GSD', default=1.2, type=float,
                        help='meters per pixel')
    parser.add_argument('--a1', default=6, type=int,
                        help='the first buffer width (meters)')
    parser.add_argument('--a2', default=18, type=int,
                        help='the second buffer width (meters)')
    parser.add_argument('--output_graph_mask_root', default='./data/train/graph_mask/', type=str,
                        help='path to graph-based masks generated from Graph Construction')
    parser.add_argument('--output_buffer_mask_root', default='./data/train/buffer_mask/', type=str,
                        help='path to buffer-based masks generated from Buffer Inference')
    parser.add_argument('--output_proposal_mask_root', default='./data/train/proposal_mask/', type=str,
                        help='path to proposal masks')
    args = parser.parse_args()

    os.makedirs(args.output_buffer_mask_root, exist_ok=True)
    os.makedirs(args.output_graph_mask_root, exist_ok=True)
    os.makedirs(args.output_proposal_mask_root, exist_ok=True)

    name_list = os.listdir(args.img_root)
    for name in name_list:
        region_info = name[:-7]
        img = cv2.imread(args.img_root + region_info + 'sat.png', cv2.IMREAD_COLOR)
        osm = cv2.imread(args.osm_root + region_info + 'osm.png', cv2.IMREAD_GRAYSCALE)
        if np.sum(osm) < 128: # except for non-road
            continue
        #
        # Step 1: Buffer Inference
        #

        a1_p = math.ceil(args.a1 / args.GSD) # convert a1 to pixel format
        kernels = np.ones((a1_p, a1_p))
        osm = cv2.dilate(osm, kernels)

        ## Generate buffer_mask offered by osm
        a2_p = math.ceil(args.a2 / args.GSD)  # convert a2 to pixel format
        buffer_mask = np.zeros(osm.shape)
        buffer_inference(osm, buffer_mask, a1_p, a2_p)

        cv2.imwrite(args.output_buffer_mask_root + region_info + 'mask.png', buffer_mask)

        #
        # Step 2: Graph Construction
        #

        ## Scribble img_marking offered by osm
        img_marking = np.zeros(img.shape)
        scribble(osm, img_marking, a2_p)
        if args.is_save_scri:
            os.makedirs(args.marking_root, exist_ok=True)
            cv2.imwrite(args.marking_root + region_info + 'scri.png', img_marking)
        else:
            img_marking = cv2.imread(args.marking_root + region_info + 'scri.png', cv2.IMREAD_COLOR)

        ## Calculating superpixels over image
        if args.is_save_superpix:
            os.makedirs(args.superpix_root, exist_ok=True)
            superpix_dir = args.superpix_root + region_info + 'superpix.png'
        else:
            superpix_dir = None
        centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(img, superpix_dir)

        ## Calculating Foreground and Background superpixel IDs
        fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)

        ## Calculating color histograms for FG and BG, respectively
        fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
        bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

        norm_hists = normalize_histograms(color_hists)

        ## Calculate the probability of each superpixel using FCN prediction (or not)
        if args.is_fcn_pred:
            fcn_pred = cv2.imread(args.pred_root + region_info + 'pred.png', 0)
            bg_loss = cal_fcn_prediction(fcn_pred, superpixels, np.unique(superpixels))
        else:
            bg_loss = None

        ## Construct a graph that takes into account superpixel-to-superpixel interaction (smoothness term), as well as superpixel-FG/BG interaction (match term)
        graph_cut = do_graph_cut([fg_cumulative_hist, bg_cumulative_hist], [fg_segments, bg_segments], norm_hists, neighbors, bg_loss)

        graph_mask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut))
        ## Note that, graph_mask is bool, conver 1 to 255 and 0 will remain 0 for displaying purpose
        graph_mask = np.uint8(graph_mask * 255)

        cv2.imwrite(args.output_graph_mask_root + region_info + 'mask.png', graph_mask)

        #
        # Step 3: Integration
        #

        proposal_mask = np.zeros(buffer_mask.shape)
        proposal_mask[graph_mask > 128] = 128
        proposal_mask[buffer_mask == 128] = 128
        proposal_mask[buffer_mask == 255] = 255

        cv2.imwrite(args.output_proposal_mask_root + region_info + 'mask.png', proposal_mask)