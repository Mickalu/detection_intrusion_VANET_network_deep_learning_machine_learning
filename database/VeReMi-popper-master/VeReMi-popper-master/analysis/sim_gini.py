#!/usr/bin/python3

import json
import os
import itertools
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams.update({'font.size': 18})

import numpy
from lib.gini import gini
import sys

import argparse

parser = argparse.ArgumentParser(description='Take result files with ground truth, computes gini coefficient for FPR and FNR, plot for each sim, as well as aggregate.')
parser.add_argument('--source', required=True, help='output of appendGT.py')
parser.add_argument('--destination', required=True, help='folder in which graphs will be stored')
parser.add_argument('--det', action='append', help='Use this argument to specify detector(s) to be analyzed')
parser.add_argument('--th', action='append', help='Use this argument to specify threshold of the detector(s) to be analyzed')
parser.add_argument('--plotSimGraphs', help='Use this switch to force a graph file for each simulation')

args = parser.parse_args()

inDir = args.source
graphDir = args.destination
detectorNames = args.det
thresholdNames= args.th
plotSimGraphs = args.plotSimGraphs

if not inDir or not graphDir or not detectorNames or not thresholdNames:
    parser.print_help()
    sys.exit(1)

# note: one agg color per detector..
aggregateColors = itertools.cycle(('b', 'g', 'r', 'c')) # , 'm', 'y', 'k')

simulationList = [f for f in os.listdir(inDir) if os.path.isfile(os.path.join(inDir, f)) and f.endswith('.json')]

# {detector -> {threshold -> {sim -> (giniFP, giniFN)}}}
gini_plot_data = {}

#(attackerType, attackerFraction, runNum, vCount, det, thld, giniFP, giniFN)
plot_data = []

for sim in simulationList:
    tmp = sim.split("_")
    attackerType = None
    attackerFraction = None
    runNumber = None
    runID = None
    vehicleCount = None
    sim_data = None

    print("working on", sim)

    inFileName = os.path.join(inDir, sim)

    # {detector -> {threshold -> {sender -> [(FP,FN,TP,TN)]}}}
    simResultPerDetector = {}
    for x in detectorNames:
        simResultPerDetector[x] = {}

    with open(inFileName, 'r') as inFile:
        first = True
        # map
        for line in inFile:

            if first:
                # header

                obj = json.loads(line)
                sim_data = obj

                attackerType = obj['attackerType']
                attackerFraction = obj['attackerFraction']
                runNumber = obj['runNumber']
                runID = obj['runID']
                vehicleCount = obj['vehicleCount']

                first = False
                continue

            obj = json.loads(line)

            if 'attackertype' in obj:
                print('duplicate header, please fix your files')

            sender = int(obj['senderID'])

            results = obj['results']

            for item in results:
                (det_name, pars, res) = item
                if det_name not in detectorNames:
                    continue

                # find values for thresholdName
                parameter_value = None

                for par in pars:
                    (key, value) = par
                    if key in thresholdNames:
                        parameter_value = value

                if not parameter_value:
                    # this detector does not have the specified parameter, skip it
                    continue

                # set the detection decision
                res_array = None
                if res == "FP":
                    res_array = (1, 0, 0, 0)
                elif res == "FN":
                    res_array = (0, 1, 0, 0)
                elif res == "TP":
                    res_array = (0, 0, 1, 0)
                elif res == "TN":
                    res_array = (0, 0, 0, 1)
                else:
                    print("Warning, unexpected result", item, det_name)

                # det_name -> parameter_value -> sender -> array of results
                simResultPerDetector.setdefault(det_name, {}).setdefault(parameter_value, {}).setdefault(sender, []).append(res_array)

    # {detector -> {threshold -> {sender -> [(FP,FN,TP,TN)]}}}
    sim_gini = {}

    for det_name in list(simResultPerDetector):
        for threshold in sorted(list(simResultPerDetector[det_name])):
            data_dict = simResultPerDetector[det_name][threshold]
            FPR_array = []
            FNR_array = []
            for sender in data_dict:
                (FP, FN, TP, TN) = map(sum, zip(*data_dict[sender]))
                if FP+TN > 0:
                    FPR = FP/(FP+TN)
                    FPR_array.append(FPR)
                elif FN+TP > 0:
                    FNR = FN/(FN+TP)
                    FNR_array.append(FNR)
                else:
                    print("Warning, weird things are happening")

            if len(FPR_array) > 0:
                giniFP = gini(numpy.array(FPR_array))
            else:
                giniFP = -1
            if len(FNR_array) > 0:
                giniFN = gini(numpy.array(FNR_array))
            else:
                giniFN = -1

            gini_plot_data.setdefault(det_name, {}).setdefault(threshold, []).append((giniFP, giniFN, sim_data))
            sim_gini.setdefault(det_name, {})[threshold] = (giniFP, giniFN)
            plot_data.append([attackerType, attackerFraction, runNumber, vehicleCount, det_name, threshold, giniFP, giniFN])

    #plot things

    if plotSimGraphs:
        for det_name in sim_gini:
            print('creating graph for', sim, 'with detector', det_name)

            (_, axes) = plt.subplots(figsize=(10, 8))
            axes.set_xlabel("threshold")
            axes.set_ylabel("giniFP")
            axes.set_xlim([0, len(sim_gini[det_name])])
            axes.set_ylim([0, 1])

            x = sorted(sim_gini[det_name].keys())
            yFP = []
            yFN = []
            for item in x:
                giniFP, giniFN = sim_gini[det_name][item]
                yFP.append(giniFP)
                yFN.append(giniFN)
            axes.bar(range(len(x)), yFP, align='center')
            plt.xticks(range(len(x)), list(x))

            plt.title(sim[:18] + " - " + det_name + " - legit")

            plt.savefig(os.path.join(graphDir, sim + "-" + det_name + "-legitimate.png"), bbox_inces="tight", format='png')
            plt.close()

            (_, axes) = plt.subplots(figsize=(10, 8))
            axes.set_xlabel("threshold")
            axes.set_ylabel("giniFN")
            axes.set_xlim([0, len(sim_gini[det_name])])
            axes.set_ylim([0, 1])

            axes.bar(range(len(x)), yFN, align='center')
            plt.xticks(range(len(x)), list(x))

            plt.title(sim[:18] + " - " + det_name + " - attackers")

            plt.savefig(os.path.join(graphDir, sim + "-" + det_name + "-attackers.png"), bbox_inces="tight", format='png')
            plt.close()


# plot_data contains a list of 8-arrays (attackerType, attackerFraction, runNumber, vehicleCount, detector, threshold, giniFP, giniFN)

# plot key, (attackerType, attackerFraction) -> lines: key, (name) -> data: key, (thld) -> [giniFPArray, giniFNArray]
newData = {}

for (attackerType, attackerFraction, runNumber, vehicleCount, detector, threshold, giniFP, giniFN) in plot_data:
    # aggregate runNumber away, discard vehicleCount
    # graph will contain (giniFP, giniFN) curve for multiple detectors, drawn in order by threshold (i.e., line plot), which is an aggregate
    # other data (attacker type and fraction of attackers) will be shown in each plot title
    linesObject = newData.setdefault((attackerType, attackerFraction), {}) 

    thresholds = linesObject.setdefault(detector, {}) 

    (giniFPArray, giniFNArray) = thresholds.setdefault(threshold, [[], []])

    giniFPArray.append(giniFP)
    giniFNArray.append(giniFN)

for (attackerType, attackerFraction) in newData:

    (fig, axes) = plt.subplots(figsize=(10,5))
    axes.set_xlabel("FPR dispersion (Gini index)")
    axes.set_ylabel("FNR dispersion (Gini index)")
    axes.set_xlim([0,1])
    axes.set_ylim([0,1])

    title = "attacker type " + str(attackerType) + " (" + str(attackerFraction*100) + "% attackers)"
    for name in newData[(attackerType, attackerFraction)]:
        label = name
        XY = []
        for threshold in newData[(attackerType, attackerFraction)][name]:
            (giniFPArray, giniFNArray) = newData[(attackerType, attackerFraction)][name][threshold]
            #note: stdev here shows the variation in the *population of samples*, i.e., is a biased estimator of the standard deviation of the underlying normal distribution, if this is normally distributed
            XY.append([numpy.mean(giniFPArray), numpy.mean(giniFNArray), numpy.std(giniFPArray), numpy.std(giniFNArray)])
        (x, y, xerr, yerr) = zip(*XY)
        color = next(aggregateColors)
        axes.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='--o', label=name, color=color)
        minthld = min(newData[(attackerType, attackerFraction)][name].keys())
        maxthld = max(newData[(attackerType, attackerFraction)][name].keys())
        axes.annotate(minthld, xy=(x[0],y[0]), xytext=(x[0]-0.1, y[0]-0.10), arrowprops=dict(facecolor=color, shrink=0.1))
        axes.annotate(maxthld, xy=(x[-1],y[-1]), xytext=(x[-1]-0.1, y[-1]-0.10), arrowprops=dict(facecolor=color, shrink=0.1))

    # Shrink current axis by 20%
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.9])

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(title)

    plt.savefig(os.path.join(graphDir, "gini-attacker-" + str(attackerType) + "-frac-" + str(attackerFraction) + ".svg"), bbox_inces="tight", format='svg')
    plt.close()
