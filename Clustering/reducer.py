from operator import itemgetter
from defdict import *
import sys

def main(args):
    point2centroid = defaultdict(list)

    for line in sys.stdin:
        line = line.strip()
        # parse the input we got from mapper.py into a dictionary
        centroid, point = line.split('\t')
        point2centroid[centroid].append(point)
        pointX = pointY = 0
        newCentroid = oldCentroid = ''

        for centroid in point2centroid:
            sumX = sumY = count = newX = newY = 0
            oldCentroid += centroid
            oldCentroid += ''

            for point in point2centroid[centroid]:
                pointX, pointY = point.split(',')
                sumX += int(pointX)
                sumY += int(pointY)
                count += 1

        newX = sumX / count
        newY = sumY / count
        newCentroid += 'newX' + ',' + 'newY'
        newCentroid += ''
        print(newCentroid)

if __name__ == "__main__": main(sys.argv)