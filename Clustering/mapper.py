import sys, math, random

class Point:
    # self.coords is a list of coordinates for this Point
    # self.n is the number of dimensions this Point lives in (ie, its space)
    # self.reference is an object bound to this Point
    # Initialise new Points
    def __init__(self, coords, reference=None):
        self.coords = coords
        self.n = len(coords)
        self.reference = reference
    # Return a string representation of this Point
    def __repr__(self):
        return str(self.coords)

def main(args):
    # variable to hold map output
    outputmap = ''
    # centroid details are stored in an input file
    cfile = "centroidinput.txt"
    infilecen = open(cfile, "r")
    centroid = infilecen.readline()
    # print centroid
    for point in sys.stdin:
        # remove leading and trailing whitespace
        point = point.strip()
        # split the line into words
        points = point.split()
        # remove leading and trailing whitespace
        centroid = centroid.strip()
        # split the centroid into centroids
        centroids = centroid.split()

        for value in points:
            dist = 0
            minDist = 999999
            bestCent = 0

            for c in centroids:
                # split each co-ordinate of the centroid and the point
                cSplit = c.split(',')
                vSplit = value.split(',')
                # To handle non-numeric value in centroid or input points
                try:
                    dist = (((int(cSplit[0]) - int(vSplit[0]))**2) +
                            ((int(cSplit[1]) - int(vSplit[1]))**2))**.5
                    if dist < minDist:
                        minDist = dist
                        bestCent = c
                except ValueError:
                    pass
                print('%s\t%s' % (bestCent, value))

def makeCentroids():
    cfile = open("centroidinput.txt", "r")
    seed = cfile.readline()
    cfile.close()
    FILE1 = open("centroidpoints.txt", "w")
    FILE1.writelines(seed)
    FILE1.close()
    return seed

if __name__ == "__main__": main(sys.argv)