__author__="huziy"
__date__ ="$Aug 10, 2011 3:49:16 PM$"

from shapely.geometry import Point, Polygon
import numpy as np

class AffineTransform():

    def __init__(self):
        self.transMatrix = None
        self.shift = None
        pass

    def define_from_points(self, sourceToDest = {}):
        if len(sourceToDest) != 3:
            raise ValueError('only 2d transform is implemented')

        rhs = np.matrix(6 * [0.0]).transpose()
        mat = np.matrix(np.zeros((6,6)))

        sourcePoints = sourceToDest.keys()
        for i in xrange(0, 6, 2):
            source = sourcePoints[i / 2]
            dest = sourceToDest[source]
            # @type dest Point
            rhs[i] = dest.x
            rhs[i + 1] = dest.y

            for j in xrange(2):
                mat[i + j, 0 + 3 * j] = source.x
                mat[i + j, 1 + 3 * j] = source.y
                mat[i + j, 2 + 3 * j] = 1.0

        v = mat.getI() * rhs

        self.transMatrix = np.matrix(np.zeros((2,2)))
        self.transMatrix[0,0] = v[0,0]
        self.transMatrix[0,1] = v[1,0]
        self.transMatrix[1,0] = v[3,0]
        self.transMatrix[1,1] = v[4,0]
        self.shift = np.matrix([v[2,0], v[5,0]]).transpose()


    def transform(self, geometry):
        if type(geometry) == type(Point()):
            v = np.matrix([geometry.x, geometry.y]).transpose()
            res = self.transMatrix * v + self.shift
            return Point(res[0,0], res[1,0])
        if type(geometry) == type(tuple([])):
            return self.transform(Point(geometry))
        elif type(geometry) == type(Polygon()):
            # @type geometry Polygon
            ps = geometry.exterior.coords
            newPs = []
            for p in ps:
                thePoint = self.transform(p)
                newPs.extend(thePoint.coords)
            return Polygon(newPs)
        else:
            raise ValueError('Unknown geometry type: {0}'.format(geometry) )




def test_translation():
    aft = AffineTransform()

    points = {}
    p1 = Point(0,0)
    p2 = Point(1,1)
    p3 = Point(0,1)

    points[p1] = Point(1,1)
    points[p2] = Point(2,2)
    points[p3] = Point(1,2)


    aft.define_from_points(points)
    p1 = aft.transform(Point(10,10))
    assert  Point(11.0,11.0).wkb == p1.wkb


def test_rotation():
    aft = AffineTransform()

    points = {}
    p1 = Point(0,0)
    p2 = Point(1,1)
    p3 = Point(0,1)

    points[p1] = Point(0,0)
    points[p2] = Point(-1,-1)
    points[p3] = Point(0,-1)


    aft.define_from_points(points)
    assert aft.transform(Point(10,10)).wkb == Point(-10, -10).wkb


def test_polygon():
    aft = AffineTransform()

    points = {}
    p1 = Point(0,0)
    p2 = Point(1,1)
    p3 = Point(0,1)

    points[p1] = Point(0 + 10,0 + 15)
    points[p2] = Point(-1 + 10,-1 + 15)
    points[p3] = Point(0 + 10,-1 + 15)

    
    pts = []
    pts.extend(p1.coords)
    pts.extend(p2.coords)
    pts.extend(p3.coords)
    pts.extend(p1.coords)
    polygon = Polygon([(-1, -1), (1,-1), (1,1), (-1,-1)])

    aft.define_from_points(points)

    print polygon.wkt

    p1 = aft.transform(polygon)
    print p1.wkt

def test():
    test_translation()
    test_rotation()
    test_polygon()



if __name__ == "__main__":
    test()
    print "Hello World"
