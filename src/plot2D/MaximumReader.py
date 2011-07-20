import os.path
__author__="huziy"
__date__ ="$Mar 14, 2011 5:15:32 PM$"

import os

import application_properties
application_properties.set_current_directory()

class Point():
    def __init__(self, i = 0, j = 0):
        self.data = []
        self.i = i
        self.j = j
    pass


class VincentMaximumsReader():
    def __init__(self, data_path = 'data/streamflows/Vincent_annual_max/aex.txt'):
        self.points = []
        self._points_map = {}
        self._parse_file(os.path.join(data_path))
        pass

    def _parse_file(self, file_path = ''):
        f = open(file_path)
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0:
                continue
            if line.strip() == '':
                continue

            fields = line.split()
            data = map(float, fields[3:])
            p = Point()
            p.data = data
            p.i = int(fields[1])
            p.j = int(fields[2])
            self._points_map[(p.i, p.j)] = data
            self.points.append(p)
        f.close()





    def get_data_at(self, i, j):
        return self._points_map[(i, j)]

def test():
    v = VincentMaximumsReader()
    print v.points[0].data
if __name__ == "__main__":
    test()
    print "Hello World"
