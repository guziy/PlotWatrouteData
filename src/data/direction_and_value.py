
__author__="huziy"
__date__ ="$Jul 31, 2011 11:59:38 PM$"


iShifts = [1,  1,  0, -1, -1, -1,  0, 1]
jShifts = [0, -1, -1, -1,  0,  1,  1, 1]
values =  [1,  2,  4,  8, 16, 32, 64, 128]


##Converts direction to value and vice versa

def to_value(i, j, iNext, jNext):
    di = iNext - i
    dj = jNext - j

    if iNext < 0 or jNext < 0:
        return -1

    for v, iShift, jShift in zip(values, iShifts, jShifts):
        if di == iShift and dj == jShift:
            return v
    return -1


def to_indices(i, j, value):
    if value < 0: return -1, -1
    for v, iShift, jShift in zip(values, iShifts, jShifts):
        if value == v:
            return i + iShift, j + jShift
    return -1, -1



if __name__ == "__main__":
    print "Hello World"
