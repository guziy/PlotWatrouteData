from data.basin import Basin
__author__="huziy"
__date__ ="$Jul 23, 2011 12:43:38 PM$"



class Cell():
    """
    Class representing a grid cell
    """
    def __init__(self, id = None, ix = -1, jy = -1):
        self.id = id
        self.next = None
        self.previous = []
        self.area = -1
        self.clay = -1
        self.sand = -1
        self.number_of_upstream_cells = -1  #number of cells that flow into this cell
        self.drainage = -1 #number of cells that flow into this cell
        self.drainage_area = -1 #drainage area in km**2
        self.ibn = 3
        self.chslp = -1
        self.x = ix
        self.y = jy
        self.basin = None
        self.is_end_cell = False
        self.direction_value = -1
        self.basins = []
        self.topo = None
        #rid  - is the indication river or lake, 0 - river
        self.rid = 0
        self.rout = 0
        self.channel_length = -1
        self.polygon = None
        self.next_id = -1
        self.lon = -1
        self.lat = -1


    def set_next(self, next_cell):
        """
        set the next cell for the current cell
        """
        if self == next_cell:
            self.next = None
            print 'endorheic'
            return

        #if changing the next cell
        #if self.next is not None and self in self.next.previous:
        #    self.next.previous.remove(self)

        self.next = next_cell
        if next_cell is not None:
            next_cell.add_previous(self)

        #sanity checks
        if self.next is not None:
            assert self in self.next.previous
        assert self not in self.previous


    def get_upstream_cells(self):
        """
        returns a list of all upstream cells
        """
        result = []
        for prev in self.previous:
            result.extend(prev.get_upstream_cells())
            result.append(prev)
        return result

    def coords(self):
        return self.x, self.y


    def get_cells_upflow(self, basin):
        result = []
        for prev in self.previous:
            if prev in basin.cells:
                result.extend(prev.get_cells_upflow(basin))
                result.append(prev)
        return result

    def set_coords(self, i, j):
        self.x = i
        self.y = j

    def add_previous(self, prev):
        if prev == self:
            print 'endorheic'

        if prev not in self.previous:
            self.previous.append(prev)
        if len(self.previous) > 8:
            assert(len(self.previous) <= 8)
        

    def is_connected_to(self, other_cell):
        """
        returns True if the cell is connected to  the other cell, False otherwize
        """
        current = self
        i = 0
        path = []
        while current is not None:

            if other_cell in path:
                print 'Here'
                return True

            #closed loop
            if current in path:
                return False
            path.append(current)


            if current == other_cell:
                return True


            if current.basin != other_cell.basin or current.basin is None:
                return False

            if current.next is not None:
                if current.basin != current.next.basin:
                    return False

            current = current.next

            i += 1
            assert i < 10000
        return False


    def calculate_drainage_area(self):
        """
        calculate drainage area for the cell
        """
        if self.drainage_area >= 0:
            return

        self.drainage_area = self.area
        for prev in self.previous:
            if prev.basin is None:
                continue
            prev.calculate_drainage_area()
            self.drainage_area += prev.drainage_area
#            self.drainage_area += prev.area

    def calculate_number_of_upstream_cells(self):
        """
        returns the number of upstream cells which
        flow into the current cell
        """
        if self.number_of_upstream_cells >= 0:
            return self.number_of_upstream_cells
        else:
            result = 0
            for the_previous in self.previous:
                result += the_previous.calculate_number_of_upstream_cells()
            result += len(self.previous)
        self.number_of_upstream_cells = result
        return result
        pass



    def set_common_basin(self, basin):
        """
        set the same basin for the current and all upstream cells
        """
        # @type theCell Cell
        for prev in self.previous:
            prev.set_common_basin(basin)

        # @type basin Basin
        basin.add_cell(self)




if __name__ == "__main__":
    print "Hello World"
