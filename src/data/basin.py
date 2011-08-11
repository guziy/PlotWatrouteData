__author__="huziy"
__date__ ="$Jul 23, 2011 3:42:54 PM$"


class Basin():
    '''
    Class representing a basin
    '''
    def __init__(self, id = -1, name = "unknown"):
        self.id = id
        self.name = name
        self.cells = []
        self.exit_cells = []
        self.description = ''

    def add_cell(self, the_cell):
        self.cells.append(the_cell)
        the_cell.basins.append(self)
        the_cell.basin = self


    def _drainage_for_cell(self, the_cell):
        if the_cell.drainage >= 0:
            return
        drainage = 0
        for prev in the_cell.previous:
            if prev not in self.cells:
                continue
            self._drainage_for_cell(prev)
            drainage += prev.drainage + 1
        the_cell.drainage = drainage


    def _calculate_internal_drainage(self):
        #initialize drainages
        for the_cell in self.cells:
            the_cell.drainage = -1
        for the_cell in self.cells:
            self._drainage_for_cell(the_cell)
        pass


    def _try_to_determine_exits(self):
        self._calculate_internal_drainage()
        drainage = -1
        result_cell = None
        for the_cell in self.cells:
            if drainage < the_cell.drainage and the_cell.next not in self.cells:
                result_cell = the_cell
                drainage = the_cell.drainage
        result_cell.next = None
        self.exit_cells.append(result_cell)

    def set_exit_cells(self, i_list = None, j_list = None):
        '''
        Sets exit cells for the basin using cell indices
        '''
        if i_list == None or j_list == None:
            self._try_to_determine_exits()
        else:
            for i, j in zip(i_list,  j_list):
                print i, j
                assert cells[i][j] in self.cells, "outlet not in mask of the basin %s" % self.name

                self.exit_cells.append(cells[i][j])
                cells[i][j].basin = self

                #if the outflow points to the cell that does not belong to any basin
                if cells[i][j].next != None and cells[i][j].next.basin == None:
                    cells[i][j].set_next( None )

                #if the outflow points to the cell in the same basin
                if cells[i][j].next != None and cells[i][j].next.basin == cells[i][j].basin:
                    cells[i][j].set_next(None)





    def get_max_i(self):
        '''
            returns maximum horizontal cell index in the basin
        '''
        i_max = -1
        for the_cell in self.cells:
            if i_max < the_cell.x:
                i_max = the_cell.x
        return i_max


    def get_min_i(self):
        '''
            returns minimum horizontal cell index in the basin
        '''
        i_min = -1
        for the_cell in self.cells:
            if i_min == -1:
                i_min = the_cell.x

            if i_min > the_cell.x:
                i_min = the_cell.x
        return i_min


    def get_max_j(self):
        '''
            returns maximum vertical cell index in the basin
        '''
        j_max = -1
        for the_cell in self.cells:
            if j_max < the_cell.y:
                j_max = the_cell.y
        return j_max

    def get_min_j(self):
        '''
            returns minimum vertical cell index in the basin
        '''
        j_min = -1
        for the_cell in self.cells:
            if j_min == -1:
                j_min = the_cell.y

            if j_min > the_cell.y:
                j_min = the_cell.y
        return j_min


    def get_approxim_middle_indices(self):
        _i_list = []
        _j_list = []
        for cell in self.cells:
            _i_list.append(cell.x)
            _j_list.append(cell.y)
        return np.mean(_i_list), np.mean(_j_list)



if __name__ == "__main__":
    print "Hello World"
