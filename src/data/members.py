__author__="huziy"
__date__ ="$20 dec. 2010 12:03:42$"


from datetime import timedelta

current_ids = [ 'aet', 'aev', 'aey', 'aez', 'afa']
future_ids =  [ 'aeu', 'aew','afc', 'afd' , 'afb']

current2future = dict(zip(current_ids, future_ids))

h3 = timedelta(hours = 3)
h6 = timedelta(hours = 6)
id_to_step = {
    'aet' : h6,
    'aev' : h3,
    'aex' : h6,
    'aey' : h6,
    'aez' : h6,
    'afa' : h6
}


control_id = 'aex'

all_current = []
all_current.extend(current_ids)
#all_current.append(control_id)


all_future = []
all_future.extend(future_ids)


test_current = ['aey']
test_future = ['afc']

all_members = []
all_members.extend(current_ids)
all_members.extend(future_ids)



#current_ids = all_current

#current_ids = test_current
#future_ids = test_future


if __name__ == "__main__":
    print "Hello World"
