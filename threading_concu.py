import threading

global_var = 0


def modify(value):
    global global_var
    while True:
        global_var = value
        assert global_var == value  # this fails if the other thread has
        # changed global_var in the meantime


t1 = threading.Thread(target=modify, args=(1,))
t2 = threading.Thread(target=modify, args=(2,))


t1.start()
t2.start()
