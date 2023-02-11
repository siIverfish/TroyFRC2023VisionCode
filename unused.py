
"""
cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    #print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

NetworkTables.initialize(server='10.39.52.2')
NetworkTables.addConnectionListener(connectionListener, immediateNotify=True)

#waits for Network Tables
with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()
#get the table
vision_nt = NetworkTables.getTable('Vision')
""" 



"""
vision_nt.putNumber('xError', coord[0] - center_coord[0])
vision_nt.putNumber('yError', center_coord[1] - coord[1])
vision_nt.putNumber('area', M['m00'])
"""