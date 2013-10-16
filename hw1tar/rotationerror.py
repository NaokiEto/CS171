class RotateError(Exception):
   def __init__(self):
      print "For rotation, x, y, and z cannot all be 0!"
