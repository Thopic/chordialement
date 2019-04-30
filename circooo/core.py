import matplotlib.pyplot as plt

def circo(source, target, data=None):
    """ Draw a circo
    @ Arguments:
    - source/target: two array of the same size that contains 
      every link shown in the circo. Alternatively if data is
      not None, two strings that correspond to column names 
      in the dataframe data.
    """
