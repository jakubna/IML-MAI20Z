def label_encoding(value, classes: bytearray):
    """Function written based on label encoding rule
        Parameters
        ----------
        value : value to interpret
        classes: array of classes
        Returns
        -------
        self : returns index of x value in classes array.
        """
    return classes.index(value)