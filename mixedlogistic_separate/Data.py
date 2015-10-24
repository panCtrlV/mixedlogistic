class Data(object):
    def __init__(self, Xm, Xr, y, m):
        self.Xm = Xm
        self.Xr = Xr
        self.y = y
        self.m = m
        self.n = len(y)

    def _formatArray(self, x):
        if self.n < 5:
            return '\n'.join('{}:\t{}'.format(*example) for example in enumerate(x))
        else:
            s1 = '\n'.join('{}:\t{}'.format(*example) for example in enumerate(x[:3]))
            s2 = '\n'.join('{}:\t{}'.format(i+self.n-3, val) for (i, val) in enumerate(x[self.n-3:]))
            return '\n'.join([s1, '...', s2])

    def __str__(self):
        line1 = "Hidden layer covariates:"
        line2 = self._formatArray(self.Xm) if self.Xm is not None else 'None'

        line3 = "Observed layer covariates:"
        line4 = self._formatArray(self.Xr) if self.Xr is not None else 'None'

        line5 = "Response with {} categories:".format(self.m+1)
        line6 = self._formatArray(self.y)

        return '\n'.join([line1, line2, line3, line4, line5, line6])