import numpy as _np

class LifetimeMeasurement:

    def __init__(self, capacity):
        self._I = _np.array([])
        self._t = _np.array([])
        self._l = _np.array([])
        self._capacity = capacity

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        if value < 0 or value == self._capacity: return
        self._capacity = value
        if (value < len(self._t)):
            self._t = self._t[value+1:]
            self._I = self._I[value+1:]
            self._l = self._l[value+1:]

    @property
    def size(self):
        return len(self._t)

    @property
    def last_data(self):
        t = self._t[-1]
        I = self._I[-1]
        l = self._l[-1]
        return I,t,l

    def add_measurement(self,I,t):
        if self.size < self._capacity:
            if self.size == 0:
                self._l = _np.array([0])
            else:
                self._l = _np.append(self._l, self._l[-1])
            self._t = _np.append(self._t, t)
            self._I = _np.append(self._I, I)
        else:
            self._t = _np.roll(self._t, -1); self._t[-1] = t
            self._I = _np.roll(self._I, -1); self._I[-1] = I
            self._l = _np.roll(self._l, -1); self._l[-1] = self._l[-2]

    def calc_lifetime(self):
        #return self._analysis1()
        return self._analysis1()

    def _analysis2(self):

        min_size = 100
        if self.size < min_size: return None, None
        t = self._t - self._t[0]
        I = self._I
        nrpts = min_size
        tau, A, B = self._fit1(I[-nrpts:],t[-nrpts:])
        residue = (A + B * t - I) / 0.010
        idx, *_ = _np.where(abs(residue)>5)
        if len(idx):
            nrpts = idx[-1]
            tau, A, B = self._fit1(I[-nrpts:],t[-nrpts:])
        else:
            tau, A, B = self._fit1(I,t)
        return tau, A, B

    def _analysis1(self):
        if self.size < 50: return None, None
        t = self._t - self._t[0]
        I = self._I
        tau, A, B = self._fit1(I,t)
        if tau is not None:
            self._l[-1] = tau
        return tau, A, B

    def _fit1(self, I, t):
        m11 = t.size
        m12 = _np.sum(t)
        m22 = _np.dot(t,t)
        b1  = _np.sum(I)
        b2  = _np.dot(t,I)
        detM = m11 * m22 - m12 * m12
        A = (m22 * b1 - m12 * b2) / detM
        B = (-m12 * b1 + m11 * b2) / detM
        tau = -A / B
        return tau, A, B

    def _fit2(self, I, t):
        m11 = _np.cumsum(_np.ones(I.size))
        m12 = _np.cumsum(t)
        m22 = _np.cumsum(t*t)
        b1  = _np.cumsum(I)
        b2  = _np.cumsum(t*I)
        detM = m11 * m22 - m12 * m12
        A = ( m22 * b1 - m12 * b2) / detM
        B = (-m12 * b1 + m11 * b2) / detM
        _np.outer(B,t)
        residue = A + B * t - I
        tau = -A / B
        return tau, A, B
