"""Image Fitting Classes."""


import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches

from scipy.optimize import curve_fit as _curve_fit


# NOTE: lnls560-linux was used in benchmarking
#
#   processor	    : 0
#   vendor_id	    : GenuineIntel
#   cpu family	    : 6
#   model		    : 158
#   model name	    : Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz
#   stepping	    : 10
#   microcode	    : 0x96
#   cpu MHz		    : 900.024
#   cache size	    : 12288 KB
#   physical id	    : 0
#   siblings	    : 12
#   core id		    : 0
#   cpu cores	    : 6
#   apicid		    : 0
#   initial apicid	: 0
#   fpu		        : yes
#   fpu_exception	: yes
#   cpuid level	    : 22
#   wp		        : yes


class CurveFitGauss:
    """."""

    SATURATION_8BITS = 2**8-1

    @staticmethod
    def generate_gaussian_1d(
            indcs, sigma=None, mean=0, amplitude=0,
            offset=0, rand_amplitude=0, saturation_threshold=None):
        """Generate a gaussian with given distribution parameters.
        
        Args:
            indcs (int | tuple | list | np.array) : pixel index array def. 
            amp (float) : gaussian intensity amplitude
            offset (float) : gaussian intensity offset
            rand_amp (float) : gaussian point intensity random amplitude
            saturation_threshold (float) : intensity above which image is set
                to saturated
            size (int) : image size.
            sigma (float) : gaussian sigma [pixel]
            mean (float) : gaussian mean [pixel]

        Output:
            data (np.array) : gaussian curve
            indcs (np.array) : pixel indice array
        """
        # benchmark for size=1280
        #   39.8 µs ± 148 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)

        indcs, sigma, mean, amplitude, \
            offset, rand_amplitude, saturation_threshold = \
                CurveFitGauss._process_args(
                    indcs, sigma, mean, amplitude,
                    offset, rand_amplitude, saturation_threshold,
                    )
        data = offset + amplitude * _np.exp(-0.5 * ((indcs - mean)/sigma)**2)
        if rand_amplitude is not None:
            data += (_np.random.rand(*data.shape) - 0.5) * rand_amplitude
        if saturation_threshold is not None:
            data[data > saturation_threshold] = saturation_threshold
        return data, indcs

    @staticmethod
    def generate_gaussian_2d(
            indcs, sigma=None, mean=None,
            amplitude=None, offset=None,
            rand_amplitude=None, saturation_threshold=None,
            angle=0
            ):
        """Generate a bigaussian with given distribution parameters.
        
        Args:
            indcs (tuple(2) | list(2) | np.array(2)) :
                2-component (y and x) pixel index definition. Each component is
                a (int | tuple | list | np.array) each pixel indices.
            sigma (tuple(2) | list(2) | np.array(2)) :
                x and y sigma values (int | float) [pixel]
            mean (tuple(2) | list(2) | np.array(2)) :
                x and y mean values [pixel]
            amplitude (float) : bigaussian intensity amplitude
            offset (float) : bigaussian intensity offset
            rand_amplitude (float) : gaussian point intensity random amplitude
            saturation_threshold (float) :
                intensity above which image is set to saturated

        Output:
            data (np.array) : gaussian curve
            indcsx (np.array) : x pixel indice array (input copy)
            indcsy (np.array) : y pixel indice array (input copy)
        """
        # benchmark for size=(1024, 1280)
        # 35.1 ms ± 833 µs per loop
        #   (mean ± std. dev. of 7 runs, 10 loops each)
        indcsx, indcsy = indcs
        sigmax, sigmay = sigma if sigma is not None else [None] * 2
        meanx, meany = mean if mean is not None else [None] * 2
        
        indcsx, sigmax, meanx, \
            amplitude, offset, rand_amplitude, saturation_threshold = \
                CurveFitGauss._process_args(
                    indcsx, sigmax, meanx,
                    amplitude, offset, rand_amplitude, saturation_threshold)

        indcsy, sigmay, meany, \
            amplitude, offset, rand_amplitude, saturation_threshold = \
                CurveFitGauss._process_args(
                    indcsy, sigmay, meany,
                    amplitude, offset, rand_amplitude, saturation_threshold)

        y = indcsy - meany
        x = indcsx - meanx
        mx, my = _np.meshgrid(x, y)
        mxl = _np.cos(angle) * mx - _np.sin(angle) * my
        myl = _np.sin(angle) * mx + _np.cos(angle) * my
        data = offset + \
            amplitude * _np.exp(-0.5 * ((mxl/sigmax)**2 + (myl/sigmay)**2))
        if rand_amplitude:
            data += (_np.random.rand(*data.shape) - 0.5) * rand_amplitude
        if saturation_threshold is not None:
            data[data > saturation_threshold] = saturation_threshold
        return data, indcsx, indcsy

    @staticmethod
    def normalize(data, maxval):
        """."""
        # benchmark for size=(1024, 1280)
        #   2.91 ms ± 97.8 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)
        # benchmark for size=1280
        #   9.28 µs ± 407 ns per loop
        #   (mean ± std. dev. of 7 runs, 100000 loops each)
    
        new_data = data * (maxval/data.max())
        return new_data

    @staticmethod
    def fit_gaussian(proj, indcs, center, offset):
        """."""
        indc = indcs - center  # centered fitting
        proj = proj.copy() - offset
        sel = proj > 0  # fit only positive data
        vecy, vecx = proj[sel], indc[sel]
        logy = _np.log(vecy)
        pfit = _np.polyfit(vecx, logy, 2)
        if pfit[0] < 0:
            sigma = _np.sqrt(-1/pfit[0]/2) if pfit[0] < 0 else 0.0
            mu = pfit[1] * sigma**2
            amp = _np.exp(pfit[2] + (mu/sigma)**2/2)
            mu += center
        else:
            sigma, mu, amp, offset = [0] * 4
            mu += center
        return sigma, mu, amp, offset

    @staticmethod
    def calc_fit(image, proj, indcs, center):
        """."""
        # get roi gaussian fit
        # proj, indcs, center = \
        #     image.roiy_proj, image.roiy_indcs, image.roiy_center
        param = CurveFitGauss.fit_gaussian(
            proj, indcs, center, image.intensity_min)
        # sigmay, meany, ampy, offsety = paramy
        if param[0] > 0:
            gfit, *_ = CurveFitGauss.generate_gaussian_1d(indcs, *param)
            roi_gaussian_fit = gfit
            error = _np.sum((gfit - proj)**2)
            error /= _np.sum(proj**2)
            roi_gaussian_error = _np.sqrt(error)
        else:
            roi_gaussian_error = float('Inf')
        fit = (param, roi_gaussian_fit, roi_gaussian_error)
        return fit

    @staticmethod
    def _process_args(
        indcs, sigma, mean, amplitude,
        offset, rand_amplitude, saturation_threshold):
        sigma = sigma or float('Inf')
        if isinstance(indcs, (int, float)):
            indcs = _np.arange(int(indcs))
        elif isinstance(indcs, (tuple, list)):
            indcs = _np.array(indcs)
        elif isinstance(indcs, _np.ndarray):
            indcs = _np.array(indcs)
        else:
            raise ValueError('Invalid indcs!')
        if indcs.size < 2:
            raise ValueError('Invalid indcs!')
        elif indcs.size == 2:
            indcs = _np.arange(*indcs)
        amplitude = amplitude or 0
        offset = offset or 0
        rand_amplitude = rand_amplitude or 0
        res = (
            indcs, sigma, mean, amplitude,
            offset, rand_amplitude, saturation_threshold
            )
        return res


class Image1D:
    """1D-Images."""

    SATURATION_8BITS = CurveFitGauss.SATURATION_8BITS

    def __init__(self, data, saturation_threshold=SATURATION_8BITS):
        """."""
        # benchmark for size=1280
        #   6.83 µs ± 71.9 ns per loop
        #   (mean ± std. dev. of 7 runs, 100000 loops each)
        self._data = None
        self._saturation_threshold = saturation_threshold
        self._is_saturated = None
        self._update_image(data)

    @property
    def data(self):
        """Return image data as numpy array."""
        return self._data

    @data.setter
    def data(self, value):
        """Set image."""
        self._update_image(value)

    @property
    def saturation_threshold(self):
        """."""
        return self._saturation_threshold

    @saturation_threshold.setter
    def saturation_threshold(self, value):
        """."""
        self._saturation_threshold = value
        self._update_image(self.data)

    @property
    def shape(self):
        """Return image shape"""
        return self.data.shape

    @property
    def size(self):
        """Return number of pixels."""
        return self.data.size

    @property
    def intensity_min(self):
        """Return image min intensity value."""
        return _np.min(self.data)

    @property
    def intensity_max(self):
        """Return image max intensity value."""
        return _np.max(self.data)

    @property
    def intensity_sum(self):
        """Return image sum intensity value."""
        return _np.sum(self.data)

    @property
    def is_saturated(self):
        """Check if image is saturated."""
        return self._is_saturated

    def imshow(self, fig=None, axis=None, crop=None):
        """."""
        crop = crop or [0, self.data.size]

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        data = self.data[slice(*crop)]
        axis.plot(data)
        axis.set_xlabel('pixel indices')
        axis.set_ylabel('Projection intensity')

        return fig, axis

    def generate_gaussian_1d(self, indcs=None, *args, **kwargs):
        """Generate a gaussian with given distribution parameters."""        
        indcs = indcs or self.size
        return CurveFitGauss.generate_gaussian_1d(
            indcs=indcs, *args, **kwargs)

    def __str__(self):
        """."""
        res = ''
        res += f'size            : {self.size}'
        res += f'\nintensity_min   : {self.intensity_min}'
        res += f'\nintensity_max   : {self.intensity_max}'
        res += f'\nintensity_avg   : {self.intensity_sum/self.size}'
        res += f'\nintensity_sum   : {self.intensity_sum}'
        res += f'\nsaturation_val  : {self.saturation_threshold}'
        res += f'\nsaturated       : {self.is_saturated}'
        return res

    # --- private methods ---
    
    def _update_image(self, data):
        """."""
        # # print('Image1D._update_image')
        self._data = _np.asarray(data)
        if self.saturation_threshold is None:
            self._is_saturated = False
        else:
            self._is_saturated = \
                _np.any(self.data >= self.saturation_threshold)


class Image2D:
    """2D-Images."""

    SATURATION_8BITS = CurveFitGauss.SATURATION_8BITS

    def __init__(self, data, saturation_threshold=SATURATION_8BITS):
        """."""
        # benchmark for sizes=(1024, 1280):
        #   558 µs ± 121 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)
        
        self._data = None
        self._saturation_threshold = saturation_threshold
        self._is_saturated = None
        self._update_image(data)

    @property
    def data(self):
        """Return image data as numpy array."""
        return self._data

    @data.setter
    def data(self, value):
        """Set image."""
        self._update_image(value)

    @property
    def saturation_threshold(self):
        """."""
        return self._saturation_threshold

    @saturation_threshold.setter
    def saturation_threshold(self, value):
        """."""
        self._saturation_threshold = value
        self._update_image(self.data)

    @property
    def shape(self):
        """Return image shape"""
        return self.data.shape

    @property
    def sizey(self):
        """Return image first dimension size."""
        return self.shape[0]

    @property
    def sizex(self):
        """Return image second dimension size."""
        return self.shape[1]

    @property
    def size(self):
        """Return number of pixels."""
        return self.sizey * self.sizex

    @property
    def intensity_min(self):
        """Return image min intensity value."""
        # benchmark for sizes=(1024, 1280):
        #   598 µs ± 90.9 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)
        return _np.min(self.data)

    @property
    def intensity_max(self):
        """Return image max intensity value."""
        return _np.max(self.data)

    @property
    def intensity_sum(self):
        """Return image sum intensity value."""
        return _np.sum(self.data)

    @property
    def is_saturated(self):
        """Check if image is saturated."""
        return self._is_saturated

    def imshow(self, fig=None, axis=None, cropx=None, cropy=None):
        """."""
        cropx, cropy = Image2D.update_roi(self.data, cropx, cropy)

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        data = self.data[slice(*cropy), slice(*cropx)]
        axis.imshow(data)

        return fig, axis

    def generate_gaussian_2d(self, indcsx=None, indcsy=None, *args, **kwargs):
        """Generate a bigaussian with distribution parameters."""    
        indcsy = indcsy or self.sizey
        indcsx = indcsx or self.sizex
        indcs = [indcsx, indcsy]
        return CurveFitGauss.generate_gaussian_2d(indcs, *args, **kwargs)

    def __str__(self):
        """."""
        res = ''
        res += f'sizey           : {self.sizey}'
        res += f'\nsizex           : {self.sizex}'
        res += f'\nintensity_min   : {self.intensity_min}'
        res += f'\nintensity_max   : {self.intensity_max}'
        res += f'\nintensity_avg   : {self.intensity_sum/self.size}'
        res += f'\nintensity_sum   : {self.intensity_sum}'
        res += f'\nsaturation_val  : {self.saturation_threshold}'
        res += f'\nsaturated       : {self.is_saturated}'
        return res

    @staticmethod
    def update_roi(data, roix, roiy):
        roiy = roiy or [0, data.shape[0]]
        roix = roix or [0, data.shape[1]]
        return roix, roiy

    @staticmethod
    def project_image(data, axis):
        axis_ = 1 if axis == 0 else 0
        image = _np.sum(data, axis=axis_)
        return image

    # --- private methods ---
    
    def _update_image(self, data):
        """."""
        # print('Image2D._update_image')
        self._data = _np.asarray(data)
        if self.saturation_threshold is None:
            self._is_saturated = False
        else:
            self._is_saturated = \
                _np.any(self.data >= self.saturation_threshold)


class Image1D_ROI(Image1D):
    """1D-Image ROI."""

    def __init__(self, data, roi=None, *args, **kwargs):
        """."""
        # benchmark for size=1280
        #   28.8 µs ± 194 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)
        self._roi = None
        self._roi_indcs = None
        self._roi_proj = None
        self._roi_center = None
        self._roi_fwhm = None
        super().__init__(data=data, *args, **kwargs)
        self._update_image_roi(roi)
        
    @property
    def roi(self):
        """."""
        return self._roi

    @roi.setter
    def roi(self, value):
        """."""
        self._update_image_roi(value)

    @property
    def roi_indcs(self):
        """Image roi indices."""
        return self._roi_indcs

    @property
    def roi_proj(self):
        """Return image roi projection."""
        return self._roi_proj

    @property
    def roi_center(self):
        """Image roi center position."""
        return self._roi_center

    @property
    def roi_fwhm(self):
        """Image roi fwhm."""
        return self._roi_fwhm

    def imshow(
            self, fig=None, axis=None, crop = None,
            color_ellip=None, color_roi=None):
        """."""
        color_ellip = None if color_ellip == 'no' else color_ellip or 'tab:red'
        color_roi = None if color_roi == 'no' else color_roi or [0.5, 0.5, 0]
        crop = crop or [0, self.data.size]

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        # plot image
        data = Image1D_ROI._trim_image(self.data, crop)
        axis.plot(data)

        if color_ellip:
            centerx = self.roi_center - crop[0]
            # plot center
            axis.axvline(x=centerx, color=color_ellip)
            axis.axvline(x=centerx + self.roi_fwhm/2, ls='--',
                color=color_ellip)
            axis.axvline(x=centerx - self.roi_fwhm/2, ls='--',
                color=color_ellip)

        if color_roi:
            # plot roi
            roi1, roi2 = self.roi
            axis.axvline(x=roi1, color=color_roi)
            axis.axvline(x=roi2, color=color_roi)

        return fig, axis

    def create_trimmed(self):
        """Create a new image trimmed to roi."""
        # benchmark for size=1280:
        #   29.3 µs ± 390 ns per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)
        data = Image1D_ROI._trim_image(self.data, self.roi)
        return Image1D_ROI(data=data)

    def __str__(self):
        """."""
        res = super().__str__()
        res += f'\nroi             : {self.roi}'
        res += f'\nroi_center      : {self.roi_center}'
        res += f'\nroi_fwhm        : {self.roi_fwhm}'

        return res

    def _update_image_roi(self, roi):
        """."""
        # print('Image1D._update_image_roi')
        roi = roi or [0, self._data.size]    
        indcs = Image1D_ROI._calc_indcs(self._data, roi)
        proj = self.data[slice(*roi)]
        hmax = _np.where(proj > (proj.max() - self.data.min())/2)[0]
        fwhm = hmax[-1] - hmax[0]
        center = indcs[0] + _np.argmax(proj)
        
        self._roi, self._roi_indcs, self._roi_proj, \
            self._roi_center, self._roi_fwhm = roi, indcs, proj, center, fwhm

    @staticmethod
    def _calc_indcs(data, roi=None):
        """Return roi indices within image"""
        if roi is None:
            roi = [0, data.size]
        if roi[1] <= data.size:
            return _np.arange(data.size)[slice(*roi)]
        else:
            return None

    @staticmethod
    def _trim_image(image, roi):
        return image[slice(*roi)]


class Image2D_ROI(Image2D):
    """2D-Image ROI."""

    def __init__(self, data, roix=None, roiy=None, *args, **kwargs):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   1.71 ms ± 203 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        self._imagey = None
        self._imagex = None
        super().__init__(data=data, *args, **kwargs)
        self._update_image_roi(roix, roiy)

    @property
    def imagey(self):
        """."""
        return self._imagey

    @property
    def imagex(self):
        """."""
        return self._imagex

    @property
    def roiy(self):
        """."""
        return self.imagey.roi

    @roiy.setter
    def roiy(self, value):
        """."""
        self._update_image_roi(self._roix, value)

    @property
    def roix(self):
        """."""
        return self.imagex.roi

    @roix.setter
    def roix(self, value):
        """."""
        self._update_image_roi(value, self._roiy)

    @property
    def roi(self):
        """."""
        return [self.imagex.roi, self.imagey.roi]

    @roi.setter
    def roi(self, value):
        """."""
        self._update_image_roi(*value)

    def imshow(
            self, fig=None, axis=None,
            cropx = None, cropy = None,
            color_ellip=None, color_roi=None):
        """."""
        color_ellip = None if color_ellip == 'no' else color_ellip or 'tab:red'
        color_roi = None if color_roi == 'no' else color_roi or 'yellow'
        cropx, cropy = Image2D.update_roi(self.data, cropx, cropy)
        x0, y0 = cropx[0], cropy[0]

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        # plot image
        data = Image2D_ROI._trim_image(self.data, cropx, cropy)
        axis.imshow(data, extent=None)

        if color_ellip:
            # plot center
            axis.plot(
                self.imagex.roi_center - x0, self.imagey.roi_center - y0, 'o',
                ms=2, color=color_ellip)

            # plot intersecting ellipse at half maximum
            ellipse = _patches.Ellipse(
                xy=(self.imagex.roi_center - x0, self.imagey.roi_center - y0),
                width=self.imagex.roi_fwhm, height=self.imagey.roi_fwhm,
                angle=0, linewidth=1,
                edgecolor=color_ellip, fill='false', facecolor='none')
            axis.add_patch(ellipse)

        if color_roi:
            # plot roi
            roix1, roix2 = self.roix
            roiy1, roiy2 = self.roiy
            width, height = _np.abs(roix2-roix1), _np.abs(roiy2-roiy1)
            rect = _patches.Rectangle(
                (roix1 - x0, roiy1 - y0),
                width, height, linewidth=1, edgecolor=color_roi,
                fill='False',
                facecolor='none')
            axis.add_patch(rect)

        return fig, axis

    def create_trimmed(self):
        """Create a new image timmed to roi."""
        # benchmark for sizes=(1024, 1280), roi all
        #   187 µs ± 2.56 µs per loop
        #   (mean ± std. dev. of 7 runs, 10000 loops each)
    
        data = Image2D_ROI._trim_image(self.data, self.roix, self.roiy)
        return Image2D_ROI(data=data)

    def __str__(self):
        """."""
        res = super().__str__()
        res += '\n--- projx ---\n'
        res += self.imagex.__str__()
        res += '\n--- projy ---\n'
        res += self.imagey.__str__()
        # res += f'\nroiy            : {self.roiy}'
        # res += f'\nroix            : {self.roix}'
        # res += f'\nroiy_center     : {self.imagey.roi_center}'
        # res += f'\nroix_center     : {self.imagex.roi_center}'
        # res += f'\nroiy_fwhm       : {self.imagey.roi_fwhm}'
        # res += f'\nroix_fwhm       : {self.imagex.roi_fwhm}'

        return res

    def _update_image_roi(self, roix, roiy):
        """."""
        # print('Image2D_ROI._update_image_roi')
        
        roix, roiy = Image2D.update_roi(self.data, roix, roiy)
        
        data = Image2D.project_image(self._data, 0)
        self._imagey = Image1D_ROI(data=data, roi=roiy)
        data = Image2D.project_image(self._data, 1)
        self._imagex = Image1D_ROI(data=data, roi=roix)

    @staticmethod
    def _trim_image(image, roix, roiy):
        return image[slice(*roiy), slice(*roix)]


class Image2D_CMom(Image2D_ROI):
    """Image 2D with normalized central moments."""

    def __init__(self, *args, **kwargs):
        """."""
        self._roix_meshgrid = None
        self._roiy_meshgrid = None
        self._cmomx = None
        self._cmomy = None
        self._cmomyy = None
        self._cmomxy = None
        self._cmomxx = None
        super().__init__(*args, **kwargs)
        self._update_image_roi()

    @property
    def roix_meshgrid(self):
        """."""
        return self._roix_meshgrid

    @property
    def roiy_meshgrid(self):
        """."""
        return self._roiy_meshgrid

    @property
    def roi_cmomy(self):
        """."""
        return self._cmomy

    @property
    def roi_cmomx(self):
        """."""
        return self._cmomx

    @property
    def roi_cmomyy(self):
        """."""
        return _np.sqrt(self._cmomyy)

    @property
    def roi_cmomxy(self):
        """."""
        return _np.arctan(
            self._cmomxy / _np.sqrt(self.roi_sigmax * self.roi_sigmay))

    @property
    def roi_cmomxx(self):
        """."""
        return _np.sqrt(self._cmomxx)

    def calc_central_moment(self, order_x, order_y):
        """."""
        # 9.98 ms ± 19.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
        mgx = self.roix_meshgrid - self._cmomx
        mgy = self.roiy_meshgrid - self._cmomy
        sumpq = _np.sum(mgx**order_x * mgy**order_y * self.roi_data)
        mompq = sumpq / self.roi_data.size
        return mompq

    def imshow(
            self, fig=None, axis=None,
            cropx = None, cropy = None,
            color_ellip=None, color_roi=None):
        """."""
        color_ellip = None if color_ellip == 'no' else color_ellip or 'tab:red'
        color_roi = None if color_roi == 'no' else color_roi or 'yellow'
        cropx, cropy = Image2D.update_roi(self.data, cropx, cropy)
        x0, y0 = cropx[0], cropy[0]

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        # plot image
        data = Image2D_ROI._trim_image(self.data, cropx, cropy)
        axis.imshow(data, extent=None)

        if color_ellip:
            # plot center
            axis.plot(
                self.roi_meanx - x0, self.roi_meany - y0, 'o',
                ms=2, color=color_ellip)

            # plot intersecting ellipse at half maximum
            ellipse = _patches.Ellipse(
                xy=(self.roi_meanx - x0, self.roi_meany - y0),
                width=self.roi_sigmax, height=self.roi_sigmay, angle=0,
                linewidth=1,
                edgecolor=color_ellip, fill='false', facecolor='none')
            axis.add_patch(ellipse)

        if color_roi:
            # plot roi
            roix1, roix2 = self.roix
            roiy1, roiy2 = self.roiy
            width, height = _np.abs(roix2-roix1), _np.abs(roiy2-roiy1)
            rect = _patches.Rectangle(
                (roix1 - x0, roiy1 - y0),
                width, height, linewidth=1, edgecolor=color_roi,
                fill='False',
                facecolor='none')
            axis.add_patch(rect)

        return fig, axis

    def __str__(self):
        """."""
        res = super().__str__()
        res += f'\nroi_cmomx       : {self.roi_meanx}'
        res += f'\nroi_cmomy       : {self.roi_meany}'
        res += f'\nroi_cmomxx      : {self.roi_sigmax}'
        res += f'\nroi_cmomyy      : {self.roi_sigmay}'
        res += f'\nroi_cmomxy      : {self.roi_angle}'
        return res

    def _update_image_roi(self, roix=None, roiy=None):
        """."""
        # 30.8 ms ± 200 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
        super()._update_image_roi(roix=roix, roiy=roiy)

        self._roix_meshgrid, self._roiy_meshgrid = \
            _np.meshgrid(self.roix_indcs, self.roiy_indcs)
        self._cmomx, self._cmomy = self._calc_cmom1()
        self._cmomxx = self.calc_central_moment(2, 0)
        self._cmomxy = self.calc_central_moment(1, 1)
        self._cmomyy = self.calc_central_moment(0, 2)

    def _calc_cmom1(self):
        #17.6 µs ± 108 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        cmom0 = _np.sum(self.roix_proj)  # same as for y
        cmomx = _np.sum(self.roix_proj * self.roix_indcs) / cmom0
        cmomy = _np.sum(self.roiy_proj * self.roiy_indcs) / cmom0
        return cmomx, cmomy


class Image1D_Fit(Image1D_ROI):
    """1D Image Fit."""

    def __init__(self, *args, curve_fit=None, **kwargs):
        """."""
        # benchmark for size=1280
        #   586 µs ± 1.56 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)

        self._roi_mean = None
        self._roi_sigma = None
        self._roi_amp = None
        self._roi_fit = None
        self._roi_fit_error = None
        self._curve_fit = curve_fit or CurveFitGauss
        super().__init__(*args, **kwargs)
        self._update_image_roi(*args, **kwargs)

    @property
    def roi_sigma(self):
        """Image roiy fitted gaussian sigma."""
        return self._roi_sigma

    @property
    def roi_mean(self):
        """Image roiy fitted gaussian mean."""
        return self._roi_mean

    @property
    def roi_amplitude(self):
        """Image roiy fitted gaussian amplitude."""
        return self._roi_amp

    @property
    def roi_fit_error(self):
        """."""
        return self._roi_fit_error

    @property
    def roi_fit(self):
        """."""
        return self._roi_fit, self.roi_indcs

    def set_saturation_flag(self, value):
        """."""
        self._is_saturated = value is True

    def __str__(self):
        """."""
        res = super().__str__()
        res += f'\nroi_amplitude   : {self.roi_amplitude}'
        res += f'\nroi_mean        : {self.roi_mean}'
        res += f'\nroi_sigma       : {self.roi_sigma}'
        res += f'\nroi_fit_err     : {100*self.roi_fit_error} %'

        return res

    def _update_image_roi(self, roi=None, *args, **kwargs):
        """."""
        # print('Image1D_Fit._update_image_roi')
        super()._update_image_roi(roi=roi)

        # fit roi
        param, roi_fit, roi_error = \
            self._curve_fit.calc_fit(
                self, self.roi_proj, self.roi_indcs, self.roi_center)
        self._roi_sigma, self._roi_mean, self._roi_amp, _ = param
        self._roi_fit, self._roi_fit_error = roi_fit, roi_error


class Image2D_Fit(Image2D):
    
    """2D Image Fit."""

    def __init__(self, *args, curve_fit=None, **kwargs):
        """."""
        # benchmark for sizes=(1024, 1280)
        #   2.7 ms ± 23.2 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)

        self._fitx = None
        self._fity = None
        self._angle = 0
        self._curve_fit = curve_fit or CurveFitGauss
        super().__init__(*args, **kwargs)
        self._update_image_fit()

    @property
    def fity(self):
        """."""
        return self._fity

    @property
    def fitx(self):
        """."""
        return self._fitx

    @property
    def roi(self):
        """."""
        return self.fitx.roi, self.fity.roi

    @roi.setter
    def roi(self, value):
        """."""
        self._update_image_fit(*value)

    def plot_projections(
            self, fig=None, axis=None):
        """."""
        if None in (fig, axis):
            fig, axis = _plt.subplots()

        colorx, colory = [0, 0, 0.7], [0.7, 0, 0]

        axis.plot(
            self.fitx.roi_indcs, self.fitx.roi_proj,
            color=colorx, alpha=1.0,
            lw=5, label='roix_proj')
        vecy, vecx = self.fitx.roi_fit
        if vecy is not None:
            axis.plot(
                vecx, vecy, color=[0.5, 0.5, 1], alpha=1.0,
                lw=2, label='roix_fit')

        axis.plot(
            self.fity.roi_indcs, self.fity.roi_proj,
            color=colory, alpha=1.0, lw=5,
            label='roiy_proj')
        vecy, vecx = self.fity.roi_fit
        if vecy is not None:
            axis.plot(
                vecx, vecy, color=[1, 0.5, 0.5], alpha=1.0,
                lw=2, label='roiy_fit')

        axis.legend()
        axis.grid()
        axis.set_ylabel('ROI pixel indices')
        axis.set_ylabel('Projection Intensity')

    def __str__(self):
        """."""
        res = super().__str__()
        res += '\n--- fitx ---\n'
        res += self.fitx.__str__()
        res += f'\nroi_amplitude   : {self.fitx.roi_amplitude}'
        res += f'\nroi_mean        : {self.fitx.roi_mean}'
        res += f'\nroi_sigma       : {self.fitx.roi_sigma}'
        res += f'\nroi_fit_err     : {100*self.fitx.roi_fit_error} %'
        res += '\n--- fity ---\n'
        res += self.fity.__str__()
        res += f'\nroi_amplitude   : {self.fity.roi_amplitude}'
        res += f'\nroi_mean        : {self.fity.roi_mean}'
        res += f'\nroi_sigma       : {self.fity.roi_sigma}'
        res += f'\nroi_fit_err     : {100*self.fity.roi_fit_error} %'

        return res

    def _update_image_fit(self, roix=None, roiy=None):
        """."""
        # print('Image2D_Fit._update_image_fit')

        roix, roiy = Image2D.update_roi(self.data, roix, roiy)
        data = Image2D.project_image(self._data, 0)
        self._fity = Image1D_Fit(data=data,
            roi=roiy, curve_fit=self._curve_fit)
        self._fity.set_saturation_flag(self.is_saturated)
        data = Image2D.project_image(self._data, 1)
        self._fitx = Image1D_Fit(data=data,
            roi=roix, curve_fit=self._curve_fit)
        self._fitx.set_saturation_flag(self.is_saturated)

        
