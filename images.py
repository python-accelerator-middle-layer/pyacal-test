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


class Image2D:
    """2D-Images."""

    def __init__(self, data, saturation_intensity=2**8-1):
        """."""
        # benchmark:
        #   470 µs ± 2.08 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)
        
        self._data = None
        self._saturation_intensity = saturation_intensity
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
    def saturation_intensity(self):
        """."""
        return self._saturation_intensity

    @saturation_intensity.setter
    def saturation_intensity(self, value):
        """."""
        self._saturation_intensity = value
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
        # benchmark @ lnls560-linux, 1024 x 1280 images:
        # 638 µs ± 2.76 µs per loop
        # (mean ± std. dev. of 7 runs, 1000 loops each)
        return _np.min(self.data)

    @property
    def intensity_max(self):
        """Return image max intensity value."""
        # benchmark @ lnls560-linux, 1024 x 1280 images:
        # 635 µs ± 1.35 µs per loop
        # (mean ± std. dev. of 7 runs, 1000 loops each)
        return _np.max(self.data)

    @property
    def intensity_sum(self):
        """Return image sum intensity value."""
        # benchmark @ lnls560-linux, 1024 x 1280 images:
        # 809 µs ± 5.17 µs per loop
        # (mean ± std. dev. of 7 runs, 1000 loops each)
        return _np.sum(self.data)

    @property
    def is_saturated(self):
        """Check if image is saturated."""
        return self._is_saturated

    def imshow(self, fig=None, axis=None, cropx=None, cropy=None):
        """."""
        cropy = cropy or [0, self.data.shape[0]]
        cropx = cropx or [0, self.data.shape[1]]

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        data = self.data[slice(*cropy), slice(*cropx)]
        axis.imshow(data)

        return fig, axis

    @staticmethod
    def generate_gaussian_2d(
            amp, angle, offset, rand_amp, saturation_intensity,
            sizex, sigmax, meanx,
            sizey, sigmay, meany):
        """Generate a bigaussian with distribution parameters.
        
        Args:
            amp (float) : bigaussian intensity amplitude
            angle (float) : bigaussian XY tilt angle [rad]
            offset (float) : bigaussian intensity offset
            rand_amp (float) : bigaussian point intensity random amplitude
            saturation_intensity (float) : intensity above which image is set
                to saturated
            sizex (int) : horizontal image size.
            sigmax (float) : horizontal gaussian sigma [pixel]
            meanx (float) : horizontal gaussian mean [pixel]
            sizey (int) : vertical image size.
            sigmay (float) : vertical gaussian sigma [pixel]
            meany (float) : vertical gaussian mean [pixel]
        """
        # benchmark:
        #   43.4 ms ± 182 µs per loop
        #   (mean ± std. dev. of 7 runs, 10 loops each)
        
        indcsy = _np.arange(0, sizey)
        indcsx = _np.arange(0, sizex)
        y = indcsy - meany
        x = indcsx - meanx
        mx, my = _np.meshgrid(x, y)
        mxl = _np.cos(angle) * mx - _np.sin(angle) * my
        myl = _np.sin(angle) * mx + _np.cos(angle) * my
        data = offset + \
            amp * _np.exp(-0.5 * ((mxl/sigmax)**2 + (myl/sigmay)**2))
        if rand_amp is not None:
            data += (_np.random.rand(*data.shape) - 0.5) * rand_amp
        if saturation_intensity is not None:
            data[data > saturation_intensity] = amp
        return data

    def __str__(self):
        """."""
        res = ''
        res += f'sizey           : {self.sizey}'
        res += f'\nsizex           : {self.sizex}'
        res += f'\nintensity_min   : {self.intensity_min}'
        res += f'\nintensity_max   : {self.intensity_max}'
        res += f'\nintensity_avg   : {self.intensity_sum/self.size}'
        res += f'\nintensity_sum   : {self.intensity_sum}'
        res += f'\nsaturated       : {self.is_saturated}'
        return res

    # --- private methods ---
    
    def _update_image(self, data):
        """."""
        self._data = _np.asarray(data)
        if self.saturation_intensity is None:
            self._is_saturated = False
        else:
            self._is_saturated = \
                _np.any(self.data >= self.saturation_intensity)


class Image2D_ROI(Image2D):
    """2D-Image ROI."""

    def __init__(self, data, roix=None, roiy=None, *args, **kwargs):
        """."""
        # benchmark:
        #   3.17 ms ± 475 µs per loop
        #   (mean ± std. dev. of 7 runs, 100 loops each)

        self._roiy = None
        self._roix = None
        self._roiy_indcs = None
        self._roix_indcs = None
        self._roiy_proj = None
        self._roix_proj = None
        self._roiy_center = None
        self._roix_center = None
        self._roiy_fwhm = None
        self._roix_fwhm = None
        super().__init__(data=data, *args, **kwargs)
        self._update_image_roi(roix, roiy)

    @property
    def roi_data(self):
        """Image data in roi."""
        return self._roi_data

    @property
    def roiy(self):
        """."""
        return self._roiy

    @roiy.setter
    def roiy(self, value):
        """."""
        self._update_image_roi(self._roix, value)

    @property
    def roix(self):
        """."""
        return self._roix

    @roix.setter
    def roix(self, value):
        """."""
        self._update_image_roi(value, self._roiy)

    @property
    def roi(self):
        """Image roix and roiy."""
        return [self.roix, self.roiy]

    @roi.setter
    def roi(self, value):
        """Set image roix and roiy."""
        self._update_image_roi(value[0], value[1])

    @property
    def roiy_indcs(self):
        """Image roi indices along first dimension."""
        return self._roiy_indcs

    @property
    def roix_indcs(self):
        """Image roi indices along second dimension."""
        return self._roix_indcs

    @property
    def roiy_proj(self):
        """Return image roi projection along first dimension."""
        return self._roiy_proj

    @property
    def roix_proj(self):
        """Return image roi projection along second dimension."""
        return self._roix_proj

    @property
    def roiy_center(self):
        """Image roi center position along first dimension [pixel]."""
        return self._roiy_center

    @property
    def roix_center(self):
        """Image roi center position along second dimension [pixel]."""
        return self._roix_center

    @property
    def roiy_fwhm(self):
        """Image roi fwhm along first dimension [pixel]."""
        return self._roiy_fwhm

    @property
    def roix_fwhm(self):
        """Image roi fwhm along second dimension [pixel]."""
        return self._roix_fwhm

    def imshow(
            self, fig=None, axis=None,
            cropx = None, cropy = None,
            color_ellip=None, color_roi=None):
        """."""
        color_ellip = None if color_ellip == 'no' else color_ellip or 'tab:red'
        color_roi = None if color_roi == 'no' else color_roi or 'yellow'
        cropy = cropy or [0, self.data.shape[0]]
        cropx = cropx or [0, self.data.shape[1]]
        x0, y0 = cropx[0], cropy[0]

        if None in (fig, axis):
            fig, axis = _plt.subplots()

        # plot image
        data = Image2D_ROI._trim_image(self.data, cropx, cropy)
        axis.imshow(data, extent=None)

        if color_ellip:
            # plot center
            axis.plot(
                self.roix_center - x0, self.roiy_center - y0, 'o',
                ms=2, color=color_ellip)

            # plot intersecting ellipse at half maximum
            ellipse = _patches.Ellipse(
                xy=(self.roix_center - x0, self.roiy_center - y0),
                width=self.roix_fwhm, height=self.roiy_fwhm, angle=0,
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

    def create_trimmed(self):
        """Create a new image timmed to roi."""
        # benchmark:
        #   1.56 ms ± 19.5 µs per loop
        #   (mean ± std. dev. of 7 runs, 1000 loops each)
        data = Image2D_ROI._trim_image(self.data, self.roix, self.roiy)
        return Image2D_ROI(data=data)

    def __str__(self):
        """."""
        res = super().__str__()
        res += f'\nroiy            : {self.roiy}'
        res += f'\nroix            : {self.roix}'
        res += f'\nroiy_center     : {self.roiy_center}'
        res += f'\nroix_center     : {self.roix_center}'
        res += f'\nroiy_fwhm       : {self.roiy_fwhm}'
        res += f'\nroix_fwhm       : {self.roix_fwhm}'

        return res

    @staticmethod
    def calc_indcs(data, axis, roi=None):
        """Return roi indices within image"""
        if roi is None:
            roi = [0, data.shape[axis]]
        if roi[1] <= data.shape[axis]:
            return _np.arange(data.shape[axis])[slice(*roi)]
        else:
            return None

    def _update_image_roi(self, roix, roiy):
        """."""
        def calc_params(axis, roi):
            roi = roi or [0, self._data.shape[axis]]    
            indcs = Image2D_ROI.calc_indcs(self._data, axis, roi)
            axis_ = 1 if axis == 0 else 0
            proj = _np.sum(self.data, axis=axis_)[slice(*roi)]
            hmax = _np.where(proj > (proj.max() - self.data.min())/2)[0]
            fwhm = hmax[-1] - hmax[0]
            center = indcs[0] + _np.argmax(proj)
            return roi, indcs, proj, center, fwhm

        # vertical
        axis, roi = 0, roiy
        self._roiy, self._roiy_indcs, self._roiy_proj, \
            self._roiy_center, self._roiy_fwhm = calc_params(axis, roi)

        # horizontal
        axis, roi = 1, roix
        self._roix, self._roix_indcs, self._roix_proj, \
            self._roix_center, self._roix_fwhm = calc_params(axis, roi)

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
        cropy = cropy or [0, self.data.shape[0]]
        cropx = cropx or [0, self.data.shape[1]]
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


class CurveFitGauss:
    """."""

    @staticmethod
    def generate_gaussian_1d(indcs, sigma, mean, amp, offset):
        """."""
        data_fit = offset + amp * _np.exp(-0.5 * ((indcs - mean)/sigma)**2)
        return data_fit, indcs

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


class Image2D_Fit(Image2D_ROI):
    """2D Image Fit."""

    def __init__(self, curve_fit=None, *args, **kwargs):
        """."""
        self._roiy_mean = None
        self._roix_mean = None
        self._roiy_sigma = None
        self._roix_sigma = None
        self._roiy_amp = None
        self._roix_amp = None
        self._roiy_fit = None
        self._roix_fit = None
        self._roiy_fit_error = None
        self._roix_fit_error = None
        self._curve_fit = curve_fit or CurveFitGauss
        super().__init__(*args, **kwargs)
        self._update_image_roi()

    @property
    def roiy_sigma(self):
        """Image roiy fitted gaussian sigma."""
        return self._roiy_sigma

    @property
    def roix_sigma(self):
        """Image roix fitted gaussian sigma."""
        return self._roix_sigma

    @property
    def roiy_mean(self):
        """Image roiy fitted gaussian mean."""
        return self._roiy_mean

    @property
    def roix_mean(self):
        """Image roix fitted gaussian mean."""
        return self._roix_mean

    @property
    def roix_fit_mean(self):
        """Image roix fitted gaussian mean."""
        return self._roix_mean

    @property
    def roiy_amplitude(self):
        """Image roiy fitted gaussian amplitude."""
        return self._roiy_amp

    @property
    def roix_amplitude(self):
        """Image roix fitted gaussian amplitude."""
        return self._roix_amp

    @property
    def roiy_fit_error(self):
        """."""
        return self._roiy_fit_error

    @property
    def roix_fit_error(self):
        """."""
        return self._roix_fit_error

    @property
    def roiy_fit(self):
        """."""
        return self._roiy_fit, self.roiy_indcs

    @property
    def roix_fit(self):
        """."""
        return self._roix_fit, self.roix_indcs

    def plot_projections(
            self, fig=None, axis=None):
        """."""
        if None in (fig, axis):
            fig, axis = _plt.subplots()

        colorx, colory = [0, 0, 0.7], [0.7, 0, 0]

        axis.plot(
            self.roix_indcs, self.roix_proj, color=colorx, alpha=1.0,
            lw=5, label='roix_proj')
        vecy, vecx = self.roix_fit
        if vecy is not None:
            axis.plot(
                vecx, vecy, color=[0.5, 0.5, 1], alpha=1.0,
                lw=2, label='roix_fit')

        axis.plot(
            self.roiy_indcs, self.roiy_proj, color=colory, alpha=1.0, lw=5,
            label='roiy_proj')
        vecy, vecx = self.roiy_fit
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
        res += f'\nroiy_sigma      : {self.roiy_sigma}'
        res += f'\nroix_sigma      : {self.roix_sigma}'
        res += f'\nroiy_mean       : {self.roiy_mean}'
        res += f'\nroix_mean       : {self.roix_mean}'
        res += f'\nroiy_amplitude  : {self.roiy_amplitude}'
        res += f'\nroix_amplitude  : {self.roix_amplitude}'
        res += f'\nroiy_fit_err    : {100*self.roiy_fit_error} %'
        res += f'\nroix_fit_err    : {100*self.roix_fit_error} %'

        return res

    def _update_image_roi(self, roix=None, roiy=None):
        """."""
        super()._update_image_roi(roix=roix, roiy=roiy)

        # fit roiy
        param, roi_fit, roi_error = \
            self._curve_fit.calc_fit(
                self, self.roiy_proj, self.roiy_indcs, self.roiy_center)
        self._roiy_sigma, self._roiy_mean, self._roiy_amp, _ = param
        self._roiy_fit, self._roiy_fit_error = roi_fit, roi_error

        # fit roix
        param, roi_fit, roi_error = \
            self._curve_fit.calc_fit(
                self, self.roix_proj, self.roix_indcs, self.roix_center)
        self._roix_sigma, self._roix_mean, self._roix_amp, _ = param
        self._roix_fit, self._roix_fit_error = roi_fit, roi_error
