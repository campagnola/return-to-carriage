"""Physical units for the world's spatial and photometric scales.

The game does not simulate light transport; it just expresses light in units you
can *estimate* -- lux for illuminance (arriving light), lumens for a source's
luminous flux, candela-per-square-metre for the luminance a surface reflects
back at the eye. That only works if there is one honest length scale to hang the
inverse-square law on, and that is :data:`CELL_SIZE_M`.

Everything physical derives from this constant rather than restating it: the
point-light fall-off (see :class:`~.light.PointLight`) turns a lumen rating into
a lux field using distances measured in metres, and any speed a caller wants to
think of in metres-per-second is ``m_per_s / CELL_SIZE_M`` cells per second. The
in-game speeds (player walk/run, a fireball's flight) are still written directly
in cells/second, which with a 1 m cell read straight across as metres/second.
"""

# units used just to make the code more readable
_units = ['m', 's', 'g', 'lx', 'lm', 'cd', 'K']
_unit_vals = {'g': 0.001}

for unit in _units:
    val = _unit_vals.get(unit, 1.0)
    globals()[unit] = val
    for i, prefixes in enumerate(('p', 'n', 'uμ', 'm', '', 'k', 'M', 'G', 'T')):
        for prefix in prefixes:
            globals()[prefix + unit] = val * 10 ** (3 * (i - 4))

#: Physical size of one maze cell, in metres. The single source of truth for the
#: world's spatial scale. Cell spacing is isotropic -- the same in x and y --
#: even though glyphs are rendered as rectangles; a cell is a square metre of
#: floor, and the rectangular look is a display choice, not a world fact.
CELL_SIZE_M = 1.0 * m


