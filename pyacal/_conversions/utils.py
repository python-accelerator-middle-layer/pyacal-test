"""."""

ConverterTypes = {
    'ScaleConverter': {'scale', },
    'OffsetConverter': {'offset', },
    'LookupTableConverter': {'table_name', },
    'PolynomConverter': {'coeffs', 'limits', 'is_forward'},
    'CompanionProptyConverter': {'devname', 'propty', 'operation'},
    'MagRigidityConverter': {'devname', 'propty', 'conv_2_ev'},
}


class ConverterNames:
    ScaleConverter = 'ScaleConverter'
    OffsetConverter = 'OffsetConverter'
    LookupTableConverter = 'LookupTableConverter'
    PolynomConverter = 'PolynomConverter'
    CompanionProptyConverter = 'CompanionProptyConverter'
    MagRigidityConverter = 'MagRigidityConverter'
