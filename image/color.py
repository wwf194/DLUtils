def Str2Color(Str):
    return (0.0, 0.0, 0.0)
        
def ColorFloat2Int8(ColorFloat):
    ColorInt8 = []
    for Channel in ColorFloat:
        ColorInt8.append(round(255.0 * ColorFloat))
    return tuple(ColorInt8)

FloatColor2Int8 = ColorFloat2Int8
    