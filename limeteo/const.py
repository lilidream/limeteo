"""
地球物理常数，使用CGCS2000模型

[1]魏子卿.2000中国大地坐标系[J].大地测量与地球动力学,2008,28(06):1-5.
"""

PI = 3.14159265358979

EARTH_RADIUS: int = 6378137
"""
CGCS2000 地球长轴半径(m)
"""

LATITUDE_DISTANCE: float = EARTH_RADIUS * PI / 180
"""
CGCS2000 球模型，纬度1度对应的距离(m)
"""

EARTH_SHORT_RADIUS: float = 6356752.3141
"""
CGCS2000 地球短轴半径(m)
"""

EARTH_FLATTENING: float = 3.35281068118e-3
"""
CGCS2000 地球扁率
"""

G_ACC: float = 9.7976432224
"""
CGCS2000 重力加速度(m/s^2)
"""


SIDEREAL_DAY: int = 86164
"""
恒星日(s)
"""

EARTH_OMEGA: float = 7.292115e-5
"""
地球自转角速度(1/s)
"""
