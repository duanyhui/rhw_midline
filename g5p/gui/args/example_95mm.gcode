%
(95mm Square with intermediate keypoints)
(Origin: lower-left corner; Units: mm; Absolute coords)

G90 G21 G17 G94        (绝对, 毫米, XY平面, 进给单位/分钟)
G54
G40 G49 G80

G00 Z5.000             (安全高度)
G00 X0.000 Y0.000      (到起点)

;(M3 S1000)            (如为切削/激光按需启用)
;(M8)

G01 Z0.000 F300.0      (落刀/落笔到Z=0；切削请改为所需切深)
F800.0                 (走刀进给)

(--- Bottom edge: (0,0) -> (95,0) ---)
G01 X23.750 Y0.000     (KP B25)
G01 X47.500 Y0.000     (KP B50)
G01 X71.250 Y0.000     (KP B75)
G01 X95.000 Y0.000     (Corner BR)

(--- Right edge: (95,0) -> (95,95) ---)
G01 X95.000 Y23.750    (KP R25)
G01 X95.000 Y47.500    (KP R50)
G01 X95.000 Y71.250    (KP R75)
G01 X95.000 Y95.000    (Corner TR)

(--- Top edge: (95,95) -> (0,95) ---)
G01 X71.250 Y95.000    (KP T75)
G01 X47.500 Y95.000    (KP T50)
G01 X23.750 Y95.000    (KP T25)
G01 X0.000  Y95.000    (Corner TL)

(--- Left edge: (0,95) -> (0,0) ---)
G01 X0.000  Y71.250    (KP L75)
G01 X0.000  Y47.500    (KP L50)
G01 X0.000  Y23.750    (KP L25)
G01 X0.000  Y0.000     (Back to origin)

G00 Z5.000
;(M5)
;(M9)
G00 X0.000 Y0.000
M30
%