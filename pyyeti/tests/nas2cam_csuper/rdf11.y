clear
uses op2
f11 = 'drm2.f11'
drms = op2.procdrm12('drm2.f11')
% drms = op2.rdpostop2_drms(f11,0)
db('save', 'drm2.mat', 'drms')
