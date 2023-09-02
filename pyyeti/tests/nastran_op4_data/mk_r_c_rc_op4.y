clear
R = 25
C = 31
N = 33

rmat = zeros(R,C)
pv = floor(rand(N,1)*length(rmat))
rmat(pv) = 1000*randn(length(pv),1)

cmat = zeros(R,C)
pv = floor(rand(N,1)*length(cmat))
cmat(pv) = 1000j*randn(length(pv),1)

rcmat = rmat + cmat
db('save','r_c_rc.op4',{'rmat','cmat','rcmat'})
db('save','r_c_rc.mat',{'rmat','cmat','rcmat'})
