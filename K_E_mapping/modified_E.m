K=load('simulation_K.dat');
K_mod=load('K1_modified(80%)_aver_8modes.dat');
M=load('M.dat');
[V,D]=eig(K,M);
[V_mod,D_mod]=eig(K_mod,M);
m=diag(M)';
w_mod=diag(D_mod)';
w=diag(D)';
f=sqrt(w)/2/pi;
f_mod=sqrt(w_mod)/2/pi;
MAC=MAC_plot(V,V_mod);
f_error=abs(f_mod-f)*100./f
mac=diag(MAC)

