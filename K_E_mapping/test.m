clc
clear
close all

l=4;
l1=8;
M=load('M.dat');%質量矩陣
%X0=load('result_si_un_0504.dat');%資料輸入
%X1=load('result_si_B_0504.dat');%資料輸入
%K=X0(3:10,:);%標準構架之勁度矩陣
%Kd=X1(3:10,:);%損傷構架之勁度矩陣
K=load('simulation_K.dat');
Kd=load('simulation_K_1F_3F_80%E.dat');
%{
k0=diag(Kd);
k1=diag(Kd,1);
k2=diag(Kd,2);
Kd=diag(k0)+diag(k1,1)+diag(k1,-1)+diag(k2,2)+diag(k2,-2);
%}
[nr,nc]=size(M);
[Vd,Dd]=eig(Kd,M);
[V,D]=eig(K,M);
m=diag(M)';
wd=diag(Dd)';
w=diag(D)';
f=sqrt(w)/2/pi;
fd=sqrt(wd)/2/pi;
ms=flipud(V);
msd=flipud(Vd);

for j=1:nc
    for i=1:nc
        if i==1
            k(j,i)=w(j)*sum(m(i)*ms(i:end,j))/ms(i,j);
            kd(j,i)=wd(j)*sum(m(i)*msd(i:end,j))/msd(i,j);
        else
            k(j,i)=w(j)*sum(m(i)*ms(i:end,j))/(ms(i,j)-ms(i-1,j));
            kd(j,i)=wd(j)*sum(m(i)*msd(i:end,j))/(msd(i,j)-msd(i-1,j));
            
        end
    end
end
%k為標準構架之剪力構架型式之勁度值
%kd為損傷構架之剪力構架型式之勁度值

kv=abs(k);
kw=abs(kd);
kx=mean(kv(1:8,:));
ky=mean(kw(1:8,:));
%kz=ky./kx;
kz=kx./ky;
%{
SDI=1-(kd./k);
SDIm=[mean(SDI(2:3,:));mean(SDI(2:l1,:))];
SDIt=[SDI;SDIm];
SDIx=SDIt';
%}


