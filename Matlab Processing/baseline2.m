function y=baseline2(x,v,tidel,nsamp)

% ----  cubic splines baseline wander removing  ----
%
% function y=baseline2(x,v,tidel,nsamp)
%
% Input parameters:
%   x: ecg signal vector
%   v: nots times vector
%   tidel: reference time of the ecg vector begining
%   nsamp: number of samples to estimate the nots
%
% Output parameter:
%   y:  ECG baseline corrected
%
% Copyright (c), Jose Garcia Moros, Zaragoza University, Spain
% email: jogarmo@posta.unizar.es
% last revision: 15th May 1997
% modified by Raquel Bailon nov 2001

if nargin<4
    nsamp=5;
end
  
[n m]=size(x);   % n=number of samples   m = number of leads
y=zeros(n,m);
basl=(1:n)';
v=v-tidel; 
if v(1)<=0 v(1)=[]; end 

for i=1:m,
 z=[];
 for j=1:nsamp
     z=[z x(v+j-1,i)];
 end
 mx=mean(z,2);
 basline=spline(v,mx,basl);  
 y(:,i)=x(:,i);
 y(v(1):v(end),i)=x(v(1):v(end),i)-basline(v(1):v(end));
end

