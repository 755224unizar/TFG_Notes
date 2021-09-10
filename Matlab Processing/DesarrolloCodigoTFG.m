%% Pruebas Representacion TFG
file = 'Training_WFDB_CPS\A0010.mat';
header = 'Training_WFDB_CPS\A0010.hea';
load(file);
hea = readheader(header);
s = 0:hea.nsamp-1;
val = 1000/hea.gain(1).*val;

figure, 
subplot (311),plot(s/hea.freq, val(1,:)), xlabel 'Tiempo (s)', ylabel 'uV', title 'Derivación I';
hold on, plot(s/hea.freq, x(1,:)), legend(['Original'], ['Sin línea de base']);
grid on;
subplot (312),plot(s/hea.freq, val(6,:)), xlabel 'Tiempo (s)', ylabel 'uV', title 'Derivación aVF';
hold on, plot(s/hea.freq, x(6,:)), legend(['Original'], ['Sin línea de base']);
grid on;
subplot (313),plot(s/hea.freq, val(9,:)), xlabel 'Tiempo (s)', ylabel 'uV', title 'Derivación V3';
hold on, plot(s/hea.freq, x(9,:)), legend(['Original'], ['Sin línea de base']);
grid on;
sgtitle 'Fichero A0010.mat sin línea de base';

%% Procesado de los ficheros
% Leemos el fichero header para extraer todas las caracteristicas de la
% senial
file = 'Training_WFDB_CPS\A0001.mat';
header = 'Training_WFDB_CPS\A0001.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)

figure, plot(x(lead,:)), title (['Lead: ', num2str(lead)]), xlabel 'Time(samples)', ylabel 'uV';

% Pasamos a probar el delineador
sigdir = 'Training_WFDB_CPS\';
headir = 'Training_WFDB_CPS\';
matdir = 'Training_WFDB_CPS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    s = 0:hea.nsamp;
    x_lead = x(lead,:);
    position = [position, p];
end
lead = 2;
s = 0:hea.nsamp;
x_lead = x(lead,:);
p = position(lead);
figure, plot(s/hea.freq,x_lead), title (['Derivación: ',hea.desc(lead,:)]), xlabel 'Tiempo (s)', ylabel 'uV';
hold on, plot(p.Pon/hea.freq, x_lead(p.Pon), 'ro');          % Marcamos inicio onda P
hold on, plot(p.Poff/hea.freq,x_lead(p.Poff), 'rx');   % Marcamos final onda P
hold on, plot(p.QRSon/hea.freq, x_lead(p.QRSon), 'mo');          % Marcamos inicio QRS
hold on, plot(p.QRSoff/hea.freq, x_lead(p.QRSoff), 'mx');          % Marcamos final QRS
hold on, plot(p.Ton/hea.freq, x_lead(p.Ton), 'ko');          % Marcamos  inicio onda T
hold on, plot(p.Toff/hea.freq, x_lead(p.Toff), 'kx');          % Marcamos final onda T
legend(['ECG'],['Inicio Onda P'],['Final Onda P'],['Inicio QRS'],['Final QRS'],['Inicio Onda T'],['Final Onda T']);

%% Comprobacion de las derivaciones
% Recorro los dataset comprobando que la info esta en el mismo formato
set = ['Q' 'I' 'S'];        
header1 = 'A0001.hea';
error = 0;
hea1 = readheader(header1);
for i = 1:length(set)
    hea2 = readheader([set(i),'0001.hea']);
    % Comprobamos que los nombres de las derivaciones están escritos igual y en el mismo orden
    error = error+sum(sum(hea1.desc ~= hea2.desc));  % Calculo el numero total de errores
end

% Extraccion de datos del paciente de los comentarios del header
comments = comments_header(header1);    % Por alguna razon no lee bien los comentarios del final del header
% Solucionado posteriormente con la programación de la función
% my_comment_reader()
%% Formacion de la matriz de derivaciones alineadas

file = 'A0001.mat';
header = 'A0001.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
%[x] = load(file).val;
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)
fclose(fid);
figure, plot(x(lead,:)), title (['Lead: ', hea.desc(lead,:)]), xlabel 'Time(samples)', ylabel 'uV';

% Pasamos a probar el delineador
sigdir = 'Training_WFDB_CPS\';headir = 'Training_WFDB_CPS\';matdir = 'Training_WFDB_CPS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    s = 0:hea.nsamp;
    x_lead = x(lead,:);
    position = [position, p];
end

matriz = [];
N = 500;   % Muestras a coger a partir del evento
figure,
for lead = 1:hea.nsig
    x_lead = x(lead,:);
    row = x_lead(position(lead).QRSon(1):position(lead).QRSon(1)+N-1);
    matriz =[matriz; row];
    subplot (12,1,lead),plot(row);
end
% En teoria matriz deberia ser una matriz cuyas filas con las derivaciones,
% todas ellas comenzando el inicio del primer QRS

%% Remuestreo de las señales
% Algunas señales estan muestreadas a 257 Hz, 500 Hz o 1000 Hz. Pretendemos
% trabajar siempre con señales de 500 Hz
s_mat = 'I0005.mat';
s_hea = 'I0005.hea';
file = ['Training_WFDB_StPetersburg\',s_mat];
header = ['Training_WFDB_StPetersburg\',s_hea];
%file = 'Training_WFDB_PTB\S0005.mat';
%header = 'Training_WFDB_PTB\S0005.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
%[x] = load(file).val;
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
fclose(fid);
sig = x(:,2:end);
sig=1000/hea.gain(lead) * sig; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)
%figure, plot(x(lead,:)), title (['Lead: ', num2str(lead)]), xlabel 'Time(samples)', ylabel 'uV';
sigdir = 'Training_WFDB_StPetersburg\';headir = 'Training_WFDB_StPetersburg\';matdir = 'Training_WFDB_StPetersburg\';
figure, plot(sig(1,:)), title 'Señal original sin Remuestrear';
if hea.freq~=500
    disp('Remuestreando señal');
    sig2 = resample(sig', 500, hea.freq)'; % Remuestreamos para tenerla a 500 Hz
    sig2 = int16(sig2);
    datos = [x(:,1) sig2];
    fid2 = fopen(['Ficheros_Fs_Cambiada\',hea.recname,'.mat'], 'w+');
    fwrite(fid2, datos,'int16');
    fclose(fid2);
    %figure, plot(x(1,:)), title 'Señal original Remuestreada';
    %save(['Ficheros_Fs_Cambiada\',hea.recname,'.mat'], 'x');
    hea.freq = 500;

    fid = fopen(['Ficheros_Fs_Cambiada\',hea.recname,'.hea'], 'w');
    for i=0:hea.nsig
        if i==0
            fprintf(fid, [hea.recname,' ',num2str(hea.nsig), ' ',num2str(hea.freq),' ', num2str(size(sig2,2)), ' ', num2str(hea.btime(1:end)), '\n']);
        else
            fprintf(fid,[hea.fname(i,:), ' ', num2str(hea.fmt(i)),'+',num2str(hea.offset(i)),' ', num2str(hea.gain(i)),'/',hea.units(i,:),' ', num2str(hea.adcres(i)),' ', num2str(hea.adczero(i)),' ', num2str(hea.initval(i)),' ',num2str(hea.cksum(i)),' ', num2str(hea.bsize(i)), hea.desc(i,:),'\n' ]);
        end
    end
    fclose(fid);
    headir = 'Ficheros_Fs_Cambiada\';
    sigdir = 'Ficheros_Fs_Cambiada\';
    matdir = 'Ficheros_Fs_Cambiada\';
    
end

position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    disp(['Delineando derivacion ', num2str(lead)]);
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    if length(fieldnames(p))~=19
       p = empty_struct; 
    end
    position = [position,p];
end
% Al cambiar el numero de muestras en el .hea (de unas 460000 a 900002)
% wavedet falla diciendo: Index in position 2 exceeds array bounds. Error in wavedet (line 231) -> sig = sig(:,leadinfile);
x_lead = sig2(1,:);
figure, plot(x_lead);
hold on;
plot(position(1).qrs, x_lead(position(1).qrs), 'ro');
%% Leer Comentarios
fid =fopen('PruebaComments.hea');
comments = strings;
l = fgetl(fid);
cont = 1;
while l~=-1
   if l(1) == '#'
    if cont<=3
        s = split(l,' ');
        comments(cont) = s(end);
        cont = cont+1;
    end
   end
   l = fgetl(fid);
end
fclose(fid);
% Encapsulando el codigo en una funcion
comments = my_comment_reader('PruebaComments.hea');

%% Formacion de matriz imagen por derivacion (24/03/2021)
% Pretendemos generar una imagen que contenga toda la información de una
% derivacion, alineando cada fila de la imagen y centrandola en cada
% latido.
% Para poder coger ventanas temporales del mismo tamaño se excluiran
% aquellos primeros o ultimos latidos para los que esto no sea posible
file = 'A0001.mat';
header = 'A0001.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)

% Pasamos a probar el delineador
sigdir = 'Training_WFDB_CPS\';headir = 'Training_WFDB_CPS\';matdir = 'Training_WFDB_CPS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    s = 0:hea.nsamp;
    x_lead = x(lead,:);
    position = [position, p];
end

x_lead = x(lead,:);
matriz = [];
N_antes = hea.freq*0.5;   % Muestras a coger antes del evento
N_despues = hea.freq;     % Muestras a coger despues del evento
for i = 2:length(position(lead).QRSmainpos)-1
    
    row = x_lead(position(lead).qrs(i)-N_antes:position(lead).qrs(i)+N_despues-1);
    
    
    matriz =[matriz; row];
    
end
figure, imagesc(matriz), title('Matriz Latidos Excluyendo primer y ultimo latido');
figure,
for i = 1:size(matriz,1)
    
    subplot (24,1,i),plot(matriz(i,:));
    if(i==1)
        title('Right bundle branch block (59118001)');
    end
end
%% Otra opcion (Padding de zeros a los lados para poder coger todos los latidos)
% Si no se quieren perder los primeros y/o ultimos latidos se puede
% realizar un padding a la señal que permita tomar ventanas temporales
% siempre del mismo tamaño
lead = 1;
x_lead = x(lead,:);
matriz = [];
N_antes = hea.freq*0.5;   % Muestras a coger antes del evento
N_despues = hea.freq; 
x_lead = [zeros(1,N_antes), x_lead, zeros(1,N_despues)];
for i = 1:length(position(lead).qrs)
    
    row = x_lead(position(lead).qrs(i)-N_antes+250:position(lead).qrs(i)+N_despues-1+500);
    matriz =[matriz; row];
end
figure, imagesc(matriz), title('Matriz Latidos Completa Con Padding');
figure,
for i = 1:size(matriz,1)
    
    subplot (24,1,i),plot(matriz(i,:));
    if(i==1)
        title('Right bundle branch block (59118001)');
    end
end

datos_paciente = my_comment_reader('A0001.hea');


%% Formacion de señal escalon de latido para una sola derivacion
% Realmente esta no se utilizara como input de la red sino que usaremos la
% global de las 12 derivaciones
lead = 1;
x_lead = x(lead,:);
escalon_QRS = zeros(1, length(x_lead));
for i = 1:length(position(lead).QRSon)
    
    escalon_QRS(position(lead).QRSon(i):position(lead).QRSoff(i)) = 1;
end
figure,plot(escalon_QRS);

matriz = [];
N_antes = 500*0.5;   % Muestras a coger antes del evento
N_despues = 500;     % Muestras a coger despues del evento

for i = 2:length(position(lead).qrs)-1
    
    row = escalon_QRS(position(lead).QRSon(i)-N_antes:position(lead).QRSon(i)+N_despues-1);
    matriz =[matriz; row];
end
figure, imagesc(matriz), title('Matriz de Escalones QRS');
figure,
for i = 1:size(matriz,1)
    subplot (size(matriz,1),1,i),plot(matriz(i,:),'r');
    if(i==1)
        title('Right bundle branch block (59118001)');
    end
end
%% Comprobar latidos en todas las derivaciones para formar el eje temporal comun (29/03/2021)
file = 'A0255.mat';
header = 'A0255.hea';
lead = 1;
hea = readheader([sigdir,header]);   % -> Struct con todos los campos de informacion
fid=fopen([sigdir,file]);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)

% Pasamos a probar el delineador
sigdir = 'Training_WFDB_CPS\';headir = 'Training_WFDB_CPS\';matdir = 'Training_WFDB_CPS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    s = 0:hea.nsamp;
    x_lead = x(lead,:);
    position = [position, p];
end
% Primero intentaré representar todos los latidos en cada derivacion para
% ver si a simple vista puede apreciarse alguna detección de QRS que sea
% debida al ruido y no a un latido real

eje = 1:length(x(1,:));
figure,
for i = 1:12
    hold on,
    plot(position(i).qrs, ones(1,length(position(i).qrs))+i-1, 'k.');
    
end
title(['Latidos detectados por cada derivación ',file]), ylim([-1 13]);
% En el caso de la muestra A0001.mat todos los latidos se detectan en todas
% las derivaciones por lo que ninguna deteccion es erronea o debida al
% ruido
%% Calculo de indices globales [NO UTILIZADO]
% A continuacion, desarrollo un codigo para obtener el eje de indices
% global para todas las derivaciones
% De los latidos detectados en mas de 9 derivaciones con una distancia
% máxima de 100 ms se realizara la mediana de su QRSon y de su QRSoff,
% obteniendo el inicio y final de ese "QRS global"

N_mismo_latido = 500*100*1e-3;
ind_QRS = [];
% Para esta primera version voy a considerar que todos los latidos son
% siempre detectados por la primera derivacion
for i = 1:length(position(1).qrs)
    ind_actual=position(1).qrs(i);
    ind_QRS(i,1) = ind_actual;
    for j = 2:12
        for k=1:length(position(j).qrs)
            if(abs(ind_actual- position(j).qrs(k))<N_mismo_latido)
                ind_QRS(i,j) =  position(j).qrs(k);
            end
        end
    end
    
end
ind_QRS

% La matriz ind_QRS ahora contiene por filas los indices de cada latido
% valido en cada derivacion. Tras esto para obtener la localización global
% de cada QRS podriamos realizar la mediana de cada fila
ind_Global = [];
ind_QRS(ind_QRS==0) = NaN;
for i=1:size(ind_QRS, 1)
    row_ind = ind_QRS(i,:);
    if(length(row_ind(~isnan(row_ind)))>9)
        ind_Global(i) = median(row_ind,2, 'omitnan')';
    end
end
ind_Global = round(ind_Global);
x_lead = x(1,:);
figure, plot(x_lead, 'b');
hold on, plot(ind_Global, x_lead(ind_Global), 'r*');
title('Latidos globales (qrs global)'), title('Indices de latidos globales sobre la primera derivacion');


%% Implementacion de deteccion de latidos con tren de deltas filtrado [VERSION FINAL]

v = hanning(N_mismo_latido);
deltas = zeros(12,size(x,2));
convs = zeros(12, size(x,2)+length(v)-1);
for i = 1:12
    
    deltas(i,position(i).qrs)=1;
    convs(i,:) = conv(deltas(i,:),v);
    
end
plot(convs(1,:));
latidos = sum(convs,1);
hold on,plot(latidos);
[picos, ~] = findpeaks(latidos);
umbral = 0.6*max(picos);
[pks,gl_ind] = findpeaks(latidos,'MinPeakHeight',umbral); 
gl_ind = gl_ind-round(N_mismo_latido/2); 

%% Probamos baseline2 para eliminar la linea de base
file = 'A0001.mat';
header = 'A0001.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)

% Pasamos a probar el delineador
sigdir = 'Training_WFDB_CPS\';headir = 'Training_WFDB_CPS\';matdir = 'Training_WFDB_CPS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    s = 0:hea.nsamp;
    
    position = [position, p];
end
x_lead = x(10,:);
not_samples = hea.freq*80e-3;
%x_limpia = baseline2(x_lead', position(10).qrs, 0, not_samples)';
x_limpia = baseline2(x_lead', round(position(10).qrs - 0.08 *hea.freq), 0, not_samples)';

figure, subplot(121),plot(x(10,:)), title 'Señal Origninal';
subplot(122),
plot(x_limpia), title 'Señal limpia';
% no parece limpiar bien, ademas modifica las amplitudes generales de la
% señal y provoca distorsión en aplitud al principio y al final de la señal
% Esto era debido al mal uso de la funcion (ya arreglado)

%% Probamos delineador Multi-Lead (DESACARTADO)

file = 'A0001.mat';
header = 'A0001.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)


sigdir = 'Training_WFDB_CPS';headir = 'Training_WFDB_CPS';matdir = 'Training_WFDB_CPS';
[position,positionaux,messages] = wavedet_3D(sigdir, headir,matdir, hea.recname, 0, 'wav', 1:3, [1 Inf],0);

% Me da un error de identificador de fichero invalido


%% Algoritmo para formar el tren de pulsos rectangulares que marquen los QRS


% Tengo que identificar de todos los QRSon, el índice menor. Si en las
% siguientes 500*12e-3 muestras están las demas detecciones, tomare ese
% índice menor como el inicio del QRS. Si no hay al menos 3 detecciones en
% los siguientes 500*12e-3, descarto la muestra y busco la siguiente menor.

% Otra opcion para no tener que alinear los latidos "validos" es usar un
% anotador de QRS externo al llamar a wavedet, forzando así a que saque el
% mismo número de latidos en todas las derivaciones

% Uso de wavedet con anotador de QRS externo. Como obtengo el fichero de
% anotaciones QRS externo??


file = 'Training_WFDB_CPS\A0001.mat';
header = 'Training_WFDB_CPS\A0001.hea';
lead = 1;
hea = readheader(header);   % -> Struct con todos los campos de informacion
fid=fopen(file);
x=fread(fid,[12 Inf],'int16');
x=1000/hea.gain(lead) * x; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)

% Pasamos a probar el delineador
sigdir = 'Training_WFDB_CPS\';headir = 'Training_WFDB_CPS\';matdir = 'Training_WFDB_CPS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    position = [position, p];
end
N_mismo_latido = 500*100*1e-3;
v = hanning(N_mismo_latido);
deltas = zeros(12,size(x,2));
convs = zeros(12, size(x,2)+length(v)-1);
for i = 1:12
    
    deltas(i,position(i).qrs)=1;
    convs(i,:) = conv(deltas(i,:),v);
    
end
%plot(convs(1,:));
latidos = sum(convs,1);
%plot(latidos);
[pks,gl_ind] = findpeaks(latidos,'MinPeakHeight',8);
save('AnotacionesQRS\anot.mat','gl_ind');       % Guardo fichero de anotaciones externas
dirann = 'AnotacionesQRS\';
position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],1,1,'anot',dirann);
    position = [position, p];
end
% Ahora nos hemos asegurado que el delineador nos devuelva el mismo numero
% de ondas, ya que el anotador QRS es externo y cada derivacion tendra el
% mismo numero de latidos y por tanto el mismo numero de QRS y ondas P y T
% Tenemos que buscar los inicios y fin de escalones en QRSon y QRSoff,
% evitando las muestras que tengan valor NaN

% Formacion señal escalon de QRS
margen = hea.freq*20e-3;                % Con un margen de 12 ms ninguna deteccion cumplia las normas
conf = 0;
escalon_QRS = zeros(1,length(x(1,:)));
ini= 1;
vec_ini = zeros(1, length(position(1).QRSon));
fin = 1;
vec_fin = zeros(1, length(position(1).QRSon));
no_ini = 1;
no_fin = 1;
for i=1:length(position(1).QRSon)
   disp(['Analizando latido ', num2str(i)]);
   ini_candidates = [];
   fin_candidates = [];
   for k=1:12
     ini_candidates = [ini_candidates, position(k).QRSon(i)];
     fin_candidates = [fin_candidates, position(k).QRSoff(i)];
   end
   
   if(sum(isnan(ini_candidates))<4 && sum(isnan(fin_candidates))<4)   % Si mas de 4 indices son NaN no proceso el latido
       while no_ini         % Estaremos en el while hasta que encontremos un ini confirmado
           conf=0;
           [ini, ind_ini] = min(ini_candidates);     % Inicialmente supongo que la primera deteccion es el inicio
           for j = 1:12
                % Si despues de la supuesta primera deteccion vienen al menos 3
                % mas, esa deteccion de inicio sera correcta
               if abs(ini-position(j).QRSon(i))<=margen                                           
                   conf = conf+1;
               end
           end
           if conf>=3       % Confirmamos que ini es un valor real de inicio
               no_ini=0;
               disp('Inicio confirmado');
           else             % Si este no era un inicio real lo eliminamos de los posibles ini
                ini_candidates(ind_ini) = [];
           end
       end
       while no_fin         % Estaremos en el while hasta que encontremos un fin confirmado
           conf=0;
           [fin, ind_fin] = max(fin_candidates);     % Inicialmente supongo que la ultima deteccion es el final
           for j = 1:12
                % Si antes de la supuesta ultima deteccion vienen al menos 3
                % mas, esa deteccion de fin sera correcta
               if abs(fin-position(j).QRSoff(i))<=margen                                           
                   conf = conf+1;
               end
           end
           if conf>=3       % Confirmamos que fin es un valor real de final
               no_fin=0;
               disp('Final confirmado');
           else             % Si este no era un final real lo eliminamos de los posibles fin
               fin_candidates(ind_fin) = [];
           end
       end
       
       vec_ini(i) = ini;
       vec_fin(i) = fin;
       %disp(['Inicio del escalon: ',num2str(ini), ' Fin del escalon: ', num2str(fin)]);
       escalon_QRS(ini:fin) = 1;
       no_ini=1;
       no_fin=1;
   end
   
end
plot(escalon_QRS), title 'Señal de pulsos QRS', xlabel 'Tiempo (muestras)', ylabel 'Amplitud';
%~~~~~~~~~~~~~~~~~~ONDA P
% Formacion señal escalon de onda P
    conf = 0;
    escalon_P = zeros(1,length(x(1,:)));
    ini= 1;
    vec_ini = zeros(1, length(position(1).Pon));
    fin = 1;
    vec_fin = zeros(1, length(position(1).Pon));
    no_ini = 1;
    no_fin = 1;
    latido_nulo = 0;
    for i=1:length(position(1).Pon)
       disp(['Analizando latido ', num2str(i)]);
       ini_candidates = [];
       fin_candidates = [];
       for k=1:12
         ini_candidates = [ini_candidates, position(k).Pon(i)];
         fin_candidates = [fin_candidates, position(k).Poff(i)];
       end

       if(sum(isnan(ini_candidates))<4 && sum(isnan(fin_candidates))<4)   % Si mas de 4 indices son NaN no proceso el latido
           while no_ini         % Estaremos en el while hasta que encontremos un ini confirmado
               conf=0;
               [ini, ind_ini] = min(ini_candidates);     % Inicialmente supongo que la primera deteccion es el inicio
               for j = 1:12
                    % Si despues de la supuesta primera deteccion vienen al menos 3
                    % mas, esa deteccion de inicio sera correcta
                   if abs(ini-position(j).Pon(i))<=margen                                           
                       conf = conf+1;
                   end
               end
               if conf>=3       % Confirmamos que ini es un valor real de inicio
                   no_ini=0;
                   disp('Inicio confirmado');
               else             % Si este no era un inicio real lo eliminamos de los posibles ini
                    ini_candidates(ind_ini) = [];
               end
               if isempty(ini_candidates)
                   latido_nulo = 1;
                    disp('Inicio no encontrado');
                   break;
               end
           end
           while no_fin         % Estaremos en el while hasta que encontremos un fin confirmado
               conf=0;
               [fin, ind_fin] = max(fin_candidates);     % Inicialmente supongo que la ultima deteccion es el final
               for j = 1:12
                    % Si antes de la supuesta ultima deteccion vienen al menos 3
                    % mas, esa deteccion de fin sera correcta
                   if abs(fin-position(j).Poff(i))<=margen                                           
                       conf = conf+1;
                   end
               end
               if conf>=3       % Confirmamos que fin es un valor real de final
                   no_fin=0;
                   disp('Final confirmado');
               else             % Si este no era un final real lo eliminamos de los posibles fin
                   fin_candidates(ind_fin) = [];
               end
               if isempty(fin_candidates)
                   latido_nulo = 1;
                   disp('Final no encontrado');
                   break;
               end
           end
           if (~latido_nulo)
               vec_ini(i) = ini;
               vec_fin(i) = fin;
               %disp(['Inicio del escalon: ',num2str(ini), ' Fin del escalon: ', num2str(fin)]);
               escalon_P(ini:fin) = 1;
           end
           no_ini=1;
           no_fin=1;
       end

    end
    
    %escalon_P = senialEscalonP(x,hea,position);
%~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
matriz = [];
N_antes = hea.freq*0.5;   % Muestras a coger antes del evento (medio segundo)
N_despues = hea.freq;     % Muestras a coger despues del evento (un segundo)
for i = 2:length(gl_ind)-1
    if(gl_ind(i)-N_antes>0 && gl_ind(i)+N_despues-1<length(escalon_QRS))
        row = escalon_QRS(gl_ind(i)-N_antes:gl_ind(i)+N_despues-1);
        matriz =[matriz; row];
    end
end

figure,
imagesc(matriz), title 'Imagen Señal de Pulsos QRS (Con "sobrante")' , ylabel 'Latido', xlabel ('Muestras en el latido');
c = colorbar();
c.Label.String ='mV';

%% Formacion de las imagenes con solo dos pulsos por fila (Finalmente no utilizado)
matriz = [];
N_antes = hea.freq*0.5;   % Muestras a coger antes del evento (medio segundo)
N_despues = hea.freq;     % Muestras a coger despues del evento (un segundo)
anterior = 0;
actual = 0;

for i = 2:length(gl_ind)-1
    if(gl_ind(i)-N_antes>0 && gl_ind(i)+N_despues-1<length(escalon_P))
        pulsos = 0;
        row = escalon_P(gl_ind(i)-N_antes:gl_ind(i)+N_despues-1);
        for j=1:length(row)
            actual = row(j);
            if (anterior == 0 && actual ==1)
                pulsos = pulsos + 1;
            end
            if pulsos>2
                row(j:end)=0;
                break;
            end
            anterior = actual;
        end
        matriz =[matriz; row];
    end
end

figure,
imagesc(matriz), title 'Imagen Señal de Pulsos P (Sin "sobrante")' , ylabel 'Latido', xlabel ('Muestras en el latido');
c = colorbar();
c.Label.String ='mV';




%% Creación y escritura de un fichero .hea con la info de fs cambiada.

fid = fopen(['Ficheros_Fs_Cambiada\',hea.recname,'.hea'], 'w');
for i=0:hea.nsig
    if i==0
        fprintf(fid, [hea.recname,' ',num2str(hea.nsig), ' ',num2str(hea.freq),' ', num2str(hea.nsamp), ' ', num2str(hea.date(1)), ' ',num2str(hea.date(2)), '\n']);
    else
        fprintf(fid,[hea.fname(i,:), ' ', num2str(hea.fmt(i)),'+',num2str(hea.offset(i)),' ', num2str(hea.gain(i)),'/',hea.units(i,:),' ', num2str(hea.adcres(i)),' ', num2str(hea.adczero(i)),' ', num2str(hea.initval(i)),' ',num2str(hea.cksum(i)),' ', num2str(hea.bsize(i)),' ', hea.desc(i,:),'\n' ]);
    end
end
fclose(fid);
sigdir = 'Training_WFDB_StPetersburg\';headir = 'Ficheros_Fs_Cambiada\';matdir = 'Training_WFDB_StPetersburg\';

position = [];      % Vector de stucts para almacenar los indices del delineador
for lead = 1:hea.nsig
    p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
    if length(fieldnames(p))~=19
       p = empty_struct; 
    end
    position = [position,p];
end
    

%% Mostrar imagenes (usando cell)
% Script que genera una figura con las 15 imagenes resultantes de procesar
% un fichero ECG. En esta version se considera que las imagenes estan
% guardadas en un cell de Matlab
file = 'Matrices procesadas\';
sig = 'A0001.mat';
load([file,'Matrices_',sig]);
leads = [' I ';' II';'III';'aVR';'aVL';'aVF';' V1';' V2';' V3';' V4';' V5';' V6'];
sinteticas = ['Escalones    QRS'; 'Escalones Onda P'; 'Escalones Onda T'];
figure,
for i=1:15
    if i<=12
        subplot(3,5,i), imagesc(matrices{i}), title(['Derivación ', leads(i,:)]);
        xlabel('Timepo (muestras)'), ylabel 'Latido';
    else
        subplot(3,5,i), imagesc(matrices{i}), title(sinteticas(i-12,:));
        xlabel('Timepo (muestras)'), ylabel 'Latido';
    end
end
sgtitle (['Derivaciones y Señales Sinteticas fichero ', num2str(sig)]);



%% Mostrar imagenes (usando tensor)
% Script que genera una figura con las 15 imagenes resultantes de procesar
% un fichero ECG. En esta version se considera que las imagenes estan
% guardadas en una estructura del tipo tensor (15xVx750)
file_path = 'Matrices procesadas\';
for fichero=1:1
    %fichero=166;
    if fichero<10
        num=['0000', num2str(fichero)];
        elseif (fichero>=10 && fichero<100)
            num=['000', num2str(fichero)];
        elseif (fichero>=100 && fichero<1000)
            num=['00', num2str(fichero)];         
        elseif (fichero>=1000 && fichero<10000)
            num=['0',num2str(fichero)];
        else
            num = num2str(fichero);
    end

    
    %sig = ['S',num,'.mat'];
    sig = ['A0010','.mat'];
    load([file_path,'Matrices_',sig]);
    leads = [' I ';' II';'III';'aVR';'aVL';'aVF';' V1';' V2';' V3';' V4';' V5';' V6'];
    sinteticas = ['Escalones    QRS'; 'Escalones Onda P'; 'Escalones Onda T'];
    matrices = permute(tensor, [2 3 1]);
    figure,
    for i=1:15
        if i<=12
            subplot(3,5,i), imagesc(matrices(:,:,i)), title(['Derivación ', leads(i,:)]);
            xlabel('Timepo (muestras)'), ylabel('Latido'), hcb=colorbar;
            colorTitleHandle = get(hcb,'Title');
            titleString = 'uV';
            set(colorTitleHandle ,'String',titleString);
            %caxis([-2000 2000]);
        else
            subplot(3,5,i), imagesc(matrices(:,:,i)), title(sinteticas(i-12,:));
            xlabel('Timepo (muestras)'), ylabel('Latido'), hcb=colorbar;
            colorTitleHandle = get(hcb,'Title');
            titleString = 'uV';
            set(colorTitleHandle ,'String',titleString);
        end
    end
    
    sgtitle (['Derivaciones y Señales Sinteticas fichero ', num2str(sig)]);
    
    %saveas(gcf,['FotosProcesadas\', 'S',num, '.png'], 'png');
    %close all;
end

% Bucle para guardar las imágenes lo mas limpias posibles y por separado (sin titulo, ni nada)
for i=1:15
    figure, imagesc(matrices(:,:,i));
    saveas(gcf,['DocumentosTFG\Bibliografia\', 'A0010_',num2str(i), '.png'], 'png');
    close all;
end
%% Representación de patologías en cada fichero
% Script que analiza las apariciones de cada patologia entre los ficheros
% ya procesados, haciendo uso de un diccionario de Matlab (keySet) y un
% mapa
keySet = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 59118001, 427393009,  426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001];
valueSet = 0:26;
M = containers.Map(keySet,valueSet);
patologies = [' IAVB '; '  AF  '; '  AFL '; 'Brady '; 'CRBBB '; 'IRBBB '; 'LAnFB '; '  LAD '; ' LBBB '; 'LQRSV '; 'NSIVCB'; '  PR  ' ;'  PAC ' ;'  PVC '; '  LPR '; '  LQT '; '  QAb '; '  RAD '; ' RBBB '; '  SA  '; '  SB  '; '  NSR '; 'STach '; ' SVPB '; '  TAb '; ' TInv '; '  VPB '];
file_path = 'Tensores_procesados_1_6_2021\';
tabla = zeros(5001, 27);
for fichero=1001:6001
   
    if fichero<10
        num=['000', num2str(fichero)];
    elseif (fichero>=10 && fichero<100)
        num=['00', num2str(fichero)];
    elseif (fichero>=100 && fichero<1000)
        num=['0', num2str(fichero)];         
    else
        num=num2str(fichero);
    end
    
        
    sig = ['A',num,'.mat'];
    disp(['Analizando fichero', sig]);
    load([file_path,'Matrices_',sig]);
    for i=1:length(diagnostico)
        if isKey(M, diagnostico(i))
            tabla(fichero-1000, M(diagnostico(i))+1) = 1;
        end
    end
    
    

end

figure, imagesc(tabla);
xticks(1:27);
xtickangle(45);
xticklabels({'IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR' ,'PAC' ,'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB'})
ylabel 'Numero de fichero', title 'Etiquetas presentes en ficheros del dataset CPS (A)';

%% Estudio de frecuencia de etiquetas
% Hare una matriz 27x27 en la que cada elemento i,j será las veces que
% coincice la patología i con la j
keySet = [270492004, 164889003, 164890007, 426627000, 713427006, 713426002, 445118002, 39732003, 164909002, 251146004, 698252002, 10370003, 284470004, 427172004, 164947007, 111975006, 164917005, 47665007, 59118001, 427393009,  426177001, 426783006, 427084000, 63593006, 164934002, 59931005, 17338001];
patologies = [' IAVB '; '  AF  '; '  AFL '; 'Brady '; 'CRBBB '; 'IRBBB '; 'LAnFB '; '  LAD '; ' LBBB '; 'LQRSV '; 'NSIVCB'; '  PR  ' ;'  PAC ' ;'  PVC '; '  LPR '; '  LQT '; '  QAb '; '  RAD '; ' RBBB '; '  SA  '; '  SB  '; '  NSR '; 'STach '; ' SVPB '; '  TAb '; ' TInv '; '  VPB '];
valueSet = 0:26;
M = containers.Map(keySet,valueSet);

tabla = zeros(27,27);

etiquetas_1 = zeros(1,27);          % Numero de apariciones generales de cada etiqueta
etiquetas_simples = zeros(1,27);    % Numero de apariciones en solitario de cada etiqueta
num_etiquetas_dobles = 0;           % Numero de etiquetas dobles en el dataset
etiquetas_dobles = [];              % Nume
num_etiquetas_triples = 0;          % Numero de etiquetas triples en el dataset
etiquetas_triples = [];

num_etiquetas = zeros(1,20);        % Numero de etiquetas que hay de cada tipo.
                                    % Index 1: Nº etiq simples
                                    % Index 2: Nº etiq dobles, ...

file_path = 'Tensores_procesados_PTB_30_5_2021\';
fid =fopen('Tensores_procesados_PTB_30_5_2021\NombresFicheros.txt');
l = fgetl(fid);
while l~=-1
   
%     if fichero<10
%         num=['000', num2str(fichero)];
%     elseif (fichero>=10 && fichero<100)
%         num=['00', num2str(fichero)];
%     elseif (fichero>=100 && fichero<1000)
%         num=['0', num2str(fichero)];         
%     else
%         num=num2str(fichero);
%     end
    
        
    %sig = ['Q',num,'.mat'];
    disp(['Analizando fichero ', l]);
    
    %load([file_path,'Matrices_',sig]);
    load([file_path,l]);
    num_diag = length(diagnostico);
    if num_diag>1
       for i=1:num_diag
           if isKey(M, diagnostico(i))
                etiquetas_1(M(diagnostico(i))+1) = etiquetas_1(M(diagnostico(i))+1) +1;
           end
           for j=1:num_diag
                if isKey(M,diagnostico(i)) && isKey(M, diagnostico(j))
                     
                    tabla(M(diagnostico(i))+1, M(diagnostico(j))+1) = tabla(M(diagnostico(i))+1, M(diagnostico(j))+1) + 1;
                end
           end
       end
    else
        if isKey(M, diagnostico)
            etiquetas_1(M(diagnostico)+1) = etiquetas_1(M(diagnostico)+1) + 1;
            etiquetas_simples(M(diagnostico)+1) = etiquetas_simples(M(diagnostico)+1) +1;
            num_etiquetas(1) = num_etiquetas(1)+1;
        end
    end
    if num_diag == 2
       num_etiquetas_dobles = num_etiquetas_dobles +1;
       num_etiquetas(2) = num_etiquetas(2)+1;
%        if ~isempty(etiquetas_dobles) && sum(sum(diagnostico== etiquetas_dobles(:, 1:2)))>= 2
%            ind_1 = find(etiquetas_dobles(:,1) == diagnostico(1));
%            ind_2 = find(etiquetas_dobles(:,2) == diagnostico(2));
%            pos = find((ind_1==ind_2)==1);
%            ind = 0;
%            if length(ind_2)> length(ind_1)
%                ind = ind_2(pos);
%                
%            else
%                ind = ind_1(pos);
%            end
%            etiquetas_dobles(diagnostico(1),3) = etiquetas_dobles(ind,3) +1;
%        else
%            etiquetas_dobles = [etiquetas_dobles; diagnostico, 1];
%        end
       
    elseif num_diag == 3
            num_etiquetas_triples = num_etiquetas_triples +1;
            num_etiquetas(3) = num_etiquetas(3)+1;
    
    elseif num_diag>3
        disp(['MAS DE 3 ETIQUETAS :', num2str(num_diag)]);
        num_etiquetas(num_diag) = num_etiquetas(num_diag)+1;
    end

    l = fgetl(fid);
end
fclose(fid);
%tabla = tabla.^~eye(27);        % Pongo a 0 la diagonal
for i=1:27
    for j=1:27
        if i==j
            tabla(i,j) = 0;
            break;
        end
    end
end
figure, imagesc(tabla_general), title 'Relaciones entre etiquetas (Suma de Datasets)';
hbc = colorbar; grid on;
ylabel(hbc, 'Nº Apariciones', 'FontSize', 15);
xticks(1:27);
yticks(1:27);
xtickangle(45);
ytickangle(0);
xticklabels({'IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR' ,'PAC' ,'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB'})
yticklabels({'IAVB', 'AF', 'AFL', 'Brady', 'CRBBB', 'IRBBB', 'LAnFB', 'LAD', 'LBBB', 'LQRSV', 'NSIVCB', 'PR' ,'PAC' ,'PVC', 'LPR', 'LQT', 'QAb', 'RAD', 'RBBB', 'SA', 'SB', 'NSR', 'STach', 'SVPB', 'TAb', 'TInv', 'VPB'});
%% Comprobacion de dimensiones de tensores
% Barremos los ficheros procesados para detectar si hay alguno para el cual
% no se han generado las 15 imagenes
file_path = 'Tensores_procesados_CPS_1_6_2021\';
ficheros_nulos = '';
for fichero=1:6877
   
    if fichero<10
        num=['000', num2str(fichero)];
    elseif (fichero>=10 && fichero<100)
        num=['00', num2str(fichero)];
    elseif (fichero>=100 && fichero<1000)
        num=['0', num2str(fichero)];         
    else
        num=num2str(fichero);
    end
    
        
    sig = ['A',num,'.mat'];
    disp(['Analizando fichero', sig]);
    load([file_path,'Matrices_',sig]);
    size_tensor = size(tensor);
    if size_tensor(1)~=15 || size_tensor(3)~=750
        disp("Dimensiones Erroneas");
        ficheros_nulos = [ficheros_nulos, sig];
    end
end

%% Comprobación formato NombresFicheros.txt
% Comprobamos que el fichero txt que contiene los nombres de los ficheros
% de imagenes procesadas esta bien escrito
fid =fopen('NombresFicheros.txt');
l = fgetl(fid);
cont = 0;
while l~=-1
   if strcmp(l, '.mat') 
        disp('Línea Erronea');
   end
   l = fgetl(fid);
   cont = cont + 1;
end
fclose(fid);

%% Buscar ficheros con patologías concretas (Ficheros con solo una patología)
% Buscamos ficheros concretos que tengan estas patologias para poder formar
% un subconjunto de ejemplo para un entrenamiento de la red neuronal
% NSR: 426783006, AF: 164889003, AFL: 164890007 y PR: 10370003
%targets = [426783006, 164889003, 164890007,10370003];
matches = zeros(1,2);
targets = [426783006, 164889003];
%matches = 0;
char = 'A';
file_path = 'Tensores_procesados_CPS_1_6_2021\';
fid =fopen('Tensores_procesados_CPS_1_6_2021\NombresFicheros.txt');
l = fgetl(fid);
while l~=-1
    
    disp(['Analizando fichero ', l]);
    load([file_path,l]);
    if length(diagnostico)==1
        if sum(diagnostico==targets)==1
            copyfile([file_path,l], 'SubsetNSRvsAF');
            fileID = fopen('SubsetNSRvsAF\NombresFicheros.txt', 'a');
            fprintf(fileID, [l,'\n']);
            fclose(fileID);
            matches = matches + (diagnostico==targets);
        end
    end

    l = fgetl(fid);
end
matches

%% Los ficheros de estas patologías resultan insuficientes. Version sin exigir etiquetas simples
% Buscamos ficheros que contengan estas patologias aunque no sean etiquetas
% de esa unica patologia
% NSR: 426783006, AF: 164889003, IAVB: 270492004, RBBB: 59118001

targets = [426783006, 164889003, 270492004, 59118001];
matches = zeros(1,4);
file_path = 'Tensores_procesadosStPetersburg_30_5_2021\';
fid =fopen('Tensores_procesadosStPetersburg_30_5_2021\NombresFicheros.txt');
l = fgetl(fid);
while l~=-1
    
    disp(['Analizando fichero ', l]);
    load([file_path,l]);
    for i=1:length(diagnostico)
        if sum(diagnostico(i)==targets)==1
            copyfile([file_path,l], 'SubsetEtiquetasMultiples');
            fileID = fopen('SubsetEtiquetasMultiples\NombresFicheros.txt', 'a');
            fprintf(fileID, [l,'\n']);
            fclose(fileID);
            matches = matches + (diagnostico(i)==targets);
            break;
        end
    end

    l = fgetl(fid);
end
matches

%% Buscamos otras patologías
% Tratamos de escoger ficheros que contengan alguna de la siguientes
% patologias entre sus etiquetas para entiquecer y aumentar la complejidad
% de la red.

% NSR: 426783006, AF: 164889003, IAVB: 270492004, RBBB: 59118001, LBBB: 164909002 

% Finalmente se entreno el modelo de red mas complejo con un subconjunto de
% estas 4 patologias:
% NSR: 426783006, AF: 164889003, IAVB: 270492004, RBBB: 59118001

targets = [426783006, 164889003, 270492004, 59118001];%, 164909002];
%targets = keySet(13);
matches = zeros(1,4);

nsr = 0;
af = 0;
iavb = 0;
rbbbs = 0;

file_path = 'Tensores_procesados_CPS_1_6_2021\';
fid =fopen('Tensores_procesados_CPS_1_6_2021\NombresFicheros.txt');
%numero_etiquetas = zeros(1,10);
l = fgetl(fid);
while l~=-1
        
    disp(['Analizando fichero ', l]);
    load([file_path,l]);
    if length(diagnostico)==1
        if sum(diagnostico(1)==targets)==1
%             copyfile([file_path,l], 'Subset4PatologiasV2');
%             fileID = fopen('Subset4PatologiasV2\NombresFicheros.txt', 'a');
%             fprintf(fileID, [l,'\n']);
%             fclose(fileID);
            ind= find((diagnostico==targets)==1);
            if ind==4
                rbbbs = rbbbs+1;
                if rbbbs<=625
                    matches(ind) = matches(ind) + 1;
                    copyfile([file_path,l], 'SubsetNSRvsAFvsAIVBvsRBBB');
                    fileID = fopen('SubsetNSRvsAFvsAIVBvsRBBB\NombresFicheros.txt', 'a');
                    fprintf(fileID, [l,'\n']);
                    fclose(fileID);
                end
            end
            if ind==3
                iavb = iavb+1;
                if iavb<=625
                    matches(ind) = matches(ind) + 1;
                    copyfile([file_path,l], 'SubsetNSRvsAFvsAIVBvsRBBB');
                    fileID = fopen('SubsetNSRvsAFvsAIVBvsRBBB\NombresFicheros.txt', 'a');
                    fprintf(fileID, [l,'\n']);
                    fclose(fileID);
                end
            end
            if ind==2
                af = af+1;
                if af<=625
                    matches(ind) = matches(ind) + 1;
                    copyfile([file_path,l], 'SubsetNSRvsAFvsAIVBvsRBBB');
                    fileID = fopen('SubsetNSRvsAFvsAIVBvsRBBB\NombresFicheros.txt', 'a');
                    fprintf(fileID, [l,'\n']);
                    fclose(fileID);
                end
            end
            if ind==1
                nsr = nsr+1;
                if nsr<=625
                    matches(ind) = matches(ind) + 1;
                    copyfile([file_path,l], 'SubsetNSRvsAFvsAIVBvsRBBB');
                    fileID = fopen('SubsetNSRvsAFvsAIVBvsRBBB\NombresFicheros.txt', 'a');
                    fprintf(fileID, [l,'\n']);
                    fclose(fileID);
                end
            end

        end
    end
    %numero_etiqutas(i) = numero_etiqutas(i)+1;
    l = fgetl(fid);
end
fclose(fid);
matches

%% Analizamos las dimensiones de un dataset
% Comprobacion de si algun fichero de imagenes procesadas no ha generado
% imágenes.
% Los ficheros que se encontraron solo supusieron un 0.004% del total, por
% lo que se descartaron para el entrenamiento
file_path = 'SubsetEtiquetasMultiples\';
fid =fopen('SubsetEtiquetasMultiples\NombresFicheros.txt');
l = fgetl(fid);
erroneos = [];
num_erroneos = 0;
while l~=-1
    
    disp(['Analizando fichero ', l]);
    load([file_path,l]);
    s = size(tensor);
    if s(2)==0
        disp('Fichero Inválido');
        erroneos = [erroneos, ' ', l];
        num_erroneos = num_erroneos + 1;
    end
    
    l = fgetl(fid);
end

%% Prestaciones de las variaciones en el modelo AUC vs Dimensionalidad
% Tras realizar los entrenamientos y evaluar los distintos modelos se han
% obtenido estos valores de AUC media entre las clases y de numero de
% parametros del modelo 
auc_modelo3 = 0.9733623399992303;
param_modelo3 = 927628;

auc_conv4_sin_stride = 0.9690202971092837;
param_conv4_sin_stride = 2009740;

auc_conv4_con_stride = 0.9711416187829142;
param_conv4_con_stride = 1223308;

auc_lin3 = 0.9741791196908961;
param_lin3 = 992012;

auc_conv4_lin01_lin3 = 0.9719185086884544;
param_conv4_lin01_lin3 = 2601484;

auc_1con5_32 = 0.973727870074335;
param_1con5_32 = 796192;

auc_2con5_32 = 0.9723643792468178;
param_2con5_32 = 117749944;

auc_1con5_16 = 0.9686113198208173;
param_1con5_16 = 203606;

auc_2con5_16 = 0.9693688269521479;
param_2con5_16 = 29526178;


semilogx(log10(param_conv4_sin_stride), auc_conv4_sin_stride, 'rx', 'MarkerSize', 10);
grid on, hold on;
semilogx(log10(param_conv4_con_stride), auc_conv4_con_stride, 'bx', 'MarkerSize', 10);
%hold on;
semilogx(log10(param_lin3), auc_lin3, 'kx', 'MarkerSize', 10);
%hold on;
semilogx(log10(param_conv4_lin01_lin3), auc_conv4_lin01_lin3, 'gx', 'MarkerSize',10);
%hold on;
semilogx(log10(param_1con5_32), auc_1con5_32, 'c^', 'MarkerSize', 7);
%hold on;
semilogx(log10(param_2con5_32), auc_2con5_32, 'm^', 'MarkerSize', 7);
%hold on;
semilogx(log10(param_1con5_16), auc_1con5_16, 'g^' , 'MarkerSize', 7);
%hold on;
semilogx(log10(param_2con5_16), auc_2con5_16, 'r^', 'MarkerSize',7);
%hold on;
semilogx(log10(param_modelo3), auc_modelo3, 'b*', 'MarkerSize', 10);

hold off;
xlabel 'Número de parámetros del modelo (log10)', ylabel 'AUC media del modelo', title 'AUC vs Complejidad del modelo';
%legend(['conv4 (sin stride)'], ['conv4 (con stride)'], ['lin3'], ['conv4 + lin01 + lin3'], ['1.5, 32 canales'],['2.5, 32 canales'], ['1.5, 16 canales'], ['2.5, 16 canales'], ['Modelo Inicial']);
legend(['Modelo 1 (sin stride)'], ['Modelo 1 (con stride)'], ['Modelo 2'], ['Modelo 3'], ['Modelo 4-32-1.5'],['Modelo 4-32-2.5'], ['Modelo 4-16-1.5'], ['Modelo 4-16-2.5'], ['Modelo 0 (Modelo Base)']);


