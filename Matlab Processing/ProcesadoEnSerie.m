%% Procesado
%Definimos un struct vacio para la gestion de errores del delineador
empty_struct = struct('Pon',{[]},'P',{[]},'Poff',{[]}, 'QRSon',{[]},'Q',{[]},'R',{[]},'Fiducial',{[]},'qrs',{[]},'Rprima',{[]},'S',{[]},'QRSoff',{[]}, 'Ton',{[]},'T',{[]},'Tprima',{[]},'Toff',{[]},'Ttipo',{[]},'QRSmainpos',{[]},'QRSmaininv',{[]},'Pprima',{[]});
% Indicamos el dataset que queremos procesar
hea_path = 'Training_WFDB_CPS\';
char = 'A';
% hea_path = 'Training_WFDB_CPS_2\';
% char = 'Q';
%hea_path = 'Training_WFDB_PTB\';
%char = 'S';
% hea_path = 'Training_WFDB_StPetersburg\';
% char = 'I';

% Iteraremos sobre los diferentes ficheros del dataset generando para cada
% ECG las 15 imagenes procesadas necesarias para el entrenamiento y/o
% evaluacion de la red neuronal
for fichero=1:549
   remuestreo = 0;      % Booleano que nos indica si hemos remuestrado la señal
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~LECTURA DATOS
    disp(['Procesando fichero ',num2str(fichero)]);
    if fichero<10
        num=['000', num2str(fichero)];
        elseif (fichero>=10 && fichero<100)
                num=['00', num2str(fichero)];
            elseif (fichero>=100 && fichero<1000)
                    num=['0', num2str(fichero)];
                else
                    num = num2str(fichero);
     end
    
    file_path = hea_path;
    filname = [char,num,'.mat'];
    file = [file_path,filname];
    header = [hea_path,char,num,'.hea'];
    lead = 1;
    hea = readheader(header);   % -> Struct con todos los campos de informacion
    comments = my_comment_reader(header);       % Leemos el fichero .hea
    diagnostico = str2num(comments(3));         % Extraemos el/los diagnosticos
    fid=fopen(file);
    x=fread(fid,[12 Inf],'int16');
    fclose(fid);
    sig = x(:,2:end);
    senializacion = x(:,1);
    sig=1000/hea.gain(lead) * sig; % in microvolts (para tener la señal en microvoltios incluso siendo otra la ganancia)
    %sigdir = 'Training_WFDB_StPetersburg\';headir = 'Training_WFDB_StPetersburg\';matdir = 'Training_WFDB_StPetersburg\';
    sigdir = hea_path; headir = hea_path ;matdir = hea_path;
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~REMUESTREO
    % Si se detecta que el ECG no esta muestreado a 500Hz se remuestrea y
    % guarda en otro fichero
    if hea.freq~=500
        remuestreo = 1;
        disp('Remuestreando señal');
        sig2 = resample(sig', 500, hea.freq)'; % Remuestreamos para tenerla a 500 Hz
        clear x;
        x = sig2;
        sig2 = int16(sig2);
        datos = [senializacion sig2];
        
        fid2 = fopen(['Ficheros_Fs_Cambiada\',hea.recname,'.mat'], 'w+');
        fwrite(fid2, datos,'int16');
        fclose(fid2);
        
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
    else
        clear x;
        x = sig;
        remuestreo = 0;
    end
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~DETECTOR DE PICOS NO FISIOLOGICOS
    % Aplicamos un criterio sobre el rango dinámico de la señal, por el
    % cual aplicaremos un filtro de mediana al superarse un rango maximo
    filtrar = 0;
    max_diferencia = 5000;
    for i=1:hea.nsig
        if max(x(i,:))-min(x(i,:))> max_diferencia
            disp('Hay que filtrar');
            filtrar = 1;
           % Aplico filtro de mediana y guardo resultado como x(i,:)
           x(i,:) = medfilt1(x(i,:), 5);
        end
        
    end
    % Guardo fichero en carpeta correspondiente con formato int16
    carpeta = '';
    % En caso de haber remuestreado anteriormente la señal que debere
    % filtrar sera la que se encuentra ya remuestreada en el directorio 'Ficheros_Fs_Cambiada'
    if filtrar == 1
        if remuestreo == 1      
            carpeta = 'Ficheros_Fs_Cambiada\';
        else
            carpeta = 'FicherosFiltrados\';
            copyfile([hea_path,hea.recname,'.hea'], 'FicherosFiltrados');
        end
        datos = [senializacion, x];
        fid2 = fopen([carpeta,hea.recname,'.mat'], 'w+');
        fwrite(fid2, datos,'int16');
        fclose(fid2);

        headir = carpeta;
        sigdir = carpeta;
        matdir = carpeta;
    end
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~DELINEADOR
    disp('Delineando señal');
    position = [];      % Vector de stucts para almacenar los indices del delineador
    for lead = 1:hea.nsig
        try
        p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],0);
        if length(fieldnames(p))~=19
           p = empty_struct; 
        end
        position = [position,p];
        catch
            disp('Warning: Fallo de wavedet capturado. Introduciendo empty_struct');
            position = [position,empty_struct];
            continue
        end
    end
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~BASELINE
    disp('Eliminando Baseline');
    not_samples = hea.freq*80e-3;
    x_limpia = zeros(size(x,1),size(x,2));
    for lead = 1:hea.nsig
        disp(['Limpiando lead: ', num2str(lead)]);
        x_lead = double(x(lead,:));
        % En señales dañadas o ruidosas el delineador falla poniendo NaN en las anotaciones
        % o un unico valor (en vez de un vector de indices). En estos casos
        % baseline2 falla, asi que evito su ejecucion y no limpio esa
        % derivacion
        if length(position(lead).qrs)>1         
            x_limpia(lead,:) = baseline2(x_lead', round(position(lead).qrs - 0.08 *hea.freq), 0, not_samples)';
        else
           x_limpia(lead, :) = x_lead; 
        end
    end
    clear x;
    x=x_limpia;
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GLOBAL IND
    disp('Calculando indices globales');
    N_mismo_latido = 500*100*1e-3;    
    v = hanning(N_mismo_latido);
    deltas = zeros(12,size(x,2));
    convs = zeros(12, size(x,2)+length(v)-1);
    for i = 1:hea.nsig
        if ~isnan(position(i).qrs)      % Si alguno de los indices es NaN no entra en el calculo
            deltas(i,position(i).qrs)=1;
        end
        convs(i,:) = conv(deltas(i,:),v);
    end

    latidos = sum(convs,1);
    [picos, ~] = findpeaks(latidos);
    umbral = 0.6*max(picos);                        % Tomare como umbral el 60% de la amplitud de pico mayor
    [pks,gl_ind] = findpeaks(latidos,'MinPeakHeight',umbral);     
    if isempty(gl_ind)
        disp('Warning: Fallo de calculo de indices globales de QRS');
    end
    gl_ind = gl_ind-round(N_mismo_latido/2);        % Corregimos el desfase de las ventanas
    save('AnotacionesQRS\anot.mat','gl_ind');
    
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ANOTACIONES EXTERNAS
    % Delineo de nuevo las señales empleando anotaciones externas
    dirann = 'AnotacionesQRS\';
    position = [];      % Vector de stucts para almacenar los indices del delineador
    for lead = 1:hea.nsig
        p = wavedet(sigdir, headir,matdir, hea.recname, 0, 'wav', lead, [1 Inf],1,1,'anot',dirann);
        position = [position, p];
    end
    %~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~SEÑALES SINTETICAS
    disp('Generando señales sinteticas');
    % Formacion señal escalon de QRS
    margen = hea.freq*12e-3;
    conf = 0;
    escalon_QRS = zeros(1,length(x(1,:)));
    ini= 1;
    vec_ini = zeros(1, length(position(1).QRSon));
    fin = 1;
    vec_fin = zeros(1, length(position(1).QRSon));
    no_ini = 1;
    no_fin = 1;
    latido_nulo = 0;
    disp('-----------QRS');
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
               if isempty(ini_candidates)
                  latido_nulo = 1;
                  break;
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
               if isempty(fin_candidates)
                   latido_nulo = 1;
                   break;
               end
           end
           if (~latido_nulo)
               vec_ini(i) = ini;
               vec_fin(i) = fin;
               %disp(['Inicio del escalon: ',num2str(ini), ' Fin del escalon: ', num2str(fin)]);
               escalon_QRS(ini:fin) = 1;
           end
           no_ini=1;
           no_fin=1;
           latido_nulo=0;
       end

    end
    %figure, plot(escalon_QRS), title 'Señal de pulsos QRS', xlabel 'Tiempo (muestras)', ylabel 'Amplitud';
    disp('--------Onda P');
    % Formacion señal escalon de onda P
    latido_nulo=0;
    conf = 0;
    escalon_P = zeros(1,length(x(1,:)));
    ini= 1;
    vec_ini = zeros(1, length(position(1).Pon));
    fin = 1;
    vec_fin = zeros(1, length(position(1).Pon));
    no_ini = 1;
    no_fin = 1;
    for i=1:length(position(1).Pon)
       %disp(['Analizando latido ', num2str(i)]);
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
                   %disp('Inicio confirmado');
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
                   %disp('Final confirmado');
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
           latido_nulo = 0;
       end

    end
    %hold on, plot(escalon_P), title 'Señal de pulsos P', xlabel 'Tiempo (muestras)', ylabel 'Amplitud';
    disp('--------Onda T');
    % Formacion señal escalon de onda T
    latido_nulo=0;
    conf = 0;
    escalon_T = zeros(1,length(x(1,:)));
    ini= 1;
    vec_ini = zeros(1, length(position(1).Ton));
    fin = 1;
    vec_fin = zeros(1, length(position(1).Ton));
    no_ini = 1;
    no_fin = 1;
    for i=1:length(position(1).Ton)
       %disp(['Analizando latido ', num2str(i)]);
       ini_candidates = [];
       fin_candidates = [];
       for k=1:12
         ini_candidates = [ini_candidates, position(k).Ton(i)];
         fin_candidates = [fin_candidates, position(k).Toff(i)];
       end

       if(sum(isnan(ini_candidates))<4 && sum(isnan(fin_candidates))<4)   % Si mas de 4 indices son NaN no proceso el latido
           while no_ini         % Estaremos en el while hasta que encontremos un ini confirmado
               conf=0;
               [ini, ind_ini] = min(ini_candidates);     % Inicialmente supongo que la primera deteccion es el inicio
               for j = 1:12
                    % Si despues de la supuesta primera deteccion vienen al menos 3
                    % mas, esa deteccion de inicio sera correcta
                   if abs(ini-position(j).Ton(i))<=margen                                           
                       conf = conf+1;
                   end
               end
               if conf>=3       % Confirmamos que ini es un valor real de inicio
                   no_ini=0;
                   %disp('Inicio confirmado');
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
                   if abs(fin-position(j).Toff(i))<=margen                                           
                       conf = conf+1;
                   end
               end
               if conf>=3       % Confirmamos que fin es un valor real de final
                   no_fin=0;
                   %disp('Final confirmado');
               else             % Si este no era un final real lo eliminamos de los posibles fin
                   fin_candidates(ind_fin) = [];
               end
               if isempty(fin_candidates)
                   latido_nulo = 1;
                    disp('Final no encontrado');
                   break;
               end
           end
            if(~latido_nulo)
               vec_ini(i) = ini;
               vec_fin(i) = fin;
               %disp(['Inicio del escalon: ',num2str(ini), ' Fin del escalon: ', num2str(fin)]);
               escalon_T(ini:fin) = 1;
            end
            no_ini=1;
            no_fin=1;
            latido_nulo = 0;
       end

    end
    %hold on, plot(escalon_T), title 'Señal de pulsos T', xlabel 'Tiempo (muestras)', ylabel 'Amplitud';
    
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~FORMACION DE IMAGENES
    disp('Formando imagenes');
    %matrices = cell(1, hea.nsig+3);
    %matrices = zeros(length(position(lead).qrs)-2,750,15);
    matrices = [];
    matriz = [];
    N_antes = hea.freq*0.5;   % Muestras a coger antes del evento
    N_despues = hea.freq;     % Muestras a coger despues del evento
    for lead=1:hea.nsig
        x_lead = x(lead,:);
        for i = 2:length(position(lead).qrs)-1
            if(gl_ind(i)-N_antes>0 && gl_ind(i)+N_despues-1<length(x_lead))
                row = x_lead(gl_ind(i)-N_antes:gl_ind(i)+N_despues-1);
                matriz =[matriz; row];
            end
        end
        matrices(:,:,lead) =  matriz;
        matriz = [];
    end
    % matrices ahora contiene las matrices imagen de cada derivacion
    % Falta añadir las matrices imagen de las señales sinteticas
    matriz = [];
    for i = 2:length(gl_ind)-1
        if(gl_ind(i)-N_antes>0 && gl_ind(i)+N_despues-1<length(escalon_QRS))
            row = escalon_QRS(gl_ind(i)-N_antes:gl_ind(i)+N_despues-1);
            matriz =[matriz; row];
        end
    end
    matrices(:,:,hea.nsig+1) =  matriz;
    matriz = [];
    for i = 2:length(gl_ind)-1
        if(gl_ind(i)-N_antes>0 && gl_ind(i)+N_despues-1<length(escalon_P))
            row = escalon_P(gl_ind(i)-N_antes:gl_ind(i)+N_despues-1);
            matriz =[matriz; row];
        end
    end
    matrices(:,:,hea.nsig+2) =  matriz;
    matriz = [];
    for i = 2:length(gl_ind)-1
        if(gl_ind(i)-N_antes>0 && gl_ind(i)+N_despues-1<length(escalon_T))
            row = escalon_T(gl_ind(i)-N_antes:gl_ind(i)+N_despues-1);
            matriz =[matriz; row];
        end
    end
    matrices(:,:,hea.nsig+3) =  matriz;
    % ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~GUARDADO DE LAS IMAGENES
    disp('Guardando imagenes');
    matrices = single(matrices);            % Las guardamos con precisión simple
    tensor = permute(matrices, [3 1 2]);
    save(['Matrices procesadas\Matrices_',filname], 'tensor', 'diagnostico');
    
    fileID = fopen('NombresFicheros.txt', 'a');     %Escribimos el nombre del fichero procesado en el fichero de nombres
    fprintf(fileID, ['Matrices_',filname,'\n']);        
    fclose(fileID);
end


