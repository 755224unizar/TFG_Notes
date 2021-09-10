function [comments] = my_comment_reader(file)
% Funcion que lee los ficheros cabecera y devuelve en una matriz los datos
% de edad y sexo del paciente y los diagnosticos del mismo.

% Daniel Tosaus Lanza (2021)
    fid =fopen(file);
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
end

