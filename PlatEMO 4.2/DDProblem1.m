classdef DDProblem1 < PROBLEM
% <multi/many> <real> <large/none> <expensive/none>
% Benchmark MOP proposed by Zitzler, Deb, and Thiele

%------------------------------- Reference --------------------------------
% E. Zitzler, K. Deb, and L. Thiele, Comparison of multiobjective
% evolutionary algorithms: Empirical results, Evolutionary computation,
% 2000, 8(2): 173-195.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            py_M_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_M.txt');
            disp('Reading number of objectives from from py_M.txt');
            while 1
                if isfile(py_M_file)
                    M_file = fopen(py_M_file);
                    obj.M = fscanf(M_file, '%d');
                    fclose(M_file);
                    delete(py_M_file);
                    break;
                end
            end

            py_N_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_N.txt');
            disp('Reading population size from from py_N.txt');
            while 1
                if isfile(py_N_file)
                    N_file = fopen(py_N_file);
                    obj.N = fscanf(N_file, '%d');
                    fclose(N_file);
                    delete(py_N_file);
                    break;
                end
            end

            if isempty(obj.D); obj.D = 512; end
            %disp("Trying to read the lower and upper bounds from py_LOWER.txt and py_UPPER.txt");
            %disp("Run the python file now.");
            %while 1
            %    if isfile('py_LOWER.txt')
            %        obj.lower = readmatrix('py_LOWER.txt');
            %        delete 'py_LOWER.txt';
            %    end
            %    if isfile('py_UPPER.txt')
            %        obj.upper = readmatrix('py_UPPER.txt');
            %        delete 'py_UPPER.txt';
            %    end
            %    if (exist('obj.lower', 'var')) && (exist('obj.upper', 'var'))
            %        break;
            %    end
            %end
            disp('Reading lower bound from py_LOWER.txt');
            py_lower_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_LOWER.txt');
            while 1
                if isfile(py_lower_file)
                    obj.lower = readmatrix(py_lower_file);
                    delete(py_lower_file);
                    break;
                end
            end

            disp('Reading upper bound from py_UPPER.txt');
            py_upper_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_UPPER.txt');
            while 1
                if isfile(py_upper_file)
                    obj.upper = readmatrix(py_upper_file);
                    delete(py_upper_file);
                    break;
                end
            end

            obj.encoding = ones(1,obj.D);
        end
        %% Calculate objective values
        function PopObj = CalObj(~,~)
            py_OBJ_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_OBJ.txt');
            py_EMB_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_EMB.txt');
            while 1
                if (isfile(py_OBJ_file)) && ~(isfile(py_EMB_file))
                    PopObj = readmatrix(py_OBJ_file);
                    delete(py_OBJ_file);
                    %disp(strcat('read popOBJ  ', string(milliseconds(datetime('now','Timezone','UTC')-datetime('1970-01-01','Timezone','UTC')))));
                    break;
                end
            end
        end
        function R = GetOptimum(obj,N)
            R = zeros(N,obj.M);
        end
        function R = GetPF(obj)
            R = obj.GetOptimum(100);
        end
        function Population = Initialization(obj,~)
            % Initialize Population with random molecules from a dataset,
            % done on the Python end of things and just read in here.
            py_init_pop_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_init_pop.txt');
            disp('Reading initial population from from py_init_pop.txt');
            while 1
                if isfile(py_init_pop_file)
                    PopDec = readmatrix(py_init_pop_file);
                    delete(py_init_pop_file);
                    break;
                end
            end
            Population = obj.Evaluation(PopDec);
        end
        function PopDec = CalDec(obj,PopDec)
        %CalDec - Repair multiple invalid solutions.
        %
        %   Dec = obj.CalDec(Dec) repairs the invalid (not infeasible)
        %   decision variables in Dec.
        %
        %   An invalid solution indicates that it is out of the decision
        %   space, while an infeasible solution indicates that it does not
        %   satisfy all the constraints.
        %
        %   This function is usually called by PROBLEM.Evaluation.
        %
        %   Example:
        %       PopDec = Problem.CalDec(PopDec)

            Type  = arrayfun(@(i)find(obj.encoding==i),1:5,'UniformOutput',false);
            index = [Type{1:3}];
            if ~isempty(index)
                PopDec(:,index) = max(min(PopDec(:,index),repmat(obj.upper(index),size(PopDec,1),1)),repmat(obj.lower(index),size(PopDec,1),1));
            end
            index = [Type{2:5}];
            if ~isempty(index)
                PopDec(:,index) = round(PopDec(:,index));
            end

            py_repaired_emb_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'py_EMB.txt');
            matlab_repair_emb_tmp_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'matlab_REPAIR_EMB_tmp.txt');
            matlab_repair_emb_file = fullfile("~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2", '', 'matlab_REPAIR_EMB.txt');
            writematrix(PopDec, matlab_repair_emb_tmp_file);
            movefile(matlab_repair_emb_tmp_file, matlab_repair_emb_file);
            while 1
                if isfile(py_repaired_emb_file) && ~isfile(matlab_repair_emb_file)
                    PopDec = readmatrix(py_repaired_emb_file);
                    delete(py_repaired_emb_file);
                    break;
                end
            end
        end
    end
end