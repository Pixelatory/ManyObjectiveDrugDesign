% RUN THIS FIRST, THEN THE PYTHON SCRIPT!

matlab_location = "~/Desktop/ManyObjectiveDrugDesign/PlatEMO 4.2";
% Note: may need to change file location strings in DDProblem1.m

removeAllFiles(matlab_location);
disp("Deleted communication text files to start a fresh run.");
py_shutdown_file = fullfile(matlab_location, '', 'py_SHUTDOWN.txt');
random_states = [42, 182, 625, 511, 310];

start_run(@NMPSO, 1:5, random_states, matlab_location);
start_run(@GrEA, 1:5, random_states, matlab_location);
start_run(@HypE, 1:5, random_states, matlab_location);
start_run(@KnEA, 1:5, random_states, matlab_location);
start_run(@MOEADD, 1:5, random_states, matlab_location);
start_run(@ANSGAIII, 1:5, random_states, matlab_location);

writematrix([], py_shutdown_file);

function removeFileIfExists(path)
    if isfile(path)
        delete(path)
    end
end
function removeAllFiles(matlab_location)
    py_shutdown_file = fullfile(matlab_location, '', 'py_SHUTDOWN.txt');
    py_new_run_file = fullfile(matlab_location, '', 'py_NEW_RUN.txt');
    removeFileIfExists(py_shutdown_file);
    removeFileIfExists(fullfile(matlab_location, '', 'py_OBJ.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_OBJ_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_EMB.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_EMB_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LOWER.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_LOWER_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_UPPER.txt'));
    removeFileIfExists(fullfile(matlab_location', '', 'py_UPPER_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_M.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_M_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_N.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_N_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_init_pop.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_init_pop_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'matlab_REPAIR_EMB.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'matlab_REPAIR_EMB_tmp.txt'));
    removeFileIfExists(fullfile(matlab_location, '', 'py_NEW_RUN_tmp.txt'));
    removeFileIfExists(py_new_run_file);
end

function start_run(alg, run_range, random_states, matlab_location)
    py_new_run_tmp_file = fullfile(matlab_location, '', 'py_NEW_RUN_tmp.txt');
    py_new_run_file = fullfile(matlab_location, '', 'py_NEW_RUN.txt');
    removeAllFiles(matlab_location);
    for i = run_range
        rng(random_states(i)); % sets the seed to your desired value
        writematrix([random_states(i)], py_new_run_tmp_file);
        movefile(py_new_run_tmp_file, py_new_run_file);
        platemo('algorithm',alg,'problem',@DDProblem1,'maxFE',25000,'save',100000);
    end
end