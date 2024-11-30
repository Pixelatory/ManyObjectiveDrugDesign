import os
import subprocess
from pathlib import Path

import psutil
import re

from meeko import MoleculePreparation
from rdkit import Chem
from rdkit.Chem import AllChem


def dock(receptor_input,
         smiles,
         ligand_name,
         center_x=14.444,
         center_y=5.250,
         center_z=-18.278,
         size_x=20,
         size_y=20,
         size_z=20,
         lig_dir=None,
         out_dir=None,
         log_dir=None,
         conf_dir=None,
         vina_cwd=None,
         seed=None):
    timeout_duration = 10000

    # mkdir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(lig_dir, exist_ok=True)

    ligand = f"{lig_dir}/{ligand_name}.pdbqt"
    output = f"{out_dir}/{ligand_name}.pdbqt"
    config = f"{conf_dir}/{ligand_name}.txt"
    log = f"{log_dir}/{ligand_name}.txt"

    if not Path(ligand).exists():
        try:
            mol = Chem.MolFromSmiles(smiles)
            mol = AllChem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            preparator = MoleculePreparation()
            preparator.prepare(mol)
            pdbqt_string = preparator.write_pdbqt_string()
            with open(ligand, 'w') as f:
                f.write(pdbqt_string)
        except:
            print("Couldn't write as PDBQT string")
            return 1000000.0

    # Dock
    if os.path.isfile(receptor_input):
        # Create configuration files
        if not Path(config).exists():
            conf = 'receptor = ./proteins/LPA1-7yu4.pdbqt\n' + \
                   'ligand = ' + ligand + '\n' + \
                   'out = ' + output + '\n' + \
                   'center_x = ' + str(center_x) + '\n' + \
                   'center_y = ' + str(center_y) + '\n' + \
                   'center_z = ' + str(center_z) + '\n' + \
                   'size_x = ' + str(size_x) + '\n' + \
                   'size_y = ' + str(size_y) + '\n' + \
                   'size_z = ' + str(size_z) + '\n' + \
                   'thread=5000'

            if seed is not None:
                conf += 'seed = ' + str(seed) + '\n'

            with open(config, 'w') as f:
                f.write(conf)

        if not Path(output).exists():
            # Run the docking simulation
            with subprocess.Popen("./QuickVina2-GPU" +
                                  ' --config ' + config +
                                  ' --log ' + log,
                                  # ' > /dev/null 2>&1',
                                  stdout=subprocess.PIPE,
                                  cwd=vina_cwd,
                                  shell=True, start_new_session=True) as proc:
                try:
                    out = proc.communicate(timeout=timeout_duration)
                    out_str = str(out[0]).strip().replace(r"\n", "\n")
                    if out_str is not None and "CL_OUT_OF_HOST_MEMORY" in out_str:
                        print("Docking: OUT OF GPU MEMORY ERROR")
                    if proc.returncode != 0:
                        print(f"Docking error: code {proc.returncode}")
                    #proc.wait(timeout=timeout_duration)
                except subprocess.TimeoutExpired:
                    p = psutil.Process(proc.pid)
                    p.terminate()
        else:
            print("ALREADY EXISTS!")

        # Parse the docking score
        if Path(output).exists():
            score = 1000000.0
            with open(output, 'r') as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                        score = min(score, float(new_score))
                result = score

        else:
            result = 1000000.0
    else:
        raise Exception(f'Protein file: {receptor_input!r} not found!')

    return result

def calculateDockingScore(smi, protein_file, lig_dir, out_dir, log_dir, conf_dir, vina_cwd):
    # creating appropriate file names for ligands

    if Chem.MolFromSmiles(smi) is not None:
        ligand_name = smi.replace('(', '{').replace(')', '}').replace('#', '-').replace('/', '.').replace('\\', '.')
        return dock(
            protein_file,
            Chem.MolToSmiles(Chem.MolFromSmiles(smi)),
            ligand_name,
            lig_dir=lig_dir,
            out_dir=out_dir,
            log_dir=log_dir,
            conf_dir=conf_dir,
            vina_cwd=vina_cwd,
        )
    else:
        return 1000000.0
