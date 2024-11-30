import re

from tdc.single_pred import ADME, Tox


SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""

caco2 = ADME(name='Caco2_Wang')
pampa = ADME(name='PAMPA_NCATS')
hia = ADME(name='HIA_Hou')
pgp = ADME(name='Pgp_Broccatelli')
bioavail = ADME(name='Bioavailability_Ma')
lipo = ADME(name='Lipophilicity_AstraZeneca')
solu = ADME(name='Solubility_AqSolDB')
freesolv = ADME(name='HydrationFreeEnergy_FreeSolv')
bbb = ADME(name='BBB_Martins')
ppbr = ADME(name='PPBR_AZ')
vdss = ADME(name='VDss_Lombardo')
cyp2c19 = ADME(name='CYP2C19_Veith')
cyp2d6 = ADME(name='CYP2D6_Veith')
cyp3a4 = ADME(name='CYP3A4_Veith')
cyp1a2 = ADME(name='CYP1A2_Veith')
cyp2c9 = ADME(name='CYP2C9_Veith')
cyp2c9_sub = ADME(name='CYP2C9_Substrate_CarbonMangels')
cyp2d6_sub = ADME(name='CYP2D6_Substrate_CarbonMangels')
cyp3a4_sub = ADME(name='CYP3A4_Substrate_CarbonMangels')
halflife = ADME(name='Half_Life_Obach')
clear_hep = ADME(name='Clearance_Hepatocyte_AZ')
clear_micro = ADME(name='Clearance_Microsome_AZ')

ld50 = Tox(name='LD50_Zhu')
herg = Tox(name='hERG')
ames = Tox(name='AMES')
dili = Tox(name='DILI')
skin = Tox(name='Skin Reaction')
carci = Tox(name='Carcinogens_Lagunin')
clintox = Tox(name='ClinTox')

datas = [caco2, pampa, hia, pgp, bioavail, lipo, solu, freesolv, bbb, ppbr, vdss, cyp2c19, cyp2d6, cyp3a4, cyp1a2, cyp2c9, cyp2c9_sub, cyp2d6_sub, cyp3a4_sub, halflife, clear_hep, clear_micro,
         ld50, herg, ames, dili, skin, carci, clintox]

classification = ['PAMPA_NCATS', 'HIA_Hou', 'Pgp_Broccatelli', 'Bioavailability_Ma', 'BBB_Martins', 'CYP2C19_Veith', 'CYP2D6_Veith', 'CYP3A4_Veith', 'CYP1A2_Veith', 'CYP2C9_Veith', 'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'hERG', 'AMES', 'DILI', 'Skin_Reaction', 'Carcinogens_Lagunin', 'ClinTox']
regression = ['Caco2_Wang', 'Lipophilicity_AstraZeneca', 'Solubility_AqSolDB', 'HydrationFreeEnergy_FreeSolv', 'PPBR_AZ', 'VDss_Lombardo', 'Half_Life_Obach', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'LD50_Zhu']
for data in datas:
    file_name = data.name.replace(" ", "_")
    df = data.get_data()
    remove = []
    for i in range(len(df['Drug'])):
        smi = df['Drug'].iloc[i]
        tokenized_smi_len = len(re.findall(SMI_REGEX_PATTERN, smi))
        if tokenized_smi_len > 198:
            remove.append(i)
    if file_name.casefold() in [e.casefold() for e in classification]:
        path = f'data/clf/{file_name}.csv'
    elif file_name.casefold() in [e.casefold() for e in regression]:
        path = f'data/reg/{file_name}.csv'
    else:
        print(file_name)
        raise Exception()
    df = df.rename(columns={'Drug': 'SMILES', 'Y': file_name})
    df.drop(remove).to_csv(path, index=False)