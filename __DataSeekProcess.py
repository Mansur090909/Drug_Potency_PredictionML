from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from padelpy import padeldescriptor
from pathlib import Path


class DataSeekProcess:
    """
    This class will search the chembl website for target protein data, and then investigate a selected target
    \n After data cleaning, removing redundant info, and obtaining a fingerprint csv, it will generate a dataframe csv containing each compound's fingerprint and their pIC50 values to train the machine learning model
    """
    def __init__(self, target_protein, selected_target_index, fingerprint_set):
        self.target_subject = target_protein
        self.target_idx = selected_target_index
        self.fingerprint = fingerprint_set
        self.bioact_folder = Path('Bioactivity_Data_Folder')

    def run(self):
        print("Starting...")

        self.preprocess_1()
        self.process_2()
        self.smile_fingerprinter_3()

        print("Data Processing Complete")


    def preprocess_1(self):
        """Search CHEMBL and obtain data - preprocess, cleaning and saving csv file containing chembl ids, canonical smiles and standard values"""
        target = new_client.target
        target_query = target.search(self.target_subject)
        targets = pd.DataFrame.from_dict(target_query)

        selected_target = targets.target_chembl_id[self.target_idx]
        print("Processing Data on Selected Target:", selected_target)

        # obtain bioactivity data reported as IC50 values in nM
        activity = new_client.activity
        result = activity.filter(target_chembl_id=selected_target).filter(standard_type = "IC50")
        raw_df = pd.DataFrame.from_dict(result)
        raw_df.to_csv(self.bioact_folder / 'raw_bioactivity_data.csv', index=False)

        # eliminate compounds with no standard_value -- standardize our df
        standard_df = raw_df[raw_df.standard_value.notna()]

        # Label compounds either as active, inactive or intermediate
        bioactivity_class = []
        for i in standard_df.standard_value:
            if float(i) >= 10000:
                bioactivity_class.append("inactive")
            elif float(i) <= 1000:
                bioactivity_class.append("active")
            else:  # in between 1000 and 10000
                bioactivity_class.append("intermediate")

        # complete pre-processing - we want chembl ids, canonical SMILES and standard values
        selected_columns = ['molecule_chembl_id', 'canonical_smiles', 'standard_value']
        dummy_df = standard_df[selected_columns]
        preprocess_df = dummy_df.copy()
        preprocess_df['bioactivity_class'] = bioactivity_class
        preprocess_df.to_csv(self.bioact_folder / "preprocessed_bioact_data.csv")

    def process_2(self):
        """produce dataframe containing lipinski information, normalized and pIC50 values"""
        preprocess_df = pd.read_csv(self.bioact_folder / "preprocessed_bioact_data.csv")

        # generate lipinski dataframe
        lipinski_df = self.lipinski_info(preprocess_df.canonical_smiles)

        dummy_df = pd.concat([preprocess_df, lipinski_df], axis=1)

        # normalize values in dataframe
        norm_df = self.norm_value(dummy_df)

        # calculate pIC50 values from IC50 -> replace those values in dataframe
        # this dataframe now has lipinski, normalized, and pIC50 data
        lip_pic_norm_df = self.pIC50(norm_df)

        process2_df = lip_pic_norm_df[lip_pic_norm_df.bioactivity_class != 'intermediate']
        process2_df.to_csv(self.bioact_folder / "bioactivity_proc_dataclass.csv")

    def smile_fingerprinter_3(self):
        """Generate molecule smiles and fingerprint file of all filtered compounds for the model to process"""
        prefingerprint_df = pd.read_csv(self.bioact_folder / "bioactivity_proc_dataclass.csv")
        selection = ['canonical_smiles', 'molecule_chembl_id']
        selected_df = prefingerprint_df[selection]

        # create smiles file for padel to process into a fingerprint csv
        selected_df.to_csv(self.bioact_folder / "molecules.smi", sep='\t', index=False, header=False)

        fp = {
            'AtomPairs2D': 'AtomPairs2DFingerprinter.xml',
            'AtomPairs2DCount': 'AtomPairs2DFingerprintCount.xml',
            'CDK': 'Fingerprinter.xml',
            'CDKextended': 'ExtendedFingerprinter.xml',
            'CDKgraphonly': 'GraphOnlyFingerprinter.xml',
            'EState': 'EStateFingerprinter.xml',
            'KlekotaRoth': 'KlekotaRothFingerprinter.xml',
            'KlekotaRothCount': 'KlekotaRothFingerprintCount.xml',
            'MACCS': 'MACCSFingerprinter.xml',
            'PubChem': 'PubchemFingerprinter.xml',
            'Substructure': 'SubstructureFingerprinter.xml',
            'SubstructureCount': 'SubstructureFingerprintCount.xml'
        }  # dictionary containing descriptor-obtaining methods as key and xml filename as value

        xml_path = Path("padel_fp_xmls")
        fingerprint_output_file = ''.join([self.fingerprint, '.csv'])
        fingerprint_descriptortypes = fp[self.fingerprint]

        padeldescriptor(mol_dir=self.bioact_folder / 'molecules.smi',
                        d_file=self.bioact_folder / fingerprint_output_file,
                        descriptortypes=xml_path / fingerprint_descriptortypes,
                        detectaromaticity=True,
                        standardizenitro=True,
                        standardizetautomers=True,
                        threads=2,
                        removesalt=True,
                        log=True,
                        fingerprints=True)

        # X and Y dataframes to be combined
        fp_X = pd.read_csv(self.bioact_folder / fingerprint_output_file).drop(columns=['Name'])
        fp_Y = prefingerprint_df['pIC50']


        fingerprint_df = pd.concat([fp_X, fp_Y], axis=1)
        fingerprint_df.to_csv(self.bioact_folder / f"Bioactivity_Dataset_pIC50_{self.fingerprint}_fp.csv")


    @staticmethod  # ignore the errors here pls
    def lipinski_info(smiles, verbose=False):
        """
            Generates dataframe containing relevant information on lipinski rules
            \n mol. weight, octanol water part., H-bond donors and acceptors
            """
        moldata = []
        for elem in smiles:
            mol = Chem.MolFromSmiles(elem)
            moldata.append(mol)

        baseData = np.arange(1, 1)
        i = 0
        for mol in moldata:   # ignore the errors, the identifiers are there idk what's up
            desc_MolWt = Descriptors.MolWt(mol)
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_NumHDonors = Lipinski.NumHDonors(mol)
            desc_NumHAcceptors = Lipinski.NumHAcceptors(mol)

            row = np.array([desc_MolWt,
                            desc_MolLogP,
                            desc_NumHDonors,
                            desc_NumHAcceptors])

            if i == 0:
                baseData = row
            else:
                baseData = np.vstack([baseData, row])
            i += 1

        columnNames = ["MW", "LogP", "NumHDonors", "NumHAcceptors"]
        descriptors = pd.DataFrame(data=baseData, columns=columnNames)

        return descriptors

    @staticmethod
    def pIC50(dataframe):
        """
            works with the standard_value_norm column and converts from IC50 to pIC50
            """
        pic50 = []

        for i in dataframe['standard_value_norm']:  # access and iterate through the standard value norm column

            molar = i * (10 ** -9)  # convert nM to M units
            pic50.append(-np.log10(molar))

        dataframe['pIC50'] = pic50
        x = dataframe.drop('standard_value_norm', axis=1)

        return x

    @staticmethod
    def norm_value(dataframe):
        """works with the standard_value_norm column and normalizes all values"""
        norm = []

        for i in dataframe['standard_value']:
            if i > 100000000:
                i = 100000000
            norm.append(i)

        dataframe['standard_value_norm'] = norm
        norm_df = dataframe.drop('standard_value', axis=1)

        return norm_df
