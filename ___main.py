from __DataScout import data_scout
from __DataSeekProcess import DataSeekProcess
from __ModelBuilder import ModelBuilder
from __RunModel import RunModel

def demo_datascout():
    data_scout("Tau", 20)

def demo_dataseekprocess():
    DataSeekProcess("Tau", 42, "PubChem")

def demo_machinebuilder():
    demo = ModelBuilder("Bioactivity_Dataset_pIC50_PubChem_fp.csv", 'github_test_model')
    demo.train_assess_model()

def demo_runmodel():
    test = RunModel("kat_is_smart", "test_smile", "PubChem")
    test.run_predictions()
