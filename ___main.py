from __data_scout import data_scout
from __data_worker import DataSeekProcess
from __model_builder import MachineBuilder
from __model_run import RunModel

def demo_datascout():
    data_scout("Tau", 20)

def demo_dataseekprocess():
    DataSeekProcess("Tau", 42, "PubChem").run()

def demo_machinebuilder():
    demo = MachineBuilder("Bioactivity_Dataset_pIC50_PubChem_fp.csv", 'kat_is_smart')
    demo.train_assess_model()

def demo_runmodel():
    test = RunModel("kat_is_smart", "test_smile", "PubChem")
    test.run_predictions()

demo_runmodel()
