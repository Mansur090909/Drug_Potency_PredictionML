import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from pathlib import Path
import pickle as pkl


class MachineBuilder:
    """
    Builds machine learning model based on data produced by the DataWorker class
    \n cleans the data once the class is initialized, ready for machine learning model generation, assessment and comparison against other models
    """
    def __init__(self, fingerprint_filename, model_name):
        self.model_name = model_name

        self.filepath = Path('Bioactivity_Data_Folder/' + fingerprint_filename)
        self.model_path = Path('ML_Models')
        self.graph_path = Path('Graphs')

        self.model = None
        self.selected_cols = []


    def train_assess_model(self):
        """Buuild a regression model using random forests"""
        # generate the input feature axes
        df = pd.read_csv(self.filepath, index_col = 0)
        X = df.drop('pIC50', axis=1)
        Y = df.pIC50

        # remove low variance features
        selection = VarianceThreshold(threshold=(.8 * (1 - 0.8)))
        selection.fit(X)

        # save the columns that were kept
        kept = selection.get_support()
        self.selected_cols = X.columns[kept].to_list()

        # carry on with the data cleaning
        X = selection.transform(X)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


        # build regression model using random forests
        self.model = RandomForestRegressor(n_estimators=100)
        self.model.fit(X_train, Y_train)
        r2 = self.model.score(X_test, Y_test)
        print("R^2:", r2)

        Y_predict = self.model.predict(X_test)

        sns.set_theme(color_codes=True)
        sns.set_style("white")
        ax = sns.regplot(x=Y_test, y=Y_predict, scatter_kws={'alpha':0.4})
        ax.set_title(f"{self.model_name} ML Regression Assessment")
        ax.set_xlabel('Experimental pIC50', fontsize='large', fontweight='bold')
        ax.set_ylabel('Predicted pIC50', fontsize='large', fontweight='bold')
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
        ax.figure.set_size_inches(5, 5)
        plt.savefig(self.graph_path / f"{self.model_name}_assessment.pdf")

        # Save the selected feature names / settings as dataframe to be used later

        x = True
        while x:
            answer = input("Save Model? y/n: ")
            if answer.lower() == 'y':
                self.save()
                x = False
            elif answer.lower() == 'n':
                print("[Model Rejected]")
                x = False
            else:
                print("invalid character submitted: please input either 'y' or 'n' ")


    def save(self):
        """Save the machine learning model for predictions"""
        print("Saving current model...")

        path = Path(self.model_path / f"{self.model_name}.pkl")
        with open(path, 'wb') as handle:  # write and save model using 'wb' - write binary
            pkl.dump(self.model, handle)

        print("[Model Saved]")

        print("Saving model settings...")
        with open(self.model_path / f"{self.model_name}_settings.txt", 'w') as f:
            for col in self.selected_cols:
                f.write(f"{col}\n")
        print("[Model Settings Saved]")
