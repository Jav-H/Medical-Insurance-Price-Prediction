import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from feature_engine.outliers import ArbitraryOutlierCapper
from sklearn.linear_model import LinearRegression, Lasso  # Basic prediction models
from sklearn.svm import SVR  # More flexible Regression Model
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor  # Ensemble models for better predction
from xgboost import XGBRegressor  # Advanced ensemble model with high efficiency
from sklearn.model_selection import train_test_split  # Splits the dataset
from sklearn.model_selection import cross_val_score  # Evaluates model reliability
import warnings
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")


# Version 1.0 March 24. Added a function to import a csv, show an error message,plot a pie chart.
# Things to add, the bar charts, scatter plots, and more in depth details for the regression models
# QoL updates to the GUI.


# I might move some of the buttons later we will see.
def view_dataframe():
    '''
    This function allows the user to view their imported csv file as a raw pandas dataframe in a new window
    '''

    # this function still needs worked on. Need to implement a
    # scrollable frame in this function so you can scroll to the end of the df

    dataframe_window = tk.Toplevel(scrollable_frame)
    dataframe_window.title("Imported CSV as Dataframe")
    dataframe_label = ttk.Label(dataframe_window, text=df.to_string(), padding=10)
    dataframe_label.pack(pady=5)
    close_button = ttk.Button(dataframe_window, text="Close", command=dataframe_window.destroy)
    close_button.pack(pady=5)


def import_csv():
    """
    Load the csv into a pandas DF and calls the various graphing and machine algorithmic functions
    """
    filepath = filedialog.askopenfilename(title="Please select a CSV file.",
                                          # create the filepath and allow for CSV files
                                          filetypes=(("CSV files", "*.csv"),
                                                     ("All files", "*.*")))
    if filepath:  # test if the filepath exists
        try:
            global df  # create a global dataframe that can be used by every function without needing it to be passed in
            df = pd.read_csv(filepath)
            pie_chart_label = ttk.Label(scrollable_frame, text="Pie Chart Analysis",
                                        font=("Arial", 12, "bold"))  # neat label
            pie_chart_label.pack(pady=5)  # add 5 pixels vertically to add space
            view_dataframe_button = ttk.Button(scrollable_frame, text="View Dataframe", command=view_dataframe)
            view_dataframe_button.pack(pady=10)
            view_metrics_button = ttk.Button(scrollable_frame, text="View Metrics", command=display_metrics)
            view_metrics_button.pack(pady=10)
            plot_pie_chart()
            pie_label = ttk.Label(scrollable_frame,
                                  text="Left: Pictured are the percentages of the different sexes, males colored in blue and females colored "
                                       "in orange  from this analysis, we can see that the distribution is almost equal.")
            pie_label.pack(pady=5)
            pie_label2 = ttk.Label(scrollable_frame,
                                   text="Middle: This pie chart displays the percentages of individuals who smoke and those who do not. The blue section of the chart signifies the percentage of individuals who do not smoke, while the orange section signifies the individuals who do. As shown in the pie chart, we can see that the distribution is far from equal, with 79.5% of the sample not smoking, while 20.5% of users do smoke.",
                                   wraplength=400, justify="left")
            pie_label2.pack(pady=5)
            pie_label3 = ttk.Label(scrollable_frame,
                                   text="Right:This pie chart pictures the different percentages of the regions where the individuals from the sample are located, with red signifying the percentage of individuals who live in the Northeast, blue signifying those who live in the Southeast, orange signifying those who live in the Southwest, and green signifying the percentage of individuals who live in the Northwest. These percentages are approximately equal, all of which are around 25%.",
                                   wraplength=400, justify="left")
            pie_label3.pack(pady=5)

            bar_graph_label = ttk.Label(scrollable_frame, text="Bar Graph Analysis", font=("Arial", 12, "bold"))
            bar_graph_label.pack(pady=5)
            plot_bar_graphs()

            bar_label = ttk.Label(scrollable_frame,
                                  text="Top Left: This bar graph exhibits and demonstrates the average charges when separated by sex. The bars of the chart are relatively equal, meaning that gender differences have little to no effect on insurance premiums.",
                                  wraplength=400, justify="left")
            bar_label.pack(pady=5)
            bar_label1 = ttk.Label(scrollable_frame,
                                   text="Top Right: The average charges by children bar graph visualizes the average charges for individuals with different numbers of children, ranging from no children to five children. While there is a difference between some of the bars, the majority of them are similar and loosely mimic the normal curve, meaning that the number of children does have a small effect on insurance premiums",
                                   wraplength=400, justify="left")
            bar_label1.pack(pady=5)
            bar_label2 = ttk.Label(scrollable_frame,
                                   text="Bottom Left: This bar plot pictures the average charges when dividing the individuals by smoking status  as shown, the average charges are significantly higher for smokers compared to those of non-smokers. In other words, smoking status significantly affects the price of the insurance premium.",
                                   wraplength=400, justify="left")
            bar_label2.pack(pady=5)
            bar_label3 = ttk.Label(scrollable_frame,
                                   text="Bottom Right: This plot displays the average insurance premiums separating the individuals by the southeast, southwest, northeast, and northwest regions. The bars of this graph are relatively close, however, there are variations between them, signaling that there is some effect regionality has on insurance premiums. ",
                                   wraplength=400, justify="left")
            bar_label3.pack(pady=5)

            scatter_plot_label = ttk.Label(scrollable_frame, text="Scatter Plot Analysis", font=("Arial", 12, "bold"))
            scatter_plot_label.pack(pady=5)
            plot_scatter_plots()

            scatter_label = ttk.Label(scrollable_frame,
                                      text="Left: This scatterplot displays the charges of individuals based on age, while highlighting the individuals who smoke with the color blue. This shows that among all ages, smokers are charged far higher than non-smokers. Not to mention that charges increase as the individual grows older. ",
                                      wraplength=400, justify="left")
            scatter_label.pack(pady=5)
            scatter_label1 = ttk.Label(scrollable_frame,
                                       text="Right: This scatterplot displays the charges of each individual based on BMI, but also shows which individuals are also smokers. Across all different BMIs, smokers tend to pay higher premium insurance rates than those who dont. Overall, showing the significant effects smoker status has on premium rates.",
                                       wraplength=400, justify="left")
            scatter_label1.pack(pady=5)
            outlier_label = ttk.Label(scrollable_frame, text="Outlier Plots and Handling", font=("Arial", 12, "bold"))
            outlier_label.pack(pady=5)
            handle_outliers()
            outlier_label1 = ttk.Label(scrollable_frame,
                                       text="Left: The Age Boxplot shows the range, median, and the fact that there are no outliers within the sample. Due to the fact that the Age characteristic has no outliers, we can deduce that the data regarding charges for age are not skewed, distorting the statistics and misrepresenting the data.",
                                       wraplength=400, justify="left")
            outlier_label1.pack(pady=5)
            outlier_label2 = ttk.Label(scrollable_frame,
                                       text="Right: The BMI Boxplot, outliers show that there are outlying BMI values, which can misrepresent the data and individuals, so as a solution we clean the data of these outliers.",
                                       wraplength=400, justify="left")
            outlier_label2.pack(pady=5)
            plot_new_bmi()
            outlier_label2 = ttk.Label(scrollable_frame,
                                       text="Bottom: After cleaning the data and removing the outliers of the BMI, this boxplot displays the minimum, range, median, and maximum of the BMI values.",
                                       wraplength=400, justify="left")
            outlier_label2.pack(pady=5)
            welcome_label.config(text=f"'{filepath}' successfully imported")
            encoding_button = ttk.Button(scrollable_frame, text="View Encoded Data", command=encode_discrete_data)
            encoding_button.pack(pady=10)
            model_button_frame = ttk.Frame(scrollable_frame)
            model_button_frame.pack(pady=10)

            # Encoding for discrete categorial data [sex, bmi, region]
            df['sex'] = df['sex'].map({'male': 0, 'female': 1})
            df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
            df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
            X = df.drop(['charges'], axis=1)
            Y = df[['charges']]
            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                            random_state=42)  # 80:20 Train-Test Split
            lrmodel = LinearRegression()
            lrmodel.fit(xtrain, ytrain)
            print("Linear Regression:")
            LR1 = lrmodel.score(xtrain, ytrain)
            LR2 = (lrmodel.score(xtest, ytest))
            LR3 = (cross_val_score(lrmodel, X, Y, cv=5, ).mean())
            LR_label = ttk.Label(scrollable_frame, text="Linear Regression Analysis", font=("Arial", 12, "bold"))
            LR_label.pack(pady=5)

            LR_label1 = ttk.Label(scrollable_frame, text=str(LR1), padding=5)
            LR_label1.pack(pady=5)
            LR_label2 = ttk.Label(scrollable_frame, text=str(LR2), padding=5)
            LR_label2.pack(pady=5)
            LR_label3 = ttk.Label(scrollable_frame, text=str(LR3), padding=5)
            LR_label3.pack(pady=5)

            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                            random_state=42)  # 80:20 Train-Test Split
            svrmodel = SVR()
            svrmodel.fit(xtrain, ytrain)
            ypredtrain1 = svrmodel.predict(xtrain)
            ypredtest1 = svrmodel.predict(xtest)

            SVR_value1 = r2_score(ytrain, ypredtrain1)
            SVR_value2 = r2_score(ytest, ypredtest1)
            SVR_value3 = cross_val_score(svrmodel, X, Y, cv=5, ).mean()

            svr_label = ttk.Label(scrollable_frame, text='SVR Analysis', font=("Arial", 12, "bold"))
            svr_label.pack(pady=10)
            SVR_label1 = ttk.Label(scrollable_frame, text=str(SVR_value1), padding=5)
            SVR_label1.pack(pady=5)
            SVR_label2 = ttk.Label(scrollable_frame, text=str(SVR_value2), padding=5)
            SVR_label2.pack(pady=5)
            SVR_label3 = ttk.Label(scrollable_frame, text=str(SVR_value3), padding=5)
            SVR_label3.pack(pady=5)

            xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.2,
                                                            random_state=42)  # 80:20 Train-Test Split
            rfmodel = RandomForestRegressor(random_state=42)
            rfmodel.fit(xtrain, ytrain)
            ypredtrain2 = rfmodel.predict(xtrain)
            ypredtest2 = rfmodel.predict(xtest)
            RFR_value1 = r2_score(ytrain, ypredtrain2)
            RFR_value2 = r2_score(ytest, ypredtest2)
            RFR_value3 = cross_val_score(rfmodel, X, Y, cv=5, ).mean()

            RFR_label = ttk.Label(scrollable_frame, text='RFR Analysis', font=("Arial", 12, "bold"))
            RFR_label.pack(pady=10)
            RFR_label1 = ttk.Label(scrollable_frame, text=str(RFR_value1), padding=5)
            RFR_label1.pack(pady=5)
            RFR_label2 = ttk.Label(scrollable_frame, text=str(RFR_value2), padding=5)
            RFR_label2.pack(pady=5)
            RFR_label3 = ttk.Label(scrollable_frame, text=str(RFR_value3), padding=5)
            RFR_label3.pack(pady=5)

            xgmodel = XGBRegressor()
            xgmodel.fit(xtrain, ytrain)
            ypredtrain4 = xgmodel.predict(xtrain)
            ypredtest4 = xgmodel.predict(xtest)
            XGB_value1 = r2_score(ytrain, ypredtrain4)
            XGB_value2 = r2_score(ytest, ypredtest4)
            XGB_value3 = cross_val_score(xgmodel, X, Y, cv=5, ).mean()
            from sklearn.model_selection import GridSearchCV
            estimator = XGBRegressor()
            param_grid = {'n_estimators': [10, 15, 20, 40, 50], 'max_depth': [3, 4, 5], 'gamma': [0, 0.15, 0.3, 0.5, 1]}
            grid = GridSearchCV(estimator, param_grid, scoring="r2", cv=5)
            grid.fit(xtrain, ytrain)
            XGB_value4 = grid.best_params_
            xgmodel = XGBRegressor(n_estimators=15, max_depth=3, gamma=0)
            xgmodel.fit(xtrain, ytrain)
            ypredtrain4 = xgmodel.predict(xtrain)
            ypredtest4 = xgmodel.predict(xtest)
            XGB_value5 = r2_score(ytrain, ypredtrain4)
            XGB_value6 = r2_score(ytest, ypredtest4)
            XGB_value7 = cross_val_score(xgmodel, X, Y, cv=5, ).mean()

            XGB_label = ttk.Label(scrollable_frame, text='XGBoost Analysis', font=("Arial", 12, "bold"))
            XGB_label.pack(pady=10)
            XGB_label1 = ttk.Label(scrollable_frame, text=str(XGB_value1), padding=5)
            XGB_label1.pack(pady=5)
            XGB_label2 = ttk.Label(scrollable_frame, text=str(XGB_value2), padding=5)
            XGB_label2.pack(pady=5)
            XGB_label3 = ttk.Label(scrollable_frame, text=str(XGB_value3), padding=5)
            XGB_label3.pack(pady=5)
            XGB_label4 = ttk.Label(scrollable_frame, text=str(XGB_value1), padding=5)
            XGB_label4.pack(pady=5)
            XGB_label5 = ttk.Label(scrollable_frame, text=str(XGB_value2), padding=5)
            XGB_label5.pack(pady=5)
            XGB_label6 = ttk.Label(scrollable_frame, text=str(XGB_value3), padding=5)
            XGB_label6.pack(pady=5)
            XGB_label7 = ttk.Label(scrollable_frame, text=str(XGB_value3), padding=5)
            XGB_label7.pack(pady=5)

            feats = pd.DataFrame(data=grid.best_estimator_.feature_importances_, index=X.columns,
                                 columns=['Importance'])
            important_features = feats[feats['Importance'] > 0.01]
            important_features
            df.drop(df[['sex', 'region']], axis=1, inplace=True)
            Xf = df.drop(df[['charges']], axis=1)
            X = df.drop(df[['charges']], axis=1)
            xtrain, xtest, ytrain, ytest = train_test_split(Xf, Y, test_size=0.2, random_state=42)
            finalmodel = XGBRegressor(n_estimators=15, max_depth=3, gamma=0)
            finalmodel.fit(xtrain, ytrain)
            ypredtrain4 = finalmodel.predict(xtrain)
            ypredtest4 = finalmodel.predict(xtest)
            print(r2_score(ytrain, ypredtrain4))
            print(r2_score(ytest, ypredtest4))
            print(cross_val_score(finalmodel, X, Y, cv=5, ).mean())
            from pickle import dump
            from tkinter import messagebox
            dump(finalmodel, open('insurancemodelf.pkl', 'wb'))

            root = tk.Tk()
            root.title("Predictor")

            ttk.Label(root, text="Enter Age:").pack(pady=2)
            entry1 = ttk.Entry(root, width=30)
            entry1.pack(pady=5)

            ttk.Label(root, text="Select Sex:").pack(pady=2)
            dropdown_values1 = ["Male", "Female"]
            dropdown1 = ttk.Combobox(root, values=dropdown_values1, width=27)
            dropdown1.pack(pady=5)
            dropdown1.set("Choose...")

            ttk.Label(root, text="Enter BMI:").pack(pady=2)
            entry2 = ttk.Entry(root, width=30)
            entry2.pack(pady=5)

            ttk.Label(root, text="Enter Number of Children:").pack(pady=2)
            entry3 = ttk.Entry(root, width=30)
            entry3.pack(pady=5)

            ttk.Label(root, text="Do you smoke?").pack(pady=2)
            dropdown_values2 = ["Yes", "No"]
            dropdown2 = ttk.Combobox(root, values=dropdown_values2, width=27)
            dropdown2.pack(pady=5)
            dropdown2.set("Choose...")

            ttk.Label(root, text="Select Region:").pack(pady=2)
            dropdown_values3 = ["Northeast", "Northwest", "Southeast", "Southwest"]
            dropdown3 = ttk.Combobox(root, values=dropdown_values3, width=27)
            dropdown3.pack(pady=5)
            dropdown3.set("Choose...")

            prediction_label = ttk.Label(root, text="", justify="left")
            prediction_label.pack()

            def get_input_data():
                "This function gets the inputed data and creates a datraframe."
                global new_data
                global finalmodel
                global prediction_label
                text1 = entry1.get().strip()  # age
                dropdown_value1 = dropdown1.get().strip().lower()  # sex
                text2 = entry2.get().strip()  # bmi
                text3 = entry3.get().strip()  # children
                dropdown_value2 = dropdown2.get().strip().lower()  # smoker status
                dropdown_value3 = dropdown3.get().strip().lower()  # region

                if not text1 or not text2 or not text2 or dropdown_value1 == "Choose..." or dropdown_value2 == "Choose..." or dropdown_value3 == "Choose...":
                    messagebox.showerror("Error!", "All fields must be filled!")
                    return

                new_data = pd.DataFrame(
                    {'age': text1, 'sex': dropdown_value1, 'bmi': text2, 'children': text3, 'smoker': dropdown_value2,
                     'region': dropdown_value3},
                    index=[0])
                new_data['smoker'] = new_data['smoker'].map({'yes': 1, 'no': 0})
                new_data['age'] = pd.to_numeric(new_data['age'], errors='coerce')
                new_data['bmi'] = pd.to_numeric(new_data['bmi'], errors='coerce')
                new_data['children'] = pd.to_numeric(new_data['children'], errors='coerce')
                new_data = new_data.drop(columns=['sex', 'region'])

                return new_data

            def make_prediction(data):
                """This function makes a prediction given the new data."""
                if data is None or finalmodel is None:
                    return None
                try:
                    prediction = finalmodel.predict(data)[0]
                    return prediction
                except Exception as e:
                    messagebox.showerror("Error!", f"Unexpected error: {e}")
                    return None

            def update_prediction_label():
                """This function updates label with the predicted insurance charge."""
                data = get_input_data()  # Get user input
                prediction = make_prediction(data)  # Make prediction

                if prediction is not None:
                    prediction_label.config(text=f"Predicted Insurance Premium: ${prediction:,.2f}")

            ok_button = ttk.Button(root, text="OK", command=update_prediction_label)
            ok_button.pack(pady=10)

        except FileNotFoundError:
            show_error_message(f"the CSV file: '{filepath}' was not found. Does it exist?")
        except pd.errors.EmptyDataError:
            show_error_message(f"The CSV file: '{filepath}' is empty.")
        except ValueError:
            show_error_message(f"The file: '{filepath}' is not a csv file.")
        except Exception as error:
            show_error_message(f"The following error occured: '{error}")


def show_error_message(message):
    """
    This function creates a window on top of the application window in order to display the error message.
    It also creates a button the user can click on after reading the message and then destroys the window.
    param:
    message: the error message generated by the program
    """
    error_window = tk.Toplevel(window)
    error_window.title("Error")
    error_label = ttk.Label(error_window, text=message, padding=20)
    error_label.pack()
    ok_button = ttk.Button(error_window, text="OK", command=error_window.destroy)
    ok_button.pack(
        pady=10)  # pady adds 10 pixels of space around the button. pack organizes button and labels in the parent widget (window)


def plot_pie_chart():
    """
    This function creates a pie chart that is then displayed on the main window. Below is code from the Geeks Website.
    """
    features = ['sex', 'smoker', 'region']
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, col in enumerate(features):
        x = df[col].value_counts()
        axes[i].pie(x.values, labels=x.index, autopct='%1.1f%%', textprops={'fontsize': 8})
        axes[i].set_title(col, fontsize=10)
    canvas_pie = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_pie.draw()
    canvas_pie.get_tk_widget().pack(pady=10)


def plot_bar_graphs():
    """
    This function will generate a bar graph with information from the dataframe and then display it.
    Subject to change.
    """
    features = ['sex', 'children', 'smoker', 'region']
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, col in enumerate(features):
        df.groupby(col)['charges'].mean().astype(float).plot.bar(ax=axes[i])
        axes[i].set_title(f'Average Charges by {col}')
    fig.tight_layout(pad=3.0)
    canvas_bargraph = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_bargraph.draw()
    canvas_bargraph.get_tk_widget().pack(pady=10)


def plot_scatter_plots():
    """
    This function will generate a scatter plot of information and display it.
    Subject to change.
    """
    import seaborn as sns
    features = ['age', 'bmi']
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, col in enumerate(features):
        sns.scatterplot(data=df, x=col,
                        y='charges',
                        hue='smoker',
                        ax=axes[i])
        axes[i].set_title(f'{col} vs Charges')
    canvas_scatterplot = FigureCanvasTkAgg(fig, master=scrollable_frame)
    canvas_scatterplot.draw()
    canvas_scatterplot.get_tk_widget().pack(pady=10)


def handle_outliers():
    '''
    This function will handle and display the outliers in the given dataset.
    '''
    df.drop_duplicates(inplace=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plt.sca(axes[0])  # puts the first graph on the left
    plt.boxplot(df['age'])
    plt.title('Age Boxplot, no outliers')
    plt.sca(axes[1])  # puts the second one on the right
    plt.boxplot(df['bmi'])
    plt.title('BMI Biplot, has outliers')
    outlier_botplots = FigureCanvasTkAgg(fig, master=scrollable_frame)
    outlier_botplots.draw()
    outlier_botplots.get_tk_widget().pack(pady=10)


def display_metrics():
    '''
    This function will calculate the means and limits of the CSV dataset.
    :return:
    '''
    Q1 = df['bmi'].quantile(0.25)
    Q2 = df['bmi'].quantile(0.5)
    Q3 = df['bmi'].quantile(0.75)
    iqr = Q3 - Q1  # inner quartile range
    lower_limit = Q1 - (1.5 * iqr)
    upper_limit = Q3 + (1.5 * iqr)
    metric_window = tk.Toplevel(scrollable_frame)
    metric_window.title("Upper and Lower Limits")
    lower_limit_label = ttk.Label(metric_window, text=f'The lower limit is: {lower_limit}', padding=10)
    lower_limit_label.pack(pady=5)
    upper_limit_label = ttk.Label(metric_window, text=f'The upper limit is: {upper_limit}', padding=10)
    upper_limit_label.pack(pady=5)


def plot_new_bmi():
    Q1 = df['bmi'].quantile(0.25)
    Q2 = df['bmi'].quantile(0.5)
    Q3 = df['bmi'].quantile(0.75)
    iqr = Q3 - Q1  # inner quartile range
    lower_limit = Q1 - (1.5 * iqr)
    upper_limit = Q3 + (1.5 * iqr)
    arb = ArbitraryOutlierCapper(min_capping_dict={'bmi': lower_limit}, max_capping_dict={'bmi': upper_limit})
    df[['bmi']] = arb.fit_transform(df[['bmi']])
    fig, axes = plt.subplots(1, figsize=(15, 6))
    plt.boxplot(df['bmi'])
    plt.title('BMI Boxplot No Outlier')
    BMI_plot = FigureCanvasTkAgg(fig, master=scrollable_frame)
    BMI_plot.draw()
    BMI_plot.get_tk_widget().pack(pady=10)


def encode_discrete_data():
    '''
    This function will encode and display the discrete categorial data [sex,bmi,region]
    :return:
    '''
    # Encoding for discrete categorial data [sex, bmi, region]
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
    df['region'] = df['region'].map({'northwest': 0, 'northeast': 1, 'southeast': 2, 'southwest': 3})
    encoding_window = tk.Toplevel(scrollable_frame)
    encoding_window.title("Encoding Discrete Data")
    encoding_label = ttk.Label(encoding_window, text=df.corr(), padding=10)
    encoding_label.pack(pady=10)
    # PROBLEM 3 >> NaN's appearing when it should be mapping the numbers accordingly
    # Solution >> Don't round up the Skew of the 'bmi' and 'age' / Run All


def scroll_region(event):
    """
    Configures the scroll region of the canvas.
    """
    canvas.configure(scrollregion=canvas.bbox("all"))

if __name__ == "__main__":
    window = tk.Tk()
    window.title("Medical Insurance Cost Predictor")
    window.geometry("1000x600")

    welcome_label = ttk.Label(window, text="Welcome to the Medical Insurance Cost Predictor. Enter a CSV file to begin.")
    welcome_label.pack(pady=10)

    csv_button = tk.Button(window, text="Import CSV File", command=import_csv)
    csv_button.pack(pady=10)

    canvas = tk.Canvas(window)
    scrollbar = tk.Scrollbar(window, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind("<Configure>", scroll_region)
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")
    
    window.mainloop()

    sys.exit() #end process