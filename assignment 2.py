import pandas as pd
import seaborn as sns  # for a heatmap
import matplotlib.pyplot as plt
"""
Function 
bar graph to show data of China and India POpulation
"""

def bar_graph_for_indicators_and_Countires(pivoted_data):
    
    ax = pivoted_data.plot(kind='bar', figsize=(20, 12),fontsize=28)
    ax.set_title('Urban population (% of total population)',fontsize=18)
    ax.set_xlabel('Year', fontsize=28)  # Adjust the fontsize for the x-axis title
    ax.set_ylabel('Value', fontsize=28)
    ax.legend(title='Country Name',  loc='upper right',fontsize=18)
    plt.show()
   
"""
Function 
heat map to show relationship of indicators how GDP and foreign investment factors 
depend on each other

"""
def heatmap_for_indicators_and_Countires(pivoted_data): 
   
    fig, ax = plt.subplots(figsize=(16, 8))
    pivoted_data = pivoted_data.apply(pd.to_numeric, errors='coerce')
    pivoted_data.to_csv('cleaned data.csv')
    font_scale = 1.8  # You can adjust the font size
    
    sns.heatmap(pivoted_data, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.7, ax=ax,vmin=-1, vmax=1)
    sns.set(font_scale=font_scale)

    #ax.set_ylabel('Year', fontsize=14)
    
    #ax.set_xlabel('Country', fontsize=14)
    ax.set_title('Corelation of India Indicators ', fontsize=26)
    plt.show() 
    
"""
Function
line graph to show time line of population increase in China and Inida and 

"""
def linegrpah_for_indicators_and_Countires(pivoted_data): 
    
    for country in pivoted_data.columns:
        plt.plot(pivoted_data.index, pivoted_data[country], marker='o', label=country)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Population Percentage', fontsize=14)
        plt.title('Population, total', fontsize=16)
        plt.legend(title='Country', bbox_to_anchor=(1.05, 1), loc='upper left')
    
"""
main Function to read and transpose data 

"""
def read_and_transpose_data(filename):
    # Read data from World Bank
    read_data = pd.read_csv(filename, skiprows=3)

    # Assuming you want to filter data related to 'Urban population' indicator
    transpose_data = pd.DataFrame(read_data).transpose()
    cleaned_data = transpose_data.dropna(axis=1,thresh=10)
    indicator_col = cleaned_data.loc['Indicator Name']
    countries_col = cleaned_data.loc['Country Name']
   
    
    cleaned_data = cleaned_data.fillna(1)
   
    years = cleaned_data.copy()
    years_column  = years.iloc[4:]
    
    
    
    
   
    return years_column, indicator_col , countries_col , cleaned_data.transpose()

# Example usage
filename = 'API_19_DS2_en_csv_v2_5998250.csv'
years_column,  indicator_col ,countries_col , cleaned_filled_data = read_and_transpose_data(filename)

"""
Function to explore data statistics inclusing describe method , pivot table ,
mean and correlation method 

"""
def explore_statistics(data, indicator_names, countries, graph):
    
    # Filter data for the selected indicators and countries
   
    selected_data = data[data['Indicator Name'].isin(indicator_names) & data['Country Name'].isin(countries)]
  
    pivoted_data = selected_data.pivot_table(index=['Country Name',], values=['1961','1970','1980','1990','2000','2010','2020'] ,aggfunc='mean')
  
    pivoted_data = pivoted_data.transpose()
    
  
    summary_stats = pivoted_data.describe()
    correlation_matrix = pivoted_data.corr()
    mean_values = pivoted_data.mean()
    std_dev_values = pivoted_data.std()
    store_data = pivoted_data.to_csv("cleaned data.csv")
    
    print(pivoted_data)
    print(correlation_matrix)
    print(mean_values)
    print(std_dev_values)
    
   
    if graph == "bar":
        bar_graph_for_indicators_and_Countires(pivoted_data)
    elif graph == "heat":
        heatmap_for_indicators_and_Countires(correlation_matrix)
    elif graph == "line":
        linegrpah_for_indicators_and_Countires(pivoted_data)
    
      

   
    return summary_stats, correlation_matrix, mean_values, std_dev_values, pivoted_data





    
    
  
# Example usage
data = cleaned_filled_data
indicator_names_bar_graph = ['Population, total','Total greenhouse gas emissions (kt of CO2 equivalent)']
countries_bar_graph = ['India','China']

indicator_names_heatmap = ['Foreign direct investment, net inflows (% of GDP)','Droughts, floods, extreme temperatures (% of population, average 1990-2009)','Population, total','Poverty headcount ratio at $2.15 a day (2017 PPP) (% of population)','Forest area (sq. km)','Total greenhouse gas emissions (kt of CO2 equivalent)']
countries_heat_graph = ['India']

indicator_names_line = ['Population, total']
countries_line_graph = ['China','India']


pivoted_data, summary_stats, correlation_matrix, mean_values, std_dev_values = explore_statistics(data, indicator_names_bar_graph, countries_bar_graph,"bar")
pivoted_data, summary_stats, correlation_matrix, mean_values, std_dev_values = explore_statistics(data, indicator_names_heatmap, countries_heat_graph,"heat")
pivoted_data, summary_stats, correlation_matrix, mean_values, std_dev_values = explore_statistics(data, indicator_names_line, countries_line_graph,"line")






def analyze_correlations_over_time(data, indicator_names):
    print(data)
    # Group data by year and calculate correlations for each year
    yearly_correlations = data.groupby(level=0).corr()
    # Calculate mean correlation for each indicator
    mean_correlations = yearly_correlations.groupby(level=1).mean()
    return yearly_correlations, mean_correlations


    print("\nYearly Correlations:")
    print(yearly_correlations)
    print("\nMean Correlations Over Time:")
    print(mean_correlations)


analyze_correlations_over_time(pivoted_data,'Total greenhouse gas emissions (kt of CO2 equivalent)')




