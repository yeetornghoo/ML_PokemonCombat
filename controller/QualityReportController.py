import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def run(data):
    #validate_null_data(data)
    #validate_heat_map(data)
    validate_pair_plot(data)
    #validate_quality_report(data)


# COUNT NUMBER OF NULL COLUMN AND GENERATE REPORT TO
def validate_null_data(data):
    try:
        f = open("output/validation/null_report.txt", "w+")
        for label, content in data.items():
            number_null=0
            column_data = data[label].isna()
            for i in column_data:
                if i:
                    number_null += 1
            output_str = "{} : {}".format(label, number_null)
            f.write(output_str+"\n")
    finally:
        f.close()


def validate_heat_map(data):
    corr_mat = data.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(14, 15))
    cbar_kws = {'orientation': "horizontal", 'pad': 0.05, 'aspect': 50}
    sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax, cmap="YlGnBu", fmt='.2f', linewidths=.3)
    plt.savefig("output/validation/heatmap_report.png", dpi=500)


def validate_pair_plot(data):
    sns.pairplot(data[['attack_force', 'defense_point', 'special_attack_force', 'special_defense_point',
                       'generation', 'legendary', 'height_m', 'weight_kg',
                       'against_ghost']])
    plt.savefig("output/validation/pairplot_report.png", dpi=500)


"""
def validate_quality_report(data):
    print(data.columns.unique())
   
    df = pd.DataFrame(data.columns)
    #df['count'] = data.shape[0]

    for col in range(len(df)):
        print(df[col])
        #dict = {col:[data.mean().get(col)]}
        #df = pd.DataFrame(dict)

    #miss
    #card
    min = data.min()
    firstQrt = data.quantile(.25)
    mean = data.mean()
    median = data.median()
    thirdQrt = data.quantile(.75)
    max = data.max()
    StandardDec = data.std()
    #print(median.get("against_bug"))
"""


