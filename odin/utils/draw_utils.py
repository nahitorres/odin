import os

from sklearn.metrics import ConfusionMatrixDisplay
from .env import get_max_val, get_min_val, get_root_logger
import math
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
logger = get_root_logger()

config = {
    "text_size": {"pie": 16, "title": 18, "barh": 16}
}


def pie_plot(values, labels, title, save_path, save, colors=None):
    plt.gcf().set_size_inches([8, 5])
    percentages = np.array(values) / np.sum(values) * 100
    if colors is None:
        patches, _ = plt.pie(values, startangle=90, normalize=True)
        legend_labels = []
        for i, label in enumerate(labels):
            legend_labels.append(str(label) + " - {:.1f} %".format(percentages[i]))
        plt.legend(patches, legend_labels, frameon=True, bbox_to_anchor=(1.05, 0.5), loc='center left', prop={'size': 14})
    else:
        patches, _ = plt.pie(values, colors=colors, startangle=90, normalize=True)
        legend_labels = []
        for i, label in enumerate(labels):
            legend_labels.append(str(label) + " - {:.1f} %".format(percentages[i]))
        plt.legend(patches, legend_labels, frameon=True, bbox_to_anchor=(1.05, 0.5), loc='center left', prop={'size': 14})
    plt.title(title, fontsize=config["text_size"]["title"])
    if save:
        plt.savefig(save_path, facecolor='white', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_class_distribution(dict_to_plot, output_path, save, distribution_type, colors=None):
    try:
        labels = dict_to_plot.keys()
        values = [dict_to_plot[d] / sum(dict_to_plot.values()) for d in dict_to_plot.keys()]
        title = f"{distribution_type} distribution among categories"
        path_to_save = os.path.join(output_path, f'{distribution_type}_cat_distribution.png')
        if colors is None:
            pie_plot(values, labels, title, path_to_save, save)
        else:
            pie_plot(values, labels, title, path_to_save, save, colors=colors)
    except ZeroDivisionError:
        logger.warn(f"No {distribution_type}")


def plot_false_positive_errors(error_values, error_names, category_metric_value, category_name, metric, output_path, save):
    labels = [label[:1].upper() for label in error_names]
    total_errors = [err[1] for err in error_values]
    colors = ['lightskyblue', 'lightyellow', 'orchid', 'coral']
    try:
        percentage_error = [e / sum(total_errors) * 100 for e in total_errors]
        title = "False Positive distribution of category {}".format(category_name)
        path_to_save = os.path.join(output_path, 'false_positive_category_{}-distribution.png'.format(category_name))
        pie_plot(percentage_error, error_names, title, path_to_save, save, colors)
    except:
        logger.warn("No errors found. Unable to plot the errors percentage for category '{}'".format(category_name))

    # Bar Plot
    plt.gcf().set_size_inches([8, 5])
    title = "False Positive impact for category {}".format(category_name)
    performance_values = [(err[0] - category_metric_value) for err in error_values]
    y_pos = np.arange(len(error_names))
    plt.barh(y_pos, performance_values, align='center', color=colors)
    plt.tick_params(labelsize=config["text_size"]["barh"])
    plt.yticks(y_pos, labels)
    plt.title(title, fontsize=config["text_size"]["title"])
    plt.xlabel(metric.replace("_", " ") + " impact", fontdict=dict(size=13))

    path_to_save = os.path.join(output_path, 'false_positive_category_{}-gain.png'.format(category_name))
    if save:
        plt.savefig(path_to_save, facecolor='white', bbox_inches='tight')
    plt.show()
    plt.close()


def plot_ap_bars(results, x, drawline):
    y_todraw = [0 if math.isnan(results[key]["value"]) else results[key]["value"] for key in results]
    y_error = []
    for key in results.keys():
        value = results[key]["std"]
        if value is None or math.isnan(value):
            y_error.append(0)
        else:
            y_error.append(value)
    plt.errorbar(x=x, y=y_todraw, yerr=y_error, ecolor='red', ls='none', lw=1, marker='_', markersize=8,
                 capsize=2)  # , marker='s', ,mec='green', ms=20, mew=4)

    if drawline:  # join all the points
        plt.plot(x, y_todraw, color='cornflowerblue')
    else:
        for i in range(0, len(x), 2):  # join every two points
            plt.plot(x[i: i + 2], y_todraw[i: i + 2], color='cornflowerblue')

    for k in range(len(y_todraw)):  # set the numbers text in bar errors
        plt.text(x[k], y_todraw[k], s=float('{:.2f}'.format(y_todraw[k])), fontsize=13)


def make_multi_category_plot(results, property_name, property_values_names, title_str, metric, save_plot, plot_path):

    xtickstep = 1
    fs = 15
    drawline = True
    plt.clf()
    plt.gcf().set_size_inches([20, 5])
    plt.style.use('seaborn-whitegrid')

    nobj = len(results.keys())
    rangex = np.zeros(1)
    maxy = 1
    xticks = np.asarray([])
    firsttick = np.zeros(nobj)
    for index, key in enumerate(results):
        if not property_name in results[key].keys():
            continue
        result = results[key][property_name]
        nres = len(result)
        if len(property_values_names) != nres:
            raise Exception("Conflict between saved and required results. "
                            "Delete previously saved results: Analyzer.clear_saved_results()")
        space = 0 if index == 0 else 1
        rangex = rangex[-1] + space + np.arange(1, nres + 1)  # expand range x in plot according to the needed lines
        plot_ap_bars(result, rangex, drawline)
        ap = np.asarray([result[key]["value"] for key in result])

        maxy = max(maxy, np.around((np.amax(ap) + 0.15) * 10) / 10)
        ap_cat = results[key]['all']["value"]
        plt.plot([rangex[0], rangex[-1]], np.asarray([1, 1]) * ap_cat, linestyle='dashed', lw=1.5, color='black')
        firsttick[index] = rangex[0]
        xticks = np.concatenate((xticks, rangex[0:rangex.size:xtickstep]))

    maxy = min(maxy, 1)
    axes = plt.gca()
    axes.set_ylim([0, maxy])
    axes.set_xlim([0, rangex[-1] + 1])
    axes.tick_params(axis="x", labelsize=14, labelrotation=-45)
    axes.tick_params(axis="y", labelsize=14)
    
    if len(property_values_names) == nres:  # TODO: CHECK THIS
        property_values_names = np.tile(property_values_names, nobj)
    plt.xticks(xticks, property_values_names)
    
    for index, key in enumerate(results):
        value = key
        if len(key) > 15:
            value = key[:2]
        if key != 'all':
            plt.text(firsttick[index], maxy + 0.02, value, fontsize=fs)

    plt.title(title_str, fontdict={'horizontalalignment': 'center',
                                   'fontsize': 18}, loc='center', pad=30)
    plt.ylabel(metric.replace("_", " "), fontdict=dict(size=14))
    if save_plot:
        plot_path = os.path.join(plot_path, title_str + ".png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def display_sensitivity_impact_plot(results, output_path, property_names, display_names, metric, save):
    valid = [True] * len(property_names)
    maxp = np.zeros((len(results.keys()), len(property_names)))
    minp = np.zeros((len(results.keys()), len(property_names)))

    # For each category, obtain the max and min of each property the different possible values
    for c, category in enumerate(results.keys()):
        for p, prop in enumerate(property_names):
            if prop in results[category].keys():
                maxp[c, p] = get_max_val(results[category][prop], "value", None)  # Max for this category
                minp[c, p] = get_min_val(results[category][prop], "value", None)  # Min for this category
            else:
                valid[p] = False

    # Obtain mean of max/min values among the different categories for each property for the valid categories
    maxval = np.mean(maxp[:, valid], 0)
    minval = np.mean(minp[:, valid], 0)

    property_names = np.array(display_names)[valid]

    # Average of  AP among categories (do not consider the different properties)
    avgval = np.mean([results[cat]["all"]["value"] for cat in results.keys()])
    if math.isnan(avgval):
        avgval = 0

    ##display graph
    plt.style.use('seaborn-whitegrid')
    fs = 12

    # Labels to show
    xticklab = np.asarray(property_names)
    xticklab = xticklab[valid]

    plt.rcParams['figure.figsize'] = (np.sum(valid) + 2, 4.5)  # dynamic width adaptation
    plt.plot([1, len(property_names)], [avgval, avgval], linestyle='dashed', lw=1,
             color='black')
    asymetric_error = [(avgval - minval), (maxval - avgval)]
    plt.errorbar(x=np.arange(1, len(property_names) + 1), y=(avgval * np.ones(len(property_names))),
                 yerr=asymetric_error,
                 ecolor='red',
                 ls='none', lw=1, marker='_', markersize=8, capsize=4, color='red')

    for x in range(len(property_names)):
        plt.text(x + 1.05, minval[x], s='{:.3f}'.format(minval[x]), fontsize=fs - 2)
        plt.text(x + 1.05, maxval[x], s='{:.3f}'.format(maxval[x]), fontsize=fs - 2)
    plt.text(0.6, avgval, s='{:.3f}'.format(avgval), fontsize=fs)

    plt.xticks(np.arange(1, len(property_names) + 1), xticklab)

    axis_x_val = plt.gca().get_xlim()
    plt.gca().set_xlim([0.55, axis_x_val[1]])

    plt.title('Sensitivity and Impact',
              fontdict={'horizontalalignment': 'center',
                        'fontsize': fs}, loc='center', pad=15)
    plt.ylabel(metric.replace("_", " "), fontdict=dict(size=13))
    path_to_save = os.path.join(output_path, 'plots_impact_strong.png')

    if save:
        plt.savefig(path_to_save, facecolor='white', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()


def plot_categories_curve(results, curve, save_plot, output_path):
    fig = go.Figure()
    if not curve == 'f1_curve':
        if curve == 'precision_recall_curve':
            y0 = 1
            y1 = 0
            xaxis_title = "Recall"
            yaxis_title = "Precision"
            title = "Precision-Recall Curve"
        else:  # roc_curve
            y0 = 0
            y1 = 1
            xaxis_title = "False Positive Rate"
            yaxis_title = "True Positive Rate"
            title = "ROC Curve"
        fig.add_shape(
            type='line', line=dict(dash='dash'),
            x0=0, x1=1, y0=y0, y1=y1
        )
    else:
        xaxis_title = "Thresholds"
        yaxis_title = "F1 scores"
        title = "F1 CURVE"

    for category in results:
        auc_score = results[category]['auc']
        x_axis = results[category]['x']
        y_axis = results[category]['y']
        name = f"{category} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, name=name, mode='lines'))
        if len(results) == 1:
            title = title + f" - {category} (AUC={auc_score:.2f})"
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(scaleanchor="x", scaleratio=1, dtick=0.1, fixedrange=False),
        xaxis=dict(constrain='domain', dtick=0.1),
        width=980, height=700,
        title=title,
        legend_title='Categories',
        font=dict(size=17)
    )
    if save_plot:
        plot_path = os.path.join(output_path, f"{curve}.png")
        fig.write_image(plot_path)
    fig.show()


def plot_reliability_diagram(result, save_plot, output_path, is_classification, category=None):
    bin_shift = (result['bins'][1] - result['bins'][0]) / 2
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Confidence Histogram', 'Reliability Diagram'))

    # Confidence histogram
    y1_acc = 1
    y1_con = 1
    if result['avg_value'] < 0.375:
        y1_acc = 0.85
    if result['avg_conf'] < 0.375:
        y1_con = 0.85
    fig.add_shape(type='line', x0=result['avg_value'], x1=result['avg_value'], y0=0, y1=y1_acc, row=1, col=1)
    fig.add_shape(type='line', line=dict(dash='dash'), x0=result['avg_conf'], x1=result['avg_conf'], y0=0, y1=y1_con,
                  row=1, col=1)
    fig.add_trace(go.Bar(x=(result['bins'] + bin_shift), y=result['counts'] / np.sum(result['counts']), name='Samples',
                         showlegend=False, marker=dict(color='rgb(0, 0, 255)')), row=1, col=1)
    fig.add_shape(type='rect', x0=0.025, y0=0.85, x1=0.375, y1=1.0, line=dict(color='RoyalBlue', width=1),
                  fillcolor='rgba(0, 0, 255, 0.1)', row=1, col=1)
    if is_classification:
        text = "accuracy"
    else:
        text = "precision"
    fig.add_trace(go.Scatter(x=[0.15, 0.15], y=[0.95, 0.90],
                             text=[text, "avg confidence"], mode='text', textposition='middle right',
                             showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_shape(type='line', x0=0.05, x1=0.11, y0=0.95, y1=0.95,
                  row=1, col=1)
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0.05, x1=0.11, y0=0.90, y1=0.90,
                  row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], title_text='% Samples', row=1, col=1, dtick=0.1)

    # Reliability diagram
    fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1, row=1, col=2)
    fig.add_shape(type='rect', x0=0.05, y0=0.8, x1=0.35, y1=1.0, line=dict(color='RoyalBlue', width=1),
                  fillcolor='rgba(0, 0, 255, 0.1)', row=1, col=2)
    fig.add_trace(go.Scatter(x=[0.2, 0.2], y=[0.95, 0.85],
                             text=[f"ECE = {result['ece']*100:.2f}", f"MCE = {result['mce']*100:.2f}"], mode='text',
                             showlegend=False, hoverinfo='skip'), row=1, col=2)
    fig.add_trace(go.Bar(x=(result['bins'] + bin_shift), y=result['values'], name='Outputs',
                         marker=dict(color='rgba(0, 0, 255, 0.5)')), row=1, col=2)
    fig.add_trace(go.Bar(x=(result['bins'] + bin_shift), y=result['gaps'], name='Gap',
                         marker=dict(color='rgba(255, 0, 0, 0.5)')), row=1, col=2)
    if is_classification:
        text = "Accuracy"
    else:
        text = "Precision"
    fig.update_yaxes(title_text=text, row=1, col=2, dtick=0.1)

    fig.update_xaxes(title_text='Confidence', range=[0, 1], dtick=0.1)
    if category is None:
        title = "Analysis of the entire dataset"
        plot_path = os.path.join(output_path, f"Reliability_Confidence_analysis.png")
    else:
        title = f"Analysis of category: {category}"
        plot_path = os.path.join(output_path, f"Reliability_Confidence_analysis - {category}.png")

    fig.update_layout(barmode='stack', bargap=0, width=1200, height=600, title_text=title)

    if save_plot:
        fig.write_image(plot_path)
    fig.show()


def display_confusion_matrix(results, categories, properties, save_plot, output_path):
    for index, category in enumerate(categories):
        plt.style.use('default')
        figure, col = plt.subplots(1, 1, figsize=[8, 8])
        disp = ConfusionMatrixDisplay(confusion_matrix=results[index])
        plt.title(f"{category}", weight='bold')
        figure.suptitle(f"Confusion Matrix\nProperties filter: {properties}")
        disp.plot(cmap='Blues', ax=col)
        plt.tight_layout()
        if save_plot:
            plot_path = os.path.join(output_path, f"Confusion_Matrix_{category}.png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=200)
        plt.show()
        plt.close()


def display_confusion_matrix_categories(results, categories, properties, save_plot, output_path):
    plt.style.use('default')
    figure, col = plt.subplots(1, 1, figsize=[16, 16], dpi=100)
    disp = ConfusionMatrixDisplay(confusion_matrix=results, display_labels=categories)
    figure.suptitle(f"Confusion Matrix\nProperties filter: {properties}")
    disp.plot(cmap='Blues', ax=col, xticks_rotation='vertical')
    plt.tight_layout()
    if save_plot:
        plot_path = os.path.join(output_path, f"Confusion_Matrix_categories.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=100)
    plt.show()
    plt.close()
