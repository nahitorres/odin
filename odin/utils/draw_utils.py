import os
from collections import defaultdict
from statistics import mean

from .env import get_max_val, get_min_val, get_root_logger
import math
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

from ..classes import Curves

logger = get_root_logger()
config = {
    "text_size": {"pie": 16, "title": 18, "label": 18, "bar_plt": 16, "cm": 16},
    "fig_size": {"pie": (8, 5), "bar_plt": (8, 5), "cm": {"small": (8, 8), "medium": (12, 12), "large": (20, 20)},
                 "multi_cat": {"small": (10, 5), "medium": (15, 5), "large": (20, 5)},
                 "plotly": {"no_legend": (800, 500), "legend": (900, 600)}},
    "dpi": {"show": 80, "save": 200}
}


def pie_plot(values, labels, title, save_path, save, colors=None):
    np.warnings.filterwarnings('ignore')
    percentages = np.array(values) / np.sum(values) * 100
    if np.isnan(percentages).any():
        logger.warning(f"No values for {labels}. Unable to plot {title}.")
        return
    plt.style.use('default')
    plt.figure(figsize=config["fig_size"]["pie"], dpi=config["dpi"]["show"])
    patches, _ = plt.pie(values, colors=colors, startangle=90, normalize=True)
    legend_labels = []
    for i, label in enumerate(labels):
        legend_labels.append(str(label) + " - {:.1f} %".format(percentages[i]))
    plt.legend(patches, legend_labels, frameon=True, bbox_to_anchor=(1.05, 0.5), loc='center left',
               prop={'size': config["text_size"]["pie"]}, ncol=int(math.ceil(len(legend_labels)/10)))
    plt.title(title, fontsize=config["text_size"]["title"])
    if save:
        plt.savefig(save_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_class_distribution(dict_to_plot, labels, output_path, save, title, colors=None):
    try:
        values = [dict_to_plot[d] / sum(dict_to_plot.values()) for d in dict_to_plot.keys()]
        path_to_save = os.path.join(output_path,f'{str(title).replace(" ", "_").replace("/", "_")}_distribution.png')
        pie_plot(values, labels, title, path_to_save, save, colors=colors)
    except ZeroDivisionError:
        logger.warn(f"No values for {title}")


def plot_false_positive_errors(error_values, error_names, category_metric_value, category_name, metric, output_path, save):
    labels = error_names
    total_errors = [err[1] for err in error_values]
    colors = list(reversed(['orange', 'lightskyblue', 'yellow', 'lightgreen', 'red', 'orchid', 'silver'])) if len(total_errors) > 3 else ['yellow', 'silver', 'red']
    try:
        percentage_error = [e / sum(total_errors) * 100 for e in total_errors]
        title = "False Positive distribution of category {}".format(category_name)
        path_to_save = os.path.join(output_path, f"false_positive_category_{str(category_name).replace('/', '_')}"
                                                 f"-distribution.png")
        pie_plot(percentage_error, error_names, title, path_to_save, save, colors)
    except:
        logger.warn("No errors found. Unable to plot the errors percentage for category '{}'".format(category_name))

    # Bar Plot
    plt.style.use('default')
    plt.figure(figsize=config["fig_size"]["bar_plt"], dpi=config["dpi"]["show"])
    title = "False Positive impact for category {}".format(category_name)
    performance_values = [(err[0] - category_metric_value) for err in error_values]
    y_pos = np.arange(len(error_names))
    plt.barh(y_pos, performance_values, align='center', color=colors)
    plt.tick_params(labelsize=config["text_size"]["bar_plt"])
    plt.yticks(y_pos, labels)
    plt.title(title, fontsize=config["text_size"]["title"])
    xlabel = metric.value.replace("_", " ")
    plt.xlabel(xlabel + " impact", fontdict=dict(size=config["text_size"]["bar_plt"]))

    path_to_save = os.path.join(output_path, f"false_positive_category_{str(category_name).replace('/', '_')}-gain.png")
    if save:
        plt.savefig(path_to_save, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["show"])
    plt.show()
    plt.close()


def plot_ap_bars(results, x, drawline, ordered_values=None):
    if ordered_values is None:
        ordered_values = list(results.keys())
    y_todraw = [0 if math.isnan(results[key]["value"]) else results[key]["value"] for key in ordered_values]
    y_error = []
    for key in ordered_values:
        value = results[key]["std"]
        if value is None or math.isnan(value):
            y_error.append(0)
        else:
            y_error.append(value)
    plt.errorbar(x=x, y=y_todraw, yerr=y_error, ecolor='red', ls='none', lw=1, marker='_', markersize=8,
                 capsize=2)

    if drawline:  # join all the points
        plt.plot(x, y_todraw, color='cornflowerblue')
    else:
        for i in range(0, len(x), 2):  # join every two points
            plt.plot(x[i: i + 2], y_todraw[i: i + 2], color='cornflowerblue')

    for k in range(len(y_todraw)):  # set the numbers text in bar errors
        plt.text(x[k], y_todraw[k], s=float('{:.2f}'.format(y_todraw[k])), fontsize=14)


def make_multi_category_plot(results, property_name, property_values_names, categories_display_names, title_str, metric,
                             save_plot, plot_path, property_values=None, split_by="meta-annotations", sort=True, ordered_values_names=None):
    xtickstep = 1
    drawline = True
    plt.style.use('seaborn-whitegrid')

    if sort:
        values_names, unordered_scores = [], defaultdict(list)
        for c in results:
            if property_name not in results[c]:
                return
            for p_v in results[c][property_name]:
                if p_v not in values_names:
                    values_names.append(p_v)
                unordered_scores[p_v].append(results[c][property_name][p_v]['value'])
        scores = [mean(unordered_scores[v]) for v in values_names]

        si = sorted(range(len(scores)), key=lambda k: scores[k], reverse=False)
        property_values_names_tmp, values_names_tmp = [], []
        for i in si:
            property_values_names_tmp.append(property_values_names[i])
            values_names_tmp.append(values_names[i])

        property_values_names = property_values_names_tmp
        ordered_values_names = values_names_tmp
    elif ordered_values_names is None:
        c = list(results.keys())[0]
        ordered_values_names = list(results[c][property_name].keys())

    # check plot size
    nvalues = len(results.keys()) * len(property_values_names)

    if nvalues < 11:
        plt.figure(figsize=config["fig_size"]["multi_cat"]["small"], dpi=config["dpi"]["show"])
    elif nvalues < 16:
        plt.figure(figsize=config["fig_size"]["multi_cat"]["medium"], dpi=config["dpi"]["show"])
    elif nvalues < 31:
        plt.figure(figsize=config["fig_size"]["multi_cat"]["large"], dpi=config["dpi"]["show"])
    else:
        if len(property_values_names) < 2 and split_by == "meta-annotations":
            logger.warning("Too many categories. Changing split_by parameter to 'categories'")
            split_by = "categories"
        if len(results) < 2 and split_by == "categories":
            logger.warning("Too many meta-annotations. Changing split_by parameter to 'meta-annotations'")
            split_by = "meta-annotations'"
        if split_by == "categories":  # split on categories
            res1 = dict(list(results.items())[len(results) // 2:])
            res2 = dict(list(results.items())[:len(results) // 2])
            make_multi_category_plot(res1, property_name, property_values_names, categories_display_names,
                                     title_str + "_1", metric, save_plot,
                                     plot_path, property_values, split_by,
                                     sort=False, ordered_values_names=ordered_values_names)
            make_multi_category_plot(res2, property_name, property_values_names, categories_display_names,
                                     title_str + "_2", metric, save_plot,
                                     plot_path, property_values, split_by,
                                     sort=False, ordered_values_names=ordered_values_names)
        elif split_by == "meta-annotations":  # split on property values
            n_values_half = int(len(property_values_names) / 2)
            make_multi_category_plot(results, property_name, property_values_names[:n_values_half],
                                     categories_display_names, title_str + "_a", metric,
                                     save_plot, plot_path, property_values=ordered_values_names[:n_values_half], split_by=split_by,
                                     sort=False, ordered_values_names=ordered_values_names[:n_values_half])
            make_multi_category_plot(results, property_name, property_values_names[n_values_half:],
                                     categories_display_names, title_str + "_b", metric,
                                     save_plot, plot_path, property_values=ordered_values_names[n_values_half:], split_by=split_by,
                                     sort=False, ordered_values_names=ordered_values_names[n_values_half:])
        else:
            raise Exception("Invalid parameter. Possible 'split_by' parameters: 'categories', 'meta-annotations'")
        return

    nobj = len(results.keys())
    rangex = np.zeros(1)
    maxy = 1
    miny = 0
    xticks = np.asarray([])
    firsttick = np.zeros(nobj)
    for index, key in enumerate(results):
        if property_name not in results[key].keys():
            continue
        if property_values is not None:
            result = {}
            for p_v in results[key][property_name].keys():
                if p_v in property_values:
                    result[p_v] = results[key][property_name][p_v]
        else:
            result = results[key][property_name]
        nres = len(result)
        space = 0 if index == 0 else 1
        rangex = rangex[-1] + space + np.arange(1, nres + 1)  # expand range x in plot according to the needed lines
        plot_ap_bars(result, rangex, drawline, ordered_values_names)
        ap = np.asarray([result[key]["value"] for key in result])
        # maxy = max(maxy, np.around((np.amax(ap) + 0.15) * 10) / 10)
        maxy = max(maxy, np.around(np.amax(ap)))
        miny = min(miny, np.around(np.amin(ap)))
        ap_cat = results[key]['all']["value"]
        plt.plot([rangex[0], rangex[-1]], np.asarray([1, 1]) * ap_cat, linestyle='dashed', lw=1.5, color='black')
        firsttick[index] = rangex[0]
        xticks = np.concatenate((xticks, rangex[0:rangex.size:xtickstep]))

    maxy = max(maxy, 1)
    miny = min(miny, 0)
    axes = plt.gca()
    axes.set_ylim([miny, maxy])
    axes.set_xlim([0, rangex[-1] + 1])
    axes.tick_params(axis="x", labelsize=config["text_size"]["label"], labelrotation=-45)
    axes.tick_params(axis="y", labelsize=config["text_size"]["label"])

    property_values_names = np.tile(property_values_names, nobj)
    plt.xticks(xticks, property_values_names, ha="left")

    for index, key in enumerate(results):
        if key != "all":
            value = categories_display_names[key]["display_name"]
            if len(value) > 20:
                value = value[:2]
            plt.text(firsttick[index], maxy + 0.05, value, fontsize=config["text_size"]["label"])

    plt.title(title_str + "\n", fontdict={'horizontalalignment': 'center',
                                   'fontsize': config["text_size"]["title"]}, loc='center', pad=30)
    ylabel = metric.value.replace("_", " ")
    plt.ylabel(ylabel, fontdict=dict(size=config["text_size"]["label"]))
    if save_plot:
        plot_path = os.path.join(plot_path, title_str.replace('/', '_') + ".png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_models_comparison_on_sensitivity_impact(results, models, property_names, display_names, metric, output_path, save_plot):
    n_models = len(models)
    n_categories = len(results[list(results.keys())[0]].keys())
    n_properties = len(property_names)

    maxp = np.zeros((n_models, n_categories, n_properties))
    minp = np.zeros((n_models, n_categories, n_properties))

    for m, model in enumerate(results):
        for c, category in enumerate(results[model]):
            for p, prop in enumerate(property_names):
                if prop in results[model][category]:
                    maxp[m, c, p] = get_max_val(results[model][category][prop], "value", None)  # Max for this category
                    minp[m, c, p] = get_min_val(results[model][category][prop], "value", None)  # Min for this category

    maxval = np.mean(maxp, 1)
    minval = np.mean(minp, 1)

    avgval = np.zeros(len(models))
    for i, m in enumerate(results):
        avgval[i] = np.mean([results[m][cat]["all"]["value"] for cat in results[m]])
        if math.isnan(avgval[i]):
            avgval[i] = 0

    ##display graph
    plt.style.use('seaborn-whitegrid')
    fs = 14

    x_figsize = n_models * n_properties if (n_models * n_properties) > 6 else 7
    plt.figure(figsize=(x_figsize, 5), dpi=config["dpi"]["show"])



    xtickslab_pos = []
    p_names_pos = []
    for i, p in enumerate(property_names):
        start_pos = i * (n_models + 1) + 1

        asymetric_error = [(avgval - minval[:, i]), (maxval[:, i] - avgval)]

        plt.errorbar(x=np.arange(start_pos, start_pos + n_models), y=avgval,
                     yerr=asymetric_error,
                     ecolor='red',
                     ls='none', lw=1, marker='_', markersize=8, capsize=4, color='red')

        for i_m, pos in enumerate(np.arange(start_pos, start_pos + n_models)):
            plt.text(pos + 0.05, minval[i_m, i], s='{:.3f}'.format(minval[i_m, i]), fontsize=fs - 1)
            plt.text(pos + 0.05, maxval[i_m, i], s='{:.3f}'.format(maxval[i_m, i]), fontsize=fs - 1)

        p_names_pos.append(mean([start_pos, start_pos + n_models - 1]))


        if i < len(property_names) -1:
            plt.plot([start_pos + n_models, start_pos + n_models], [0, 1], linestyle='dashed', lw=1,
                     color='black')

        xtickslab_pos.extend(np.arange(start_pos, start_pos + n_models))

    xticklab = np.tile(list(models.keys()), n_properties)
    plt.xticks(xtickslab_pos, xticklab, ha="left", fontsize=fs, rotation=-45)

    axis_y_val = plt.gca().get_ylim()

    for i, n in enumerate(display_names):
        plt.text(p_names_pos[i], axis_y_val[1]+0.01, s=n, ha='center', fontsize=fs)

    plt.title('Sensitivity and Impact models comparison',
              fontdict={'horizontalalignment': 'center',
                        'fontsize': config["text_size"]["title"]}, loc='center', pad=30)

    ylabel = metric.value.replace("_", " ")
    plt.ylabel(ylabel, fontdict=dict(size=fs))
    path_to_save = os.path.join(output_path, 'comparison_plots_impact_strong.png')

    if save_plot:
        plt.savefig(path_to_save, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])

    plt.show()
    plt.close()


def display_sensitivity_impact_plot(results, output_path, property_names, display_names, metric, save, sort=True):
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

    # Average of metric score among categories (do not consider the different properties)
    avgval = np.mean([results[cat]["all"]["value"] for cat in results.keys()])
    if math.isnan(avgval):
        avgval = 0

    ##display graph
    plt.style.use('seaborn-whitegrid')
    fs = config["text_size"]["label"]

    # Labels to show
    xticklab = np.asarray(property_names)
    xticklab = xticklab[valid]

    x_figsize = np.sum(valid) + 2 if (np.sum(valid) + 2) > 6 else 7
    plt.figure(figsize=(x_figsize, 5), dpi=config["dpi"]["show"])
    plt.plot([1, len(property_names)], [avgval, avgval], linestyle='dashed', lw=1,
             color='black')

    if sort:
        si = sorted(range(len(maxval)), key=lambda k: (maxval[k] - minval[k]), reverse=False)
        tmp_minval, tmp_maxval, tmp_xticklab = [], [], []
        for i in si:
            tmp_minval.append(minval[i])
            tmp_maxval.append(maxval[i])
            tmp_xticklab.append(xticklab[i])

        minval = np.array(tmp_minval)
        maxval = np.array(tmp_maxval)
        xticklab = np.array(tmp_xticklab)

    asymetric_error = [(avgval - minval), (maxval - avgval)]
    plt.errorbar(x=np.arange(1, len(property_names) + 1), y=(avgval * np.ones(len(property_names))),
                 yerr=asymetric_error,
                 ecolor='red',
                 ls='none', lw=1, marker='_', markersize=8, capsize=4, color='red')

    for x in range(len(property_names)):
        plt.text(x + 1.05, minval[x], s='{:.3f}'.format(minval[x]), fontsize=fs - 1)
        plt.text(x + 1.05, maxval[x], s='{:.3f}'.format(maxval[x]), fontsize=fs - 1)
    plt.text(0.6, avgval, s='{:.3f}'.format(avgval), fontsize=fs)

    plt.xticks(np.arange(1, len(property_names) + 1), xticklab, ha="left", fontsize=fs, rotation=-45)

    axis_x_val = plt.gca().get_xlim()
    plt.gca().set_xlim([0.55, axis_x_val[1]])

    plt.title('Sensitivity and Impact',
              fontdict={'horizontalalignment': 'center',
                        'fontsize': config["text_size"]["title"]}, loc='center', pad=15)
    ylabel = metric.value.replace("_", " ")
    plt.ylabel(ylabel, fontdict=dict(size=fs))
    path_to_save = os.path.join(output_path, 'plots_impact_strong.png')

    if save:
        plt.savefig(path_to_save, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_iou_analysis(results, metric, category_display_name, save_plot, output_path):
    fig = go.Figure()
    title = "IoU analysis"

    for result in results:
        fig.add_trace(go.Scatter(x=results[result]["iou"], y=results[result]["metric_values"], name=category_display_name[result], mode="lines"))
        if len(results) == 1:
            title += f" - {category_display_name[result]}"

    yaxis_title = metric.value.replace("_", " ")
    fig.update_layout(
        xaxis_title="IoU",
        yaxis_title=yaxis_title,
        xaxis=dict(constrain='domain', dtick=0.1),
        title=title,
        legend_title='Categories',
        width=980, height=700,
    )
    if save_plot:
        plot_path = os.path.join(output_path, "IoU_analysis.png")
        fig.write_image(plot_path)
    fig.show()


def plot_cams(results, metric, display_name, save_plot, output_path):
    fig = go.Figure()

    title = "CAMs analysis"
    width = config["fig_size"]["plotly"]["legend"][0]
    height = config["fig_size"]["plotly"]["legend"][1]
    if len(results) == 1:
        key = list(results.keys())[0]
        title += " - {}".format(display_name[key])
        width = config["fig_size"]["plotly"]["no_legend"][0]
        height = config["fig_size"]["plotly"]["no_legend"][1]

    for result in results:
        fig.add_trace(go.Scatter(x=results[result]["threshold"], y=results[result]["metric_values"],
                                 name=display_name[result]))

    yaxis_title = metric.value.replace("_", " ")
    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title=yaxis_title,
        xaxis=dict(constrain='domain', dtick=0.1),
        title=title,
        width=width, height=height
    )
    if save_plot:
        plot_path = os.path.join(output_path, "{}.png".format(title))
        fig.write_image(plot_path)
    fig.show()


def plot_multiple_curves(results, curve, display_names, save_plot, output_path, legend_title="Categories"):
    fig = go.Figure()

    if curve == Curves.F1_CURVE:
        xaxis_title = "Thresholds"
        yaxis_title = "F1 scores"
        title = "F1 Curve"
    elif curve == Curves.PRECISION_RECALL_CURVE:
        xaxis_title = "Recall"
        yaxis_title = "Precision"
        title = "Precision-Recall Curve"
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=1, y1=0)
    elif curve == Curves.ROC_CURVE:
        xaxis_title = "False Positive Rate"
        yaxis_title = "True Positive Rate"
        title = "ROC Curve"
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
    else:
        raise NotImplementedError(f"Invalid curve: {curve}")

    if len(results) == 1:
        key = list(results.keys())[0]
        title += " - {} (AUC={:.2f})".format(display_names[key]['display_name'], results[key]['auc'])

    for category in results:
        auc_score = results[category]['auc']
        x_axis = results[category]['x']
        y_axis = results[category]['y']
        name = f"{display_names[category]['display_name']} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=x_axis, y=y_axis, name=name, mode='lines'))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        yaxis=dict(scaleanchor="x", scaleratio=1, dtick=0.1, fixedrange=False),
        xaxis=dict(constrain='domain', dtick=0.1),
        width=980, height=700,
        title=title,
        legend_title=legend_title
    )
    if save_plot:
        plot_path = os.path.join(output_path, f"{curve.value}.png")
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
        plot_path = os.path.join(output_path, f"Reliability_Confidence_analysis - {category.replace('/', '_')}.png")

    fig.update_layout(barmode='stack', bargap=0, width=1200, height=600, title_text=title)

    if save_plot:
        fig.write_image(plot_path)
    fig.show()


def display_co_occurrence_matrix(matrix, categories_labels, save_plot, output_path, title="Co-occurrence matrix", y_cats=None, x_cats=None):
    if len(categories_labels) < 5:
        fig_size = config["fig_size"]["cm"]["small"]
    elif len(categories_labels) < 15:
        fig_size = config["fig_size"]["cm"]["medium"]
    elif len(categories_labels) < 25:
        fig_size = config["fig_size"]["cm"]["large"]
    else:
        res_a, res_b = np.array_split(matrix, 2, axis=1)
        res1, res2 = np.array_split(res_a, 2)
        res3, res4 = np.array_split(res_b, 2)

        if y_cats is None:
            tmp_cats = categories_labels
        else:
            tmp_cats = y_cats
        s_split = len(tmp_cats) + 1 if len(tmp_cats) % 2 != 0 else len(tmp_cats)
        y_cats1 = tmp_cats[:int(s_split / 2)]
        y_cats2 = tmp_cats[int(s_split / 2):]
        if x_cats is None:
            tmp_cats = categories_labels
        else:
            tmp_cats = x_cats
        s_split = len(tmp_cats) + 1 if len(tmp_cats) % 2 != 0 else len(tmp_cats)
        x_cats1 = tmp_cats[:int(s_split / 2)]
        x_cats2 = tmp_cats[int(s_split / 2):]

        display_co_occurrence_matrix(res1, x_cats1, save_plot, output_path,
                                            title=title + "_1",
                                            y_cats=y_cats1, x_cats=x_cats1)
        display_co_occurrence_matrix(res2, x_cats2, save_plot, output_path,
                                            title=title + "_2",
                                            y_cats=y_cats2, x_cats=x_cats1)
        display_co_occurrence_matrix(res3, x_cats2, save_plot, output_path,
                                            title=title + "_3",
                                            y_cats=y_cats1, x_cats=x_cats2)
        display_co_occurrence_matrix(res4, x_cats2, save_plot, output_path,
                                            title=title + "_4",
                                            y_cats=y_cats2, x_cats=x_cats2)
        return
    if x_cats is None:
        x_cats = categories_labels
    if y_cats is None:
        y_cats = categories_labels

    matrix = matrix.astype(int)
    mask = np.identity(matrix.shape[0]) if x_cats == y_cats else None
    plt.style.use('default')
    plt.figure(figsize=fig_size)
    sns.set(font_scale=1.25)  # Adjust to fit
    sns.heatmap(matrix, annot=True, cmap="Blues", cbar=False, fmt="d", mask=mask)
    plt.gca().xaxis.set_ticklabels(x_cats, rotation=-45, ha="left")
    plt.gca().yaxis.set_ticklabels(y_cats, rotation=45, va="top")
    plt.title(title, weight='bold', fontdict={"size": config["text_size"]["title"]})
    if save_plot:
        plot_path = os.path.join(output_path, ".png".format(title))
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])

    plt.show()
    plt.close()


def display_confusion_matrix(results, categories_labels, properties, save_plot, output_path):
    results = results.astype(int)
    for index, category in enumerate(categories_labels):
        plt.style.use('default')
        figure, col = plt.subplots(1, 1, figsize=config["fig_size"]["cm"]["small"], dpi=config["dpi"]["show"])
        if properties is not None:
            filter_label = "\n"
            for p_name in properties.keys():
                filter_label += f"{p_name}: {properties[p_name]}\n"
            plt.text(1.33, 0.99, f"Filters applied:\n{filter_label}", horizontalalignment='left',
                     verticalalignment='top', fontdict={"size": config["text_size"]["cm"]}, transform=col.transAxes)

        plt.title(f"Confusion Matrix: {category}", weight='bold', fontdict={"size": config["text_size"]["title"]})

        sns.set(font_scale=1.25)  # Adjust to fit
        sns.heatmap(results[index], annot=True, ax=col, cmap="Blues", cbar=False, fmt="d")

        # Labels, title and ticks
        col.set_xlabel('Predicted labels', fontdict={'size': config["text_size"]["cm"]})
        col.set_ylabel('True labels', fontdict={'size': config["text_size"]["cm"]})
        col.tick_params(axis='both', which='major', labelsize=config["text_size"]["label"])
        col.xaxis.set_ticklabels([0, 1], rotation=0)
        col.yaxis.set_ticklabels([0, 1], rotation=0)

        if save_plot:
            plot_path = os.path.join(output_path, f"Confusion_Matrix_{category.replace('/', '_')}.png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
        plt.show()
        plt.close()


def display_confusion_matrix_categories(results, categories_labels, properties, save_plot, output_path, save_title=None,
                                        y_cats=None, x_cats=None):
    plt.style.use('default')
    if save_title is None:
        save_title = "Confusion_Matrix_categories"
    if len(categories_labels) < 5:
        fig_size = config["fig_size"]["cm"]["small"]
    elif len(categories_labels) < 15:
        fig_size = config["fig_size"]["cm"]["medium"]
    elif len(categories_labels) < 25:
        fig_size = config["fig_size"]["cm"]["large"]
    else:
        res_a, res_b = np.array_split(results, 2, axis=1)
        res1, res2 = np.array_split(res_a, 2)
        res3, res4 = np.array_split(res_b, 2)

        if y_cats is None:
            tmp_cats = categories_labels
        else:
            tmp_cats = y_cats
        s_split = len(tmp_cats) + 1 if len(tmp_cats) % 2 != 0 else len(tmp_cats)
        y_cats1 = tmp_cats[:int(s_split / 2)]
        y_cats2 = tmp_cats[int(s_split / 2):]
        if x_cats is None:
            tmp_cats = categories_labels
        else:
            tmp_cats = x_cats
        s_split = len(tmp_cats) + 1 if len(tmp_cats) % 2 != 0 else len(tmp_cats)
        x_cats1 = tmp_cats[:int(s_split / 2)]
        x_cats2 = tmp_cats[int(s_split / 2):]

        display_confusion_matrix_categories(res1, x_cats1, properties, save_plot, output_path, save_title=save_title + "_1",
                                            y_cats=y_cats1, x_cats=x_cats1)
        display_confusion_matrix_categories(res2, x_cats2, properties, save_plot, output_path, save_title=save_title + "_2",
                                            y_cats=y_cats2, x_cats=x_cats1)
        display_confusion_matrix_categories(res3, x_cats2, properties, save_plot, output_path, save_title=save_title + "_3",
                                            y_cats=y_cats1, x_cats=x_cats2)
        display_confusion_matrix_categories(res4, x_cats2, properties, save_plot, output_path, save_title=save_title + "_4",
                                            y_cats=y_cats2, x_cats=x_cats2)
        return

    figure, col = plt.subplots(1, 1, figsize=fig_size, dpi=config["dpi"]["show"])
    if properties is not None:
        filter_label = "\n"
        for p_name in properties.keys():
            filter_label += f"{p_name}: {properties[p_name]}\n"
        plt.text(1.33, 0.99, f"Filters applied:\n{filter_label}", horizontalalignment='left',
                 verticalalignment='top', fontdict={"size": config["text_size"]["cm"]}, transform=col.transAxes)

    plt.title(f"Confusion Matrix", weight='bold', fontdict={"size": config["text_size"]["title"]})

    sns.set(font_scale=1.25)  # Adjust to fit
    sns.heatmap(results, annot=True, ax=col, cmap="Blues", cbar=False, fmt="d")

    # Labels, title and ticks
    col.set_xlabel('Predicted labels', fontdict={'size': config["text_size"]["cm"]})
    col.set_ylabel('True labels', fontdict={'size': config["text_size"]["cm"]})
    col.tick_params(axis='both', which='major', labelsize=config["text_size"]["label"])

    if x_cats is None:
        col.xaxis.set_ticklabels(categories_labels, rotation=-45, ha="left")
        col.yaxis.set_ticklabels(categories_labels, rotation=45, va="top")
    else:
        col.xaxis.set_ticklabels(x_cats, rotation=-45, ha="left")
        col.yaxis.set_ticklabels(y_cats, rotation=45, va="top")

    if save_plot:
        plot_path = os.path.join(output_path, f"{save_title.replace('/', '_')}.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def display_top1_top5_error(results, labels, title, metric, output_path, save):
    fig = go.Figure(data=[
        go.Bar(name='Top-1', x=labels, y=np.array(results)[:, 0]),
        go.Bar(name='Top-5', x=labels, y=np.array(results)[:, 1])
    ])
    fig.update_yaxes(title_text=metric, range=[0, 1])
    fig.update_layout(barmode='group', width=800, height=500, title_text="<b>" + title + "</b>")
    if save:
        plot_path = os.path.join(output_path, f"{title}.png")
        fig.write_image(plot_path)
    fig.show()


def plot_models_comparison_on_property(results, models, category, property_name, property_values, metric, save_plot,
                                       output_path, split_title=None):
    nvalues = len(results) * len(models)
    split = False
    if nvalues < 11:
        plt.figure(figsize=config["fig_size"]["multi_cat"]["small"], dpi=config["dpi"]["show"])
    elif nvalues < 16:
        plt.figure(figsize=config["fig_size"]["multi_cat"]["medium"], dpi=config["dpi"]["show"])
    elif nvalues < 31:
        plt.figure(figsize=config["fig_size"]["multi_cat"]["large"], dpi=config["dpi"]["show"])
    else:
        split = True

    if split:
        if split_title is None:
            split_title = ""
        res1 = dict(list(results.items())[len(results) // 2:])
        res2 = dict(list(results.items())[:len(results) // 2])
        plot_models_comparison_on_property(res1, models, category, property_name, property_values[:len(res1)], metric, save_plot, output_path, split_title=split_title + " | a")
        plot_models_comparison_on_property(res2, models, category, property_name, property_values[len(res1):], metric, save_plot, output_path, split_title=split_title + " | b")

    else:

        xtickstep = 1
        drawline = True
        plt.style.use('seaborn-whitegrid')
        nobj = len(results.keys())
        rangex = np.zeros(1)
        maxy = 1
        xticks = np.asarray([])
        firsttick = np.zeros(nobj)

        for index, key in enumerate(results):
            result = {}
            for m_v in results[key].keys():
                    result[m_v] = results[key][m_v]

            nres = len(result)
            space = 0 if index == 0 else 1
            rangex = rangex[-1] + space + np.arange(1, nres + 1)  # expand range x in plot according to the needed lines
            plot_ap_bars(result, rangex, drawline)
            ap = np.asarray([result[key]['value'] for key in result])

            maxy = max(maxy, np.around((np.amax(ap) + 0.15) * 10) / 10)
            firsttick[index] = rangex[0]
            xticks = np.concatenate((xticks, rangex[0:rangex.size:xtickstep]))

        maxy = min(maxy, 1)
        axes = plt.gca()
        axes.set_ylim([0, maxy])
        axes.set_xlim([0, rangex[-1] + 1])
        axes.tick_params(axis="x", labelsize=config["text_size"]["label"], labelrotation=-45)
        axes.tick_params(axis="y", labelsize=config["text_size"]["label"])

        if split_title is None:
            title = f"Comparison on {property_name} property - {category}"
        else:
            title = f"Comparison on {property_name} property - {category}" + split_title
        models_names = np.tile(list(models.keys()), nobj)
        plt.xticks(xticks, models_names, ha="left")

        for index, key in enumerate(results):
            if key != "all":
                value = property_values[index]
                plt.text(firsttick[index], maxy + 0.05, value, fontsize=config["text_size"]["label"])

        plt.title(title + "\n", fontdict={'horizontalalignment': 'center',
                                              'fontsize': config["text_size"]["title"]}, loc='center', pad=30)
        ylabel = metric.value.replace("_", " ")
        plt.ylabel(ylabel, fontdict=dict(size=config["text_size"]["label"]))
        if save_plot:
            plot_path = os.path.join(output_path, title.replace('/', '_') + ".png")
            plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
        plt.show()
        plt.close()


def plot_models_comparison_on_error_impact(results, errors, category, metric, save_plot, output_path):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Errors impact', 'Errors count'), horizontal_spacing = 0.25)
    colors = plotly.colors.qualitative.Plotly
    for i, model in enumerate(results):
        fig.add_trace(
            go.Bar(name=model, x=np.array(results[model])[:, 0], y=errors, legendgroup=model, showlegend=False,
                   marker_color=colors[i], orientation='h'), row=1, col=1)
        fig.add_trace(
            go.Bar(name=model, x=np.array(results[model])[:, 1], y=errors, legendgroup=model, marker_color=colors[i],
                   orientation='h'),
            row=1, col=2)
    xaxis_title = metric.value.replace("_", " ")
    fig.update_xaxes(title_text=xaxis_title, row=1, col=1)
    fig.update_xaxes(title_text="count", row=1, col=2)
    fig.update_layout(barmode='group', width=900, height=500, title_text=f"<b>{category}</b>", font=dict(size=14))
    if save_plot:
        plot_path = os.path.join(output_path, f"Comparison_{category}_errors.png")
        fig.write_image(plot_path)
    fig.show()


def plot_models_comparison_on_tp_fp_fn_tn(results, labels, title, x_title, save_plot, output_path):
    fig = go.Figure()
    for model in results:
        fig.add_trace(go.Bar(name=model, x=labels, y=list(results[model].values())))
    fig.update_xaxes(title_text=x_title)
    fig.update_layout(barmode='group', width=800, height=500, title_text=f"<b>{title}</b>")
    if save_plot:
        plot_path = os.path.join(output_path, f"{title}.png")
        fig.write_image(plot_path)
    fig.show()


def plot_false_positive_trend(results, title, save_plot, output_path):
    fig = go.Figure()

    ordered = {}
    for e in results:
        ordered[e] = np.where(results[e] > 0)[0][0] if len(np.where(results[e] > 0)[0]) > 0 else 0

    ordered = dict(sorted(ordered.items(), key=lambda item: item[1]))

    colors = {"background": 'silver',
              "loc+class": 'orchid',
              "class": 'red',
              "loc+sim": 'lightgreen',
              "sim": 'yellow',
              "localization": 'lightskyblue',
              "duplicated": 'orange',
              "correct": 'seashell'}

    for error in ordered:
        fig.add_trace(go.Scatter(y=results[error]*100,
                                 mode='lines',
                                 stackgroup='one',
                                 line=dict(width=0.5, color=colors[error]),
                                 name=error))
    fig.update_yaxes(title_text="percentage of each type", range=[1, 100], ticksuffix='%', showgrid=False)
    x_axis_label = "total detections" if "correct" in results else "total false positives"
    fig.update_xaxes(title_text=x_axis_label, showgrid=False)
    fig.update_layout(title_text="<b>" + title + "</b>", plot_bgcolor="white")
    if save_plot:
        plot_path = os.path.join(output_path, f"{title}.png")
        fig.write_image(plot_path)
    fig.show()


def plot_seasonality_trend(results, title, save_plot, output_path):
    plt.style.use('default')
    subtitles = ['Observed', 'Trend', 'Seasonal', 'Residual']
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(15, 10))
    for i, r in enumerate(results):
        axs[i].plot(r.index, r.values)
        axs[i].set_title(subtitles[i])
    fig.suptitle(title)

    if save_plot:
        plt.savefig(output_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_threshold_analysis(results, title, save_plot, output_path):
    fig = go.Figure()
    for metric in results['y'].keys():
        fig.add_trace(go.Scatter(x=results["x"], y=results["y"][metric], mode="lines", name=metric.value))

    fig.update_layout(
        xaxis_title="Threshold",
        yaxis_title='score',
        yaxis_range=[0, 1.05],
        title=title
    )
    if save_plot:
        plot_path = os.path.join(output_path, f"{title}.png")
        fig.write_image(plot_path)
    fig.show()


def plot_distribution(data, bins, x_label, title, save_plot, output_path):
    plt.style.use('default')
    all_data = []
    for k in data:
        all_data.extend(data[k])
    min_v = min(all_data)
    max_v = max(all_data)
    if min_v > 0 or max_v < 0:
        sns.displot(data, bins=bins, multiple='stack', hue_order=['continuous', 'affected', 'generic'],
                    palette=['forestgreen', 'darkorange', 'royalblue'])
    else:
        perc = np.abs(min_v)/np.abs(min_v - max_v)
        left_bins = math.ceil(bins*perc)
        right_bins = bins - left_bins
        bin_w = max_v / right_bins
        min_range = -bin_w * left_bins
        sns.displot(data, binwidth=bin_w, binrange=(min_range, max_v), multiple='stack', hue_order=['continuous', 'affected', 'generic'],
                    palette=['forestgreen', 'darkorange', 'royalblue'])

    plt.title(title)
    plt.xlabel(x_label)
    if save_plot:
        plot_path = os.path.join(output_path, f"{title}.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_RUL_trend(data, save_plot, output_path):
    plt.style.use('default')
    np_data = np.array(data)

    result = {'optimal': np_data[(np_data>= -1) & (np_data <=0)],
              'warning': np_data[(np_data> -5) & (np_data <-1)],
              'bad': np_data[(np_data<= -5) | (np_data >0)]}
    g = sns.displot(result, binwidth=1, binrange=(round(min(data)), round(max(data))),
                palette=['green', 'orange', 'red'])
    g._legend.remove()
    plt.title('RUL variation distribution')
    plt.xlabel('RUL variation')
    if save_plot:
        plot_path = os.path.join(output_path, "RUL_trend_distribution.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_gain_chart(data, save_plot, output_path):
    x = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    values = np.array([0] + data['values'])

    plt.style.use('default')

    plt.plot(x, values*100, marker='o', label=data['model'])
    plt.plot(x, x, marker='o', label='random')
    plt.title("Gain chart")
    plt.xticks(x)
    plt.xlabel('% of data sets')
    plt.ylabel('% of positive samples')

    plt.legend(loc='lower right')
    plt.grid()

    if save_plot:
        plot_path = os.path.join(output_path, "Gain_chart.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_lift_chart(data, save_plot, output_path):
    x = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    values = np.array(data['values'])

    plt.style.use('default')

    plt.plot(x, values, marker='o', label=data['model'])
    plt.plot(x, [1]*10, marker='o', label='random')
    plt.title("Lift chart")
    plt.xticks(x)
    plt.xlabel('% of data sets')
    plt.ylabel('lift')

    plt.legend(loc='upper right')
    plt.grid()

    if save_plot:
        plot_path = os.path.join(output_path, "Lift_chart.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_predicted_vs_actual(data, save_plot, output_path, id_name=None):
    plt.style.use('default')

    min_v = []
    max_v = []
    for k in data.keys():
        plt.scatter(data[k]['predicted'], data[k]['actual'])
        min_v.append(min(data[k]['actual']))
        max_v.append(max(data[k]['actual']))

    plt.plot([min(min_v), max(max_v)], [min(min_v), max(max_v)], linestyle='--', color='black')
    plt.xlabel('Predicted RUL')
    plt.ylabel('Actual RUL')

    if id_name is None:
        title = "Predicted VS Actual"
    else:
        title = f"Predicted VS Actual_{str(id_name)}"

    plt.title(title)

    if save_plot:
        plot_path = os.path.join(output_path, f"{title}.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


def plot_regression_residuals(data, save_plot, output_path, id_name=None):
    plt.style.use('default')

    max_len = []
    for k in data.keys():
        plt.scatter(data[k]['predicted'], data[k]['residuals'])
        max_len.append(max(data[k]['predicted']))
    plt.plot([0]*round(max(max_len)), linestyle='--', color='black')
    plt.title("Residuals")
    plt.xlabel('Predicted RUL')
    plt.ylabel('Residuals')

    if id_name is None:
        title = "Residuals"
    else:
        title = f"Residuals_{str(id_name)}"

    plt.title(title)

    if save_plot:
        plot_path = os.path.join(output_path, f"{title}.png")
        plt.savefig(plot_path, facecolor='white', bbox_inches='tight', dpi=config["dpi"]["save"])
    plt.show()
    plt.close()


