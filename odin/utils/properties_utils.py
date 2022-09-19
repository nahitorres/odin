import matplotlib.pyplot as plt

def get_distribution_dict(array, name_mapping = dict(), list_of_possible_values = []):
    '''
    Given an array, extracts the distribution of values contained in it and returns a dictionary.
    
    Parameters
    ----------
    array : list
        The input array containing the values for which the function computes the distribution.
    name_mapping: dict
        This dict contains the names to assign to extracted meta-properties values in the output (e.g. {0: 'Monday', 1: 'Tuesday'...})
    list_of_possible_values: list
        An array containing the possible values contained in 'array'. If a value in this list is not present in 'array', its value is set to 0 in the output dictionary, while if a value is not present in either 'array' or 'list_of_possible_values' it will not be inserted in the output dictionary. The values refer to the initial array content, and not to the names after name mapping. Default: []
        
    Returns
    -------
    distribution_dict : dict
        The dictionary containing the distribution of values
    '''
    distribution_dict = dict()
    
    if len(list_of_possible_values) > 0:
        for item in list_of_possible_values:
            distribution_dict[item] = 0
    
    for item in array:           
        if (item in distribution_dict): 
            distribution_dict[item] += 1
        else: 
            distribution_dict[item] = 1
            
    if len(name_mapping):
        distribution_dict = dict((name_mapping[key], value) for (key, value) in distribution_dict.items())
            
    return distribution_dict


def get_temporal_metaproperties_distribution(observations):
    '''
    Applied on a dataset, extracts temporal distributions about its timestamps index.
    
    Parameters
    ----------
    observations: dataframe
        The dataframe containing a timestamp index and an 'anomalies' column in which 1 indicates an anomaly
    
    Returns
    -------
    hours_distribution : dict
        The dictionary containing the distribution of hours
        
    day_week_distribution : dict
        The dictionary containing the distribution of days of the week (0 = Monday, 6 = Sunday)
        
    day_month_distribution : dict
        The dictionary containing the distribution of days of the month
        
    month_distribution : dict
        The dictionary containing the distribution of month numbers (1-12)
    '''
    anomalous_df = observations[observations['anomaly'] == 1] # consider only the anomalous points
    timestamps = anomalous_df.index

    # extract hour of the day distribution
    hours = timestamps.hour.values
    hours_distribution = get_distribution_dict(hours)

    # extract day of the week distribution - Monday = 0 --> Sunday = 6
    day_week = timestamps.dayofweek.values
    day_week_distribution = get_distribution_dict(day_week, name_mapping = {0: 'Monday', 1: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'})

    # extract day of the month distribution
    day_month = timestamps.day.values
    day_month_distribution = get_distribution_dict(day_month)

    # extract month distribution
    month = timestamps.month.values
    month_distribution = get_distribution_dict(month)
        
    return hours_distribution, day_week_distribution, day_month_distribution, month_distribution


def plot_distribution(distribution_dict, title = ""):
    '''
    Given a dictioanry, plot the historgram representing the distribution defined in it
    '''
    
    plt.bar(list(distribution_dict.keys()), distribution_dict.values())
    plt.title(title)
    plt.show()