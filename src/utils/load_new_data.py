

def load_new_data_from_dwd(start_date, end_date):
    """Load new data from DWD using the utils/data_loading_dwd.py script.
    This function assumes that the utils/ directory is in the parent directory.
    """

    import importlib
    from utils import data_loading_dwd

    importlib.reload(data_loading_dwd)
    # importlib.reload(radolan_handler)


    # Keep raw files (recommended for production)
    results = data_loading_dwd.import_radolan_recent(start_date,
                                                    end_date,
                                                    '../data/dwd/',
                                                    keep_raw=True)


    # Process raw files to timeseries for a specific region
    from utils import data_processing_dwd
    importlib.reload(data_processing_dwd)

    # Example usage with Berlin bounds
    config = {
        'bounds': {
            'min_lat': 52.4,
            'max_lat': 52.65,
            'min_lon': 13.15,
            'max_lon': 13.6
        },
        'date_range': {
            'start_date': start_date, # '2022-01-01',
            'end_date': end_date # '2025-04-30'
        },
        'region_name': 'berlin',
        'data_directory': '../data/dwd/extracted',
        'output_directory': '../data/dwd/processed'
    }

    data_processing_dwd.create_radolan_timeseries(config)



def load_new_data_from_wasserportal(start_date, end_date):
    """Load new data from Wasserportal using the utils/data_loading_wasserportal.py script.
    This function assumes that the utils/ directory is in the parent directory.
    """

    import importlib
    from utils import data_loading_wasserportal

    importlib.reload(data_loading_wasserportal)
    # Usage example:
    config_groundwater = {
        'thema': 'gws',
        'exportthema': 'gw',
        'sreihe': 'ew',
        "anzeige": "d",
        "smode": "c"
    }

    data_loading_wasserportal.get_multi_station_data(start_date,
                                                    end_date,
                                                    config_groundwater,
                                                    num_stations=None)


