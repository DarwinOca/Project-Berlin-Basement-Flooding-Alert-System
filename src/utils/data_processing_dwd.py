def get_radolan_crop_params(min_lat, max_lat, min_lon, max_lon):
    """
    Calculate cropping parameters for RADOLAN data based on lat/lon bounds.
    
    Parameters:
    -----------
    min_lat, max_lat : float
        Latitude bounds
    min_lon, max_lon : float  
        Longitude bounds
        
    Returns:
    --------
    crop_params : dict
        Dictionary containing cropping parameters:
        - 'row_slice': slice object for rows
        - 'col_slice': slice object for columns
        - 'cropped_coords': cropped coordinate grid
    """
    import wradlib as wrl
    import numpy as np

    # Get coordinate grid
    coords = wrl.georef.get_radolan_grid(900, 900, wgs84=True)
    lats, lons = coords[..., 1], coords[..., 0]

    # Build combined mask for bounding box
    mask = (lats >= min_lat) & (lats <= max_lat) & \
        (lons >= min_lon) & (lons <= max_lon)

    # Find bounding rows and columns
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        print("No data found within specified bounds")
        return None

    # Determine slice bounds
    min_row, max_row = rows.min(), rows.max() + 1
    min_col, max_col = cols.min(), cols.max() + 1

    # Create slices
    row_slice = slice(min_row, max_row)
    col_slice = slice(min_col, max_col)

    # Get cropped coordinates
    cropped_coords = coords[row_slice, col_slice]

    # print(
    #     f"Crop bounds - Rows: {min_row}-{max_row}, Cols: {min_col}-{max_col}")
    # print(f"Cropped shape will be: ({max_row-min_row}, {max_col-min_col})")

    return {
        'row_slice': row_slice,
        'col_slice': col_slice,
        'cropped_coords': cropped_coords
    }


def apply_radolan_crop(data, crop_params):
    """
    Apply cropping to RADOLAN data using pre-calculated crop parameters.
    
    Parameters:
    -----------
    data : np.array
        RADOLAN precipitation data (900x900)
    crop_params : dict
        Cropping parameters from get_radolan_crop_params()
        
    Returns:
    --------
    cropped_data : np.array
        Cropped precipitation data
    """
    if crop_params is None:
        return None

    return data[crop_params['row_slice'], crop_params['col_slice']]


def crop_radolan_map(data, min_lat, max_lat, min_lon, max_lon):
    """
    Crop singular RADOLAN data to specified lat/lon bounds.
    
    This is a convenience function that combines get_radolan_crop_params 
    and apply_radolan_crop for single-use cropping. 
    It is only relevant for visualization purposes.
    
    Parameters:
    -----------
    data : np.array
        RADOLAN precipitation data (900x900)
    min_lat, max_lat : float
        Latitude bounds
    min_lon, max_lon : float  
        Longitude bounds
        
    Returns:
    --------
    cropped_data : np.array
        Cropped precipitation data
    cropped_coords : np.array
        Corresponding coordinate grid (cropped_rows, cropped_cols, 2)
    row_indices : tuple
        (min_row, max_row) indices used for cropping
    col_indices : tuple
        (min_col, max_col) indices used for cropping
    """
    crop_params = get_radolan_crop_params(min_lat, max_lat, min_lon, max_lon)
    if crop_params is None:
        return None, None, None, None

    cropped_data = apply_radolan_crop(data, crop_params)

    return cropped_data, crop_params['cropped_coords']


def create_radolan_timeseries(config):
    """
    Create a time series of cropped RADOLAN data.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing:
        - 'bounds': dict with 'min_lat', 'max_lat', 'min_lon', 'max_lon'
        - 'date_range': dict with 'start_date', 'end_date'
        - 'region_name': str
        - 'output_directory': str (optional)
        - 'data_directory': str (optional)
    """
    bounds = config['bounds']
    date_range = config['date_range']

    # Extract parameters
    min_lat = bounds['min_lat']
    max_lat = bounds['max_lat']
    min_lon = bounds['min_lon']
    max_lon = bounds['max_lon']
    start_date = date_range['start_date']
    end_date = date_range['end_date']
    region_name = config['region_name']
    output_directory = config.get('output_directory', '../data/dwd/processed')
    data_directory = config.get('data_directory', '../data/dwd/extracted')

    # Get RADOLAN data filenames for the specified date range
    import os
    import xarray as xr
    import numpy as np
    from radolan_handler import RadolanFileHandler

    # Initialize file handler to get available files
    file_handler = RadolanFileHandler(data_directory=data_directory)
    filenames = file_handler.get_filenames_in_date_range(start_date, end_date)

    if not filenames:
        print(
            f"No RADOLAN files found for date range {start_date} to {end_date}"
        )
        return None

    # Get crop parameters
    crop_params = get_radolan_crop_params(min_lat, max_lat, min_lon, max_lon)

    if crop_params is None:
        print("Failed to determine crop parameters")
        return None

    # Get cropped coordinates - full 2D arrays
    cropped_coords = crop_params['cropped_coords']
    lons_2d = cropped_coords[..., 0].astype(
        np.float32)  # Ensure float32 for zarr compatibility
    lats_2d = cropped_coords[..., 1].astype(
        np.float32)  # Ensure float32 for zarr compatibility

    # Create projected coordinate arrays (RADOLAN grid indices)
    n_lats, n_lons = cropped_coords.shape[:2]

    # Initialize data array with float32 for better zarr compatibility
    n_times = len(filenames)
    timeseries_data = np.zeros((n_times, n_lats, n_lons), dtype=np.float32)

    # Extract dates from filenames for time coordinate
    dates = []

    # Process each file
    print(f"Processing {n_times} RADOLAN files...")
    for i, filename in enumerate(filenames):
        # Load and crop data
        radolan_data = file_handler.load_radolan_data(filename)
        cropped_data = apply_radolan_crop(radolan_data, crop_params)
        # Ensure float32 and handle NaN values properly
        timeseries_data[i] = cropped_data.astype(np.float32)

        # Extract date from filename
        date = file_handler.extract_date_from_filename(filename)
        dates.append(date)

        if (i + 1) % 50 == 0:
            print(f"Progress: processed {i + 1}/{n_times} files")

    # Create time coordinate - ensure datetime64[ns] for zarr compatibility
    time_coord = np.array(dates, dtype='datetime64[ns]')

    # Create xarray DataArray with zarr-compatible data types
    data_array = xr.DataArray(timeseries_data,
                              dims=['time', 'x', 'y'],
                              coords={
                                  'time': ('time', time_coord),
                                  'lat': (['x', 'y'], lats_2d),
                                  'lon': (['x', 'y'], lons_2d),
                              },
                              name='precipitation')

    # Add attributes
    data_array.attrs = {
        'long_name': 'Precipitation rate',
        'units': 'mm/h',
        'description': 'RADOLAN precipitation time series',
        'bounds':
        f'lat: {min_lat:.2f}-{max_lat:.2f}, lon: {min_lon:.2f}-{max_lon:.2f}',
        'date_range': f'{start_date} to {end_date}',
        'projection': 'RADOLAN polar stereographic'
    }

    # Add coordinate attributes
    data_array.coords['lon'].attrs = {
        'long_name': 'longitude',
        'units': 'degrees_east'
    }
    data_array.coords['lat'].attrs = {
        'long_name': 'latitude',
        'units': 'degrees_north'
    }

    print(f"Created time series with shape: {timeseries_data.shape}")
    print(
        f"Time range: {np.datetime_as_string(time_coord[0], unit = 'D')} to {np.datetime_as_string(time_coord[-1], unit = 'D')}"
    )

    # Create zarr filename and full path
    zarr_filename = f"radolan_{region_name}_{start_date}_{end_date}.zarr"
    zarr_path = os.path.join(output_directory, zarr_filename)

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Save to zarr
    data_array.to_dataset(name="precipitation").to_zarr(zarr_path,
                                                        mode="w",
                                                        consolidated=True,
                                                        zarr_format=2)
    print(f"Successfully saved RADOLAN time series to: {zarr_path}")
