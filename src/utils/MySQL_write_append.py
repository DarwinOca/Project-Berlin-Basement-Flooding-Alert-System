import pandas as pd

def write_precip_data(precip_ds, engine, table_name="precip_data"):
    """
    Transform precip_ds (xarray Dataset) into a DataFrame
    and write it to SQL in 10k row batches.

    Parameters
    ----------
    precip_ds : xarray.Dataset
        The precipitation dataset.
    engine : sqlalchemy.Engine
        SQLAlchemy engine connection.
    table_name : str, optional
        Name of the SQL table (default: "precip_data").
    """

    # Convert to DataFrame
    tmp_df = precip_ds.to_dataframe().reset_index()

    # Keep only date part in time
    tmp_df["time"] = tmp_df["time"].dt.date

    # Select relevant columns
    tmp_df = tmp_df[["time", "lat", "lon", "precipitation"]]
    tmp_df.rename(columns={"time": "date"}, inplace=True)

    print(f"Transformed DataFrame shape: {tmp_df.shape}")
    try:
        from IPython.display import display
        display(tmp_df.head())
    except ImportError:
        print(tmp_df.head())

    # Always create a fresh connection
    engine.dispose()

    # Write in 10k row batches, multi-row insert
    tmp_df.to_sql(
        table_name,
        engine,
        if_exists="replace",
        index=False,
        chunksize=10000,
        method="multi"
    )

    print(f"✅ Data written to table '{table_name}'")



from sqlalchemy.types import DateTime

def write_df_to_sql(df, engine, table_name, index=False, index_label=None, dtypes=None, chunksize=10000):
    """
    Write any DataFrame to SQL in batched inserts with clean connection handling.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to write.
    engine : sqlalchemy.Engine
        SQLAlchemy engine connection.
    table_name : str
        Name of the SQL table.
    index : bool, optional
        Whether to write the DataFrame index as a column. Default: False.
    index_label : str, optional
        Label for index column if index=True.
    dtypes : dict, optional
        Column dtype mapping (e.g., {"date": DateTime()}).
    chunksize : int, optional
        Number of rows per batch insert (default: 10,000).
    """

    print(f"Writing DataFrame to '{table_name}'...")
    print(f"Shape: {df.shape}")

    # Always dispose existing connection to avoid stale handles
    engine.dispose()

    # Write in batched inserts
    df.to_sql(
        table_name,
        con=engine,
        if_exists="replace",
        index=index,
        index_label=index_label,
        dtype=dtypes,
        chunksize=chunksize,
        method="multi"  # multiple rows per INSERT
    )

    print(f"✅ Successfully wrote to '{table_name}'")

