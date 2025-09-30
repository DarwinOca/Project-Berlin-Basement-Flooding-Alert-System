import datetime

def convert_date_format(date_string):
    """
    Converts a date string from 'YYYY-MM-DD' to 'DD.MM.YYYY' format.

    Args:
        date_string (str): The date string in 'YYYY-MM-DD' format.

    Returns:
        str: The converted date string in 'DD.MM.YYYY' format.
    """
    # Parse the input string into a datetime object
    date_object = datetime.datetime.strptime(date_string, '%Y-%m-%d')

    # Format the datetime object to the desired output string
    new_date_string = date_object.strftime('%d.%m.%Y')
    
    return new_date_string