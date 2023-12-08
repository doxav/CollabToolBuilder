
def get_weather(latitude, longitude, unit='c'):
    """ Get the current weather at the given latitude and longitude
    :param latitude: Latitude of the location
    :param longitude: Longitude of the location
    :param unit: Unit of the temperature (f or c)
    :return: A string with the current temperature
    """
    # Example usage:
    # weather_data = get_weather(37.7749, -122.4194, "f")
    # print(weather_data)
    import requests
    import datetime
    api_url = "https://api.open-meteo.com/v1/forecast"
    temp_unit='celsius'
    kmh_unit='kmh'
    precipitation_unit='mm'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'temperature_unit': temp_unit,
        'wind_speed_unit': kmh_unit,
        'precipitation_unit': precipitation_unit,
        'hourly': ['temperature_2m','wind_speed_10m','wind_direction_10m','precipitation','visibility'],
        'forecast_days': 1,
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        results = response.json()
    except requests.exceptions.HTTPError as errh:
        print ("Http Error:",errh)
    except requests.exceptions.ConnectionError as errc:
        print ("Error Connecting:",errc)
    except requests.exceptions.Timeout as errt:
        print ("Timeout Error:",errt)
    except requests.exceptions.RequestException as err:
        print ("OOps: Something Else",err)

    current_utc_time = datetime.datetime.utcnow()
    # print(results)
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']
    
    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]
    
    return f'The current temperature is {current_temperature}°{temp_unit}, wind speed is {results["hourly"]["wind_speed_10m"][closest_time_index]} {kmh_unit}, wind direction is {results["hourly"]["wind_direction_10m"][closest_time_index]}°, precipitation is {results["hourly"]["precipitation"][closest_time_index]} {precipitation_unit}, visibility is {results["hourly"]["visibility"][closest_time_index]} m.'

def fetch_GDELT_events_data_FR(start_date, end_date, keywords_list, country="FR"):
    """ Fetches GDLET events data from a specified area and time period.
    :param start_date: The start date of the time period to fetch data for
    :param end_date: The end date of the time period to fetch data for
    :param keywords_list: A list of keywords to search for in the event data
    :param country: The country to fetch data for (optional, default: "FR")
    :return: A Pandas DataFrame containing the results
    """
    # import gdeltdoc-doc, if gdeltdoc package does not exist pip install gdeltdoc
    try:
        from gdeltdoc import GdeltDoc, Filters
    except ImportError:
        print("The gdelt_doc_api package is not installed. Installing...")
        import subprocess
        subprocess.check_call(["python", '-m', 'pip', 'install', 'gdeltdoc'])
        from gdeltdoc import GdeltDoc, Filters

    # Initialize GDELT API
    gdelt = GdeltDoc()

    # check if near_keywords_list is an array of strings, if a list or a dict, convert to array of strings
    if isinstance(keywords_list, list):
        keywords_list = [str(item) for item in keywords_list]
    elif isinstance(keywords_list, dict):
        keywords_list = [str(item) for item in keywords_list.values()]

    if len(keywords_list) == 1:
        keywords_list = keywords_list[0]
    
    print(keywords_list)
    # Define filters
    myfilter = Filters(start_date=start_date, end_date=end_date, keyword=keywords_list, country="FR")

    try:
        # Fetch data using GDELT API
        results = gdelt.article_search(myfilter)[['url', 'title', 'seendate', 'sourcecountry']]
        return results
    except Exception as e:
        print(f"An error occurred: {e}")
        return None