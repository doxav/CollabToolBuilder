import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.default_functions import fetch_GDELT_events_data_FR

print(fetch_GDELT_events_data_FR("2021-01-01", "2021-01-02", ["maritime", "pollution"]))