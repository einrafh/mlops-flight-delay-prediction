import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Initialize environment variables
load_dotenv()

def fetch_flight_data():
    """
    Extracts raw flight data from the Aviationstack API and stores it as a JSON artifact.
    """
    api_key = os.getenv("AVIATIONSTACK_API_KEY")
    dep_iata = "SUB"
    arr_iata = "BDJ"
    
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Configuration Error: AVIATIONSTACK_API_KEY is missing from the environment variables.")

    url = "http://api.aviationstack.com/v1/flights"
    
    # Aligned the destination parameter with the target filename structure
    params = {
        'access_key': api_key,
        'dep_iata': dep_iata,
        'arr_iata': arr_iata,
    }

    print("Initiating data ingestion from Aviationstack API...")
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        output_dir = os.path.join("data", "raw")
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"flight_data_{dep_iata}_{arr_iata}_{timestamp}.json")
        
        with open(filename, 'w') as file_handler:
            json.dump(data, file_handler, indent=4)
            
        print(f"[SUCCESS] Raw data artifact generated at: {filename}")
        
        if 'pagination' in data:
            print(f"Total records retrieved: {data['pagination']['count']}")
        else:
            print("[WARNING] Pagination metadata is absent from the API payload.")
            
    except requests.exceptions.RequestException as network_error:
        print(f"[ERROR] Network communication failed: {network_error}")
    except Exception as runtime_error:
        print(f"[ERROR] An unexpected runtime exception occurred: {runtime_error}")

if __name__ == "__main__":
    fetch_flight_data()